from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


import time
import pickle
import logging
import numpy as np
import os
import copy
from functools import reduce

from .utils import ShamirL as Shamir


class MPCUser:
    """ Implement secere aggreagtion proposed in https://eprint.iacr.org/2017/281.pdf
    """

    @classmethod
    def initialize(cls, n, t, server, ttp):
        cls.n = n
        cls.t = t
        cls.server = server
        cls.ttp = ttp
        cls.hash_digest = hashes.Hash(hashes.MD5())

    @classmethod
    def get_dh_keys(cls):
        cls.d_prs = cls.ttp.d_prs

    def __init__(self, uid, raw_vecs):
        """
        Args:
            uid: user id
            raw_vecs (dict): (name, vecs)
        """
        # init self pubkey prikey for df and cert
        self.uid = uid
        self.raw_vecs = raw_vecs

        self.s_pr = X25519PrivateKey.generate()
        self.s_pb = self.s_pr.public_key()

        self.c_pr = X25519PrivateKey.generate()
        self.c_pb = self.c_pr.public_key()
        end = time.time()

    def ser_pr_key(self, private_key):
        der = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return der

    def ser_pb_key(self, public_key):
        der = public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
        )
        return der

    def send_pub_keys(self):
        s_pb_der = self.ser_pb_key(self.s_pb)
        c_pb_der = self.ser_pb_key(self.c_pb)
        digest = self.hash_digest.copy()
        digest.update(s_pb_der + c_pb_der)
        pb_der = digest.finalize()
        return self.server.collect_pub_keys(
            self.uid, self, (s_pb_der, c_pb_der, self.d_prs[self.uid].sign(pb_der))
        )

    def recv_pub_keys(self, public_keys):
        if not isinstance(public_keys, dict):
            raise ValueError("public_keys must be a dict")

        self.shared_c_keys = {}
        self.s_pb_keys = {}
        for uid, key_value in public_keys.items():
            if uid == self.uid:
                continue
            s_pb_der, c_pb_der, sign = key_value

            s_pb = X25519PublicKey.from_public_bytes(s_pb_der)
            c_pb = X25519PublicKey.from_public_bytes(c_pb_der)

            digest = self.hash_digest.copy()
            digest.update(s_pb_der + c_pb_der)
            pb_der = digest.finalize()

            try:
                self.d_prs[uid].public_key().verify(sign, pb_der)
            except InvalidSignature:
                raise ValueError(f"{uid} public key contains invalid signature")

            c_shared_key = self.c_pr.exchange(c_pb)
            digest = self.hash_digest.copy()
            c_hashed_key = digest.update(c_shared_key)
            c_hashed_key = digest.finalize()
            self.shared_c_keys[uid] = c_hashed_key

            self.s_pb_keys[uid] = s_pb

    def send_secret_share(self):
        """ round 1: Sent shamir shares to server 
        """
        self.b = os.urandom(16)

        self.b_shares = Shamir.split(self.t, self.n, self.b)
        self.pr_shares = Shamir.split(self.t, self.n, self.ser_pr_key(self.s_pr))

        self.enc_shares = {}
        for cnt, uid in enumerate(self.shared_c_keys):
            iv = os.urandom(16)
            key = self.shared_c_keys[uid]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            pt = (
                np.int32(self.uid).tobytes()
                + np.int32(uid).tobytes()
                + self.b_shares[cnt][1]
                + self.pr_shares[cnt][1]
            )

            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(pt)
            padded_data += padder.finalize()
            pt = padded_data

            self.enc_shares[uid] = iv + encryptor.update(pt) + encryptor.finalize()

        return self.server.collect_enc_shares(self.uid, self.enc_shares)

    def receive_enc_shares(self, enc_shares):
        """ round2: Receive shamir shares from at least t users
        Args: 
            enc_share (dict): (key, value) is (user id, encrypted enc share) 
        """
        if len(enc_shares) < self.t:
            raise ValueError(f"Receive {len(enc_shares)} less than {self.t} shares")

        self.recv_shares = enc_shares
        self.shared_s_keys = {}

        for uid in enc_shares:
            if uid == self.uid:
                continue
            s_pb = self.s_pb_keys[uid]
            s_shared_key = self.s_pr.exchange(s_pb)
            digest = self.hash_digest.copy()
            digest.update(s_shared_key)
            s_hashed_key = digest.finalize()
            self.shared_s_keys[uid] = s_hashed_key

        return self.shared_s_keys

    def send_masked_vecs(self):
        """ round2: Generate masked vectors for mpc
        """
        masked_vecs = {}
        for uid, key in self.shared_s_keys.items():
            seed = int.from_bytes(key, "big")
            rng = np.random.default_rng(seed)
            sign = 1 if self.uid > uid else -1
            for name, vec in self.raw_vecs.items():
                # random * 10 to enable int level perturbation
                if name not in masked_vecs:
                    masked_vecs[name] = (
                        self.raw_vecs[name] + rng.random(vec.shape) * 10 * sign
                    )
                else:
                    masked_vecs[name] += rng.random(vec.shape) * 10 * sign

        seed = int.from_bytes(self.b, "big")
        rng = np.random.default_rng(seed)
        for name, vec in masked_vecs.items():
            masked_vecs[name] += rng.random(vec.shape)
        return self.server.collect_masked_vecs(self.uid, masked_vecs)

    def consist_check(self, uid_list):
        self.uid_list = uid_list
        if len(uid_list) < self.t:
            raise ValueError(f"User list length is {len(uid_list)} less than {self.t}")

        byte_uid_list = list(map(lambda x: np.int32(x).tobytes(), uid_list))
        byte_uid_list = reduce(lambda x, y: x + y, byte_uid_list)

        digest = self.hash_digest.copy()
        digest.update(byte_uid_list)
        self.uid_list_hash = digest.finalize()
        sign = self.d_prs[self.uid].sign(self.uid_list_hash)

        return sign

    def unmask(self, user_checks):
        # verify
        if len(user_checks) < self.t:
            raise ValueError(
                f"User checks num is {len(user_checks)} less than {self.t}"
            )

        for uid, check in user_checks.items():
            if uid not in self.uid_list:
                raise ValueError(f"user {uid} is not in the uid_list (U3)")

            try:
                self.d_prs[uid].public_key().verify(check, self.uid_list_hash)
            except InvalidSignature:
                raise ValueError(f"user {uid} has invalid consistency check")

        send_s_prs = {}
        send_bs = {}

        for uid in self.recv_shares:
            if uid == self.uid:
                continue

            enc_share = self.recv_shares[uid]
            iv = enc_share[:16]
            key = self.shared_c_keys[uid]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            share = decryptor.update(enc_share[16:]) + decryptor.finalize()

            unpadder = padding.PKCS7(128).unpadder()
            share = unpadder.update(share)

            uid_share = np.frombuffer(share[:4], dtype=np.int32)[0]
            self_uid_share = np.frombuffer(share[4:8], dtype=np.int32)[0]

            if uid_share != uid or self_uid_share != self.uid:
                raise ValueError("Umask uid check failed")

            if uid in self.uid_list:
                send_bs[uid] = share[8:24]
            else:
                send_s_prs[uid] = share[24:]

        return send_s_prs, send_bs


if __name__ == "__main__":
    MPCUtil.initialize(10, 2)
    mpc_util = MPCUtil(1)
    mpc_util.send_pub_keys()
