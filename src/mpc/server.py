from .utils import ShamirL as Shamir

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

import copy
import logging
import numpy as np
from collections import defaultdict
import time 


class MPCServer:
    def __init__(self, n, t):
        self.n = n
        self.t = t

        self.hash_digest = hashes.Hash(hashes.MD5())
        self.clear()

    def clear(self):
        self.user_instances = {}
        self.recv_pub_keys = {}
        self.recv_enc_shares = {}
        self.recv_enc_shares_cnt = 0

        self.masked_vecs = {}
        self.u3 = []

        self.recv_s_prs = defaultdict(list)
        self.recv_bs = defaultdict(list)

    def collect_pub_keys(self, uid, user_instance, key):
        self.user_instances[uid] = user_instance
        self.recv_pub_keys[uid] = key

        if len(self.recv_pub_keys) == self.n:
            time1 = time.clock()
            logging.debug(f"[+] {time1} collect {self.n} public keys, start broadcast")
            self.broadcast_pub_keys()

    def broadcast_pub_keys(self):
        for uid in self.user_instances:
            self.user_instances[uid].recv_pub_keys(self.recv_pub_keys)
            self.user_instances[uid].send_secret_share()

    def collect_enc_shares(self, src_uid, enc_shares):
        if src_uid not in self.user_instances:
            return
        self.recv_enc_shares_cnt += 1

        for dst_uid in enc_shares:
            if dst_uid not in self.recv_enc_shares:
                self.recv_enc_shares[dst_uid] = {}
            self.recv_enc_shares[dst_uid][src_uid] = enc_shares[dst_uid]

        # TODO: when receive at least t (now receive all)
        if self.recv_enc_shares_cnt == self.n:
            time1 = time.clock()
            logging.debug(f"[+] {time1} collect {self.n} enc shares, start sending shares")
            self.send_secret_shares()

    def send_secret_shares(self):
        for uid in self.recv_enc_shares:
            self.user_instances[uid].receive_enc_shares(self.recv_enc_shares[uid])
            self.user_instances[uid].send_masked_vecs()

    def collect_masked_vecs(self, uid, masked_vecs):
        self.u3.append(uid)
        if self.masked_vecs == {}:
            self.masked_vecs = copy.deepcopy(masked_vecs)
        else:
            for name in masked_vecs:
                self.masked_vecs[name] += masked_vecs[name]

        if len(self.u3) == self.n:
            time1 = time.clock()
            logging.debug(f"[+] {time1} collect {self.n} masked_vecs, start consistensy check")
            self.consist_check()

    def consist_check(self):
        uid_list = self.u3
        user_checks = {}
        for uid in self.user_instances:
            user_checks[uid] = self.user_instances[uid].consist_check(uid_list)

        time1 = time.clock()
        logging.debug(f"[-] {time1} end consistensy check")

        for uid in self.user_instances:
            send_s_prs, send_bs = self.user_instances[uid].unmask(user_checks)

            for uid in send_s_prs:
                self.recv_s_prs[uid].append(send_s_prs[uid])

            for uid in send_bs:
                self.recv_bs[uid].append(send_bs[uid])

        time1 = time.clock()
        logging.debug(f"[+] {time1} start unmask")
        self.unmask()
        time1 = time.clock()
        logging.debug(f"[-] {time1} end unmask")

    def unmask(self):
        for vid in self.recv_s_prs:
            shares = self.recv_s_prs[vid]
            shares = [(i + 1, v) for i, v in zip(range(len(shares)), shares)]
            self.recv_s_prs[vid] = Shamir.combine(shares)

        for uid in self.recv_bs:
            shares = self.recv_bs[uid]
            shares = [(i + 1, v) for i, v in zip(range(len(shares)), shares)]
            self.recv_bs[uid] = Shamir.combine(shares)

        for uid in self.u3:
            seed = int.from_bytes(self.recv_bs[uid], "big")
            rng = np.random.default_rng(seed)
            for name, vec in self.masked_vecs.items():
                self.masked_vecs[name] -= rng.random(vec.shape)

            s_pb_u_der, _, _ = self.recv_pub_keys[uid]
            s_pb_u = X25519PublicKey.from_public_bytes(s_pb_u_der)

            for vid in self.recv_s_prs:
                print(self.recv_s_prs)
                s_pr_v_der = self.recv_s_prs[vid]
                s_pr_v = X25519PrivateKey.from_private_bytes(s_pr_v_der)
                s_shared_key = s_pr_v.exchange(s_pb_u)
                digest = self.hash_digest.copy()
                s_hashed_key = digest.update(s_shared_key).finalize()
                seed = int.from_bytes(s_hashed_key, "big")
                rng = np.random.default_rng(seed)
                sign = 1 if vid > uid else -1
                for name, vec in self.masked_vecs.items():
                    self.masked_vecs[name] += rng.random(vec.shape) * sign

        self.unmask_vecs = self.masked_vecs
        return self.unmask_vecs
