from Crypto.Protocol.SecretSharing import Shamir
from cryptography.hazmat.primitives import padding


class ShamirL(Shamir):
    @staticmethod
    def split(k, n, secret, ssss=False):
        if len(secret) == 16:
            return Shamir.split(k, n, secret, ssss)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(secret)
        padded_data += padder.finalize()
        secret = padded_data

        rslt = None
        for i in range(0, len(secret), 16):
            secret_i = secret[i : i + 16]
            split_i = Shamir.split(k, n, secret_i, ssss)
            if rslt is None:
                rslt = [list(i) for i in split_i]
            else:
                for index, s in split_i:
                    rslt[index - 1][1] += s

        return rslt

    @staticmethod
    def combine(shares):
        if len(shares[0][1]) == 16:
            return Shamir.combine(shares)

        rslt = b""
        for i in range(0, len(shares[0][1]), 16):
            shares_i = [(j + 1, shares[j][1][i : i + 16]) for j in range(len(shares))]
            rslt_i = Shamir.combine(shares_i)
            rslt += rslt_i

        unpadder = padding.PKCS7(128).unpadder()
        rslt = unpadder.update(rslt)

        return rslt

