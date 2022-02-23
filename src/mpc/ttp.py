from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class TrustThirdParty:
    def __init__(self):
        pass

    def gen_keys(self, user_list):
        self.d_prs = {}
        for u in user_list:
            self.d_prs[u] = Ed25519PrivateKey.generate()
