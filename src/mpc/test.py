import pickle
import yaml
import numpy as np
import logging
import math
import copy
import time

from .user import MPCUser
from .ttp import TrustThirdParty
from .server import MPCServer


class Test:
    def __init__(self, config_file="./config.yaml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self.n = math.ceil(config["n"] * config["r"])
        self.t = config["t"]

        with open("../data/small/user_indices.pkl", "rb") as f:
            self.user_indices = pickle.load(f)
            self.user_index = {
                u: np.int32(i)
                for u, i in zip(self.user_indices.keys(), range(len(self.user_indices)))
            }
        # self.n = 10
        # self.t = 3
        # self.user_index = {i: i for i in range(self.n)}

        logging.debug("----------- init !!! -----------")
        logging.debug("start init trusted third party")
        self.ttp = TrustThirdParty()
        logging.debug("end init trusted third party")

        logging.debug("start init mpc server")
        self.server = MPCServer(self.n, self.t)
        logging.debug("end init mpc server")

        logging.debug("start generate sign keys for users")
        self.ttp.gen_keys(list(self.user_index.values()))
        logging.debug("end generate sign keys for users")

        logging.debug("start init MPCUser class")
        MPCUser.initialize(self.n, self.t, self.server, self.ttp)
        logging.debug("end init MPCUser class")
        MPCUser.get_dh_keys()

    def start(self, model=None):
        # param_num = 6000 * 400
        # for name, param in model.named_parameters():
        #     param_num += param.numel()
        param_num = 1604402
        logging.debug(f"exchange {param_num} params")

        raw_vecs = {}
        user_instances = {}
        raw_vec = {"test": np.random.randn(param_num)}
        time1 = time.clock()
        logging.debug(f"{time1} ----------- start !!! -----------")
        for uid, uindex in self.user_index.items():
            raw_vecs[uindex] = raw_vec
            user_instance = MPCUser(uindex, raw_vecs[uindex])
            user_instances[uindex] = user_instance
            user_instance.send_pub_keys()

        logging.debug("----------- end !!! -----------")
        # check
        truth = {}
        name = "test"
        truth[name] = sum([raw_vecs[u][name] for u in raw_vecs])
        print(truth[name] == self.server.unmask_vecs[name])
        print(truth[name], self.server.unmask_vecs[name])

