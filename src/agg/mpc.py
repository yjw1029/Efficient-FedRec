import torch

from agg.user import UserAggregator
from mpc import MPCUser

class MPCAggregator(UserAggregator):
    def _init_grad_param(self):
        self.user_grad = {}      
        self.news_grads = {}
        self.all_sample_num = None
        self.user_optimizer.zero_grad()
        self.news_optimizer.zero_grad()

    def collect(self, user_grad, user_sample_num, news_grad, union_nid_index):
        nindex2nid = {v: k for k, v in union_nid_index.items()}
        for uindex in range(len(user_sample_num)):
            grad = {}
            for name in user_grad:
                grad[name] = user_grad[name][uindex]

            grad["news_embedding"] = news_grad[uindex]
            grad["sample_num"] = user_sample_num[uindex]         

            user_instance = MPCUser(uindex, grad)
            user_instance.send_pub_keys()
        
        agged_grad = MPCUser.server.unmask_vecs
        for nindex in nindex2nid:
            nid = nindex2nid[nindex]
            if nid in self.news_grads:
                self.news_grads[nid] += agged_grad["news_embedding"][nindex]
            else:
                self.news_grads[nid] = agged_grad["news_embedding"][nindex]

        if self.all_sample_num is None:
            self.all_sample_num = agged_grad["sample_num"].long()
        else:
            self.all_sample_num += agged_grad["sample_num"].long()

        for name, param in self.user_encoder.named_parameters():
            if param.grad is None:
                param.grad = agged_grad[name].float().cuda()
            else:
                param.grad += agged_grad[name].float().cuda()

        MPCUser.server.clear()

    def update_user_grad(self, all_sample_num):
        for name, param in self.user_encoder.named_parameters():
            if param.requires_grad:
                param.grad = param.grad / all_sample_num
        self.user_optimizer.step()