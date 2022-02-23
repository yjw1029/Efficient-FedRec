import torch
import torch.optim as optim

from agg.base import BaseAggregator

class UserAggregator(BaseAggregator):
    '''Aggregate per-user gradiant'''
    def _init_grad_param(self):
        self.user_grad = {}

        self.news_grads = {}
        self.user_optimizer.zero_grad()
        self.news_optimizer.zero_grad()

    def update(self, all_sample_num):
        self.update_user_grad(all_sample_num)
        self.update_news_grad()
        self._init_grad_param()

    def update_user_grad(self, all_sample_num):
        for name, param in self.user_encoder.named_parameters():
            if param.requires_grad:
                param.grad = torch.sum(
                    self.user_grad[name] / all_sample_num,
                    dim=0,
                ).cuda()
        self.user_optimizer.step()

    def collect(self, user_grad, news_grad):
        for name in user_grad:
            if name not in self.user_grad:
                self.user_grad[name] = user_grad[name]
            else:
                self.user_grad[name] += user_grad[name]

        # update news model params
        for nid in news_grad:
            if nid in self.news_grads:
                self.news_grads[nid] += news_grad[nid]
            else:
                self.news_grads[nid] = news_grad[nid]

