import random
import wandb
from opacus import GradSampleModule

import torch
from torch.utils.data import DataLoader

from task.train.base import BaseTrainTask
from model import PLMNR
from agg import get_agg
from data import get_dataset, get_collate_fn


class UserTrainTask(BaseTrainTask):
    """Get per user gradients and aggregate them"""

    def load_model(self):
        model_cls = PLMNR
        agg_cls = get_agg(self.config["agg_name"])
        self.agg = agg_cls(
            self.args,
            news_dataset=self.news_dataset, 
            news_index = self.news_index,
            device=self.device,
        )
        self.module = GradSampleModule(model_cls(self.args).to(self.device))

    def process_user_grad(self, model, user_mask_matrix):
        user_grad = {}
        user_sample_num = torch.sum(user_mask_matrix, dim=1).cpu()

        for name, param in model.named_parameters():
            if param.requires_grad:
                user_grad[name] = torch.einsum(
                    "ub,b...->u...", user_mask_matrix, param.grad_sample
                ).cpu()
        return user_grad, user_sample_num

    def process_news_grad(self, candidate_info, his_info):
        news_grad = {}
        candidate_news, candidate_vecs, candidate_grad = candidate_info
        his, his_vecs, his_grad = his_info

        candidate_news, candaidate_grad = (
            candidate_news.reshape(-1,),
            candidate_grad.reshape(-1, 400),
        )
        his, his_grad = his.reshape(-1,), his_grad.reshape(-1, 400)

        for nid, grad in zip(his, his_grad):
            if nid in news_grad:
                news_grad[nid] += grad
            else:
                news_grad[nid] = grad

        for nid, grad in zip(candidate_news, candaidate_grad):
            if nid in news_grad:
                news_grad[nid] += grad
            else:
                news_grad[nid] = grad
        return news_grad

    def collect_users_nids(self, train_sam, users, user_indices, nid2index):
        # TODO: replace with MPC
        user_nids = [0]
        user_sample = 0
        for user in users:
            sids = user_indices[user]
            user_sample += len(sids)
            for idx in sids:
                _, pos, neg, his, _ = train_sam[idx]
                user_nids.extend([nid2index[i] for i in list(set([pos] + neg + his))])
        return list(set(user_nids)), user_sample

    def train_on_step(self, step):
        users = random.sample(self.user_indices.keys(), self.args.user_num)
        nids, user_sample = self.collect_users_nids(self.train_sam, users, self.user_indices, self.nid2index)
        self.agg.gen_news_vecs(nids)

        train_ds = get_dataset(self.config["train_dataset_name"])(
            self.args,
            self.train_sam,
            users,
            self.user_indices,
            self.nid2index,
            self.agg,
            self.news_index
        )
        train_collate_fn = get_collate_fn(self.config["train_collate_fn_name"])
        train_dl = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=train_collate_fn,
        )
        user_sample = len(train_ds)

        self.module._module.user_encoder.load_state_dict(self.agg.user_encoder.state_dict())

        self.module.train()
        loss = 0

        # get update of benigh users
        for cnt, data in enumerate(train_dl):
            sample_num = data["batch_label"].shape[0]
            for key in ["batch_candidate_news_vecs", "batch_his_vecs", "batch_label", "user_mask_matrix"]:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(self.device)

            data["batch_candidate_news_vecs"].requires_grad = True
            data["batch_his_vecs"].requires_grad = True

            bz_loss, y_hat = self.module(data)

            loss += bz_loss.detach().cpu().numpy()
            bz_loss.backward()

            candaidate_grad = data["batch_candidate_news_vecs"].grad.detach().cpu() * (
                sample_num / user_sample
            )
            candidate_vecs = data["batch_candidate_news_vecs"].detach().cpu().numpy()
            candidate_news = data["batch_candidate_news"].numpy()

            his_grad = data["batch_his_vecs"].grad.detach().cpu() * (sample_num / user_sample)
            his_vecs = data["batch_his_vecs"].detach().cpu().numpy()
            his = data["batch_his"].numpy()


            news_grad = self.process_news_grad(
                [candidate_news, candidate_vecs, candaidate_grad], [his, his_vecs, his_grad]
            )

            user_grad, user_sample_num = self.process_user_grad(
                self.module._module.user_encoder, data["user_mask_matrix"]
            )
            self.agg.collect(user_grad, news_grad)
            self.module.zero_grad(set_to_none=True)

        loss = loss / (cnt + 1)
        self.agg.update(user_sample)

        if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)
