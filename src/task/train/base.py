import numpy as np
import pickle
import wandb
import random
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from task.base import BaseTask
from model import PLMNR
from agg import get_agg
from data import get_collate_fn, get_dataset, NewsDataset
from metrics import evaluation_split

def process_news_grad(candidate_info, his_info):
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


def process_user_grad(model_param, sample_num, user_sample):
    user_grad = {}
    for name, param in model_param:
        user_grad[name] = param.grad * (sample_num / user_sample)
    return user_grad


class BaseTrainTask(BaseTask):
    """Train baseline using large batch size SGD. Cannot compute per-user gradient"""

    def __init__(self, args, config, device):
        super().__init__(args, config, device)

        if "debug" not in args.run_name:
            wandb.init(
                project=f"{args.project_name}-{args.data}",
                config=args,
                name=f"{args.run_name}-{args.data}",
            )
            logging.info("[-] finishing initing wandb.")

    def load_data(self):
        with open(self.data_path / "bert_nid2index.pkl", "rb") as f:
            self.nid2index = pickle.load(f)

        self.news_index = np.load(
            self.data_path / "bert_news_index.npy", allow_pickle=True
        )

        with open(self.data_path / "train_sam_uid.pkl", "rb") as f:
            self.train_sam = pickle.load(f)

        with open(self.data_path / "valid_sam_uid.pkl", "rb") as f:
            self.valid_sam = pickle.load(f)

        with open(self.data_path / "user_indices.pkl", "rb") as f:
            self.user_indices = pickle.load(f)

        self.news_dataset = NewsDataset(self.news_index)


    def load_model(self):
        model_cls = PLMNR
        agg_cls = get_agg(self.config["agg_name"])
        self.agg = agg_cls(
            self.args,
            news_dataset=self.news_dataset, 
            news_index = self.news_index,
            device=self.device,
        )
        self.model = model_cls(self.args).to(self.device)

    
    def collect_users_nids(self, train_sam, users, user_indices, nid2index):
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
            self.news_index,
        )
        train_collate_fn = get_collate_fn(self.config["train_collate_fn_name"])
        train_dl = DataLoader(
            train_ds, collate_fn=train_collate_fn, batch_size=self.args.batch_size, shuffle=True, num_workers=0
        )
        user_sample = len(train_ds)

        self.model.user_encoder.load_state_dict(self.agg.user_encoder.state_dict())
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.user_lr)

        self.model.train()
        loss = 0
        for cnt, data in enumerate(train_dl):
            sample_num = data["batch_label"].shape[0]

            for key in ["batch_candidate_news_vecs", "batch_his_vecs", "batch_label"]:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(self.device)

            data["batch_candidate_news_vecs"].requires_grad = True
            data["batch_his_vecs"].requires_grad = True

            bz_loss, y_hat = self.model(data)

            # compute gradients for user model and news representations
            loss += bz_loss.detach().cpu().numpy()

            optimizer.zero_grad()
            bz_loss.backward()


            candaidate_grad = data["batch_candidate_news_vecs"].grad.detach().cpu() * (
                sample_num / user_sample
            )
            candidate_vecs = data["batch_candidate_news_vecs"].detach().cpu().numpy()
            candidate_news = data["batch_candidate_news"].numpy()

            his_grad = data["batch_his_vecs"].grad.detach().cpu() * (sample_num / user_sample)
            his_vecs = data["batch_his_vecs"].detach().cpu().numpy()
            his = data["batch_his"].numpy()

            news_grad = process_news_grad(
                [candidate_news, candidate_vecs, candaidate_grad], [his, his_vecs, his_grad]
            )
            user_grad = process_user_grad(
                self.model.user_encoder.named_parameters(), sample_num, user_sample
            )
            self.agg.collect(news_grad, user_grad)


        loss = loss / (cnt + 1)
        self.agg.update()

        if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)

    def validate(self, step):
        self.agg.gen_news_vecs(list(range(len(self.news_index))))

        self.agg.user_encoder.eval()
        user_dataset = get_dataset(self.config["user_dataset_name"])(
            self.args, self.valid_sam, self.agg.news_vecs, self.nid2index
        )
        user_vecs = []
        user_dl = DataLoader(
            user_dataset, batch_size=4096, shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for his in tqdm(user_dl):
                his = his.to(self.device)
                user_vec = self.agg.user_encoder(his).detach().cpu().numpy()
                user_vecs.append(user_vec)
        user_vecs = np.concatenate(user_vecs)

        val_scores = evaluation_split(
            self.agg.news_vecs, user_vecs, self.valid_sam, self.nid2index
        )
        val_auc, val_mrr, val_ndcg, val_ndcg10 = [
            np.mean(i) for i in list(zip(*val_scores))
        ]

        if "debug" not in self.args.run_name:
            wandb.log(
                {
                    "valid auc": val_auc,
                    "valid mrr": val_mrr,
                    "valid ndcg@5": val_ndcg,
                    "valid ndcg@10": val_ndcg10,
                },
                step=step + 1,
            )

        logging.info(
            f"[{step}] round auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}"
        )

        if val_auc > self.best_auc:
            self.best_auc = val_auc

            if "debug" not in self.args.run_name:
                wandb.run.summary["best_auc"] = self.best_auc
                wandb.run.summary["best_mrr"] = val_mrr
                wandb.run.summary["best_ndcg@5"] = val_ndcg
                wandb.run.summary["best_ndcg@10"] = val_ndcg10

            torch.save(
                {
                    "text_encoder": self.agg.text_encoder.state_dict(),
                    "user_encoder": self.agg.user_encoder.state_dict(),
                },
                self.out_model_path / f"{self.args.run_name}-{self.args.data}.pkl",
            )
            logging.info(f"[{step}] round save model")

    def start(self):
        self.best_auc = 0
        for step in range(self.args.max_train_steps):
            self.train_on_step(step)

            if (step + 1) % self.args.validation_steps == 0:
                self.validate(step)
