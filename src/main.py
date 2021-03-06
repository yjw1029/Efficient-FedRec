import argparse
from pathlib import Path

from tqdm import tqdm
import random
import wandb
import numpy as np
import os
import pickle

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from agg import Aggregator
from model import Model, TextEncoder, UserEncoder
from data import TrainDataset, NewsDataset, UserDataset
from metrics import evaluation_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "predict"]
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "../data"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "../output"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa"],
        help="decide which dataset for preprocess",
    )
    parser.add_argument("--bert_type", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--trainable_layers", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11]
    )
    parser.add_argument("--user_lr", type=float, default=0.00005)
    parser.add_argument("--news_lr", type=float, default=0.00005)
    parser.add_argument("--user_num", type=int, default=50)
    parser.add_argument("--max_his_len", type=float, default=50)
    parser.add_argument(
        "--npratio",
        type=int,
        default=20,
        help="randomly sample neg_num negative impression for every positive behavior",
    )
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--name", type=str, default="efficient-fedrec")

    args = parser.parse_args()
    return args


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


def collect_users_nids(train_sam, users, user_indices, nid2index):
    user_nids = [0]
    user_sample = 0
    for user in users:
        sids = user_indices[user]
        user_sample += len(sids)
        for idx in sids:
            _, pos, neg, his, _ = train_sam[idx]
            user_nids.extend([nid2index[i] for i in list(set([pos] + neg + his))])
    return list(set(user_nids)), user_sample


def train_on_step(
    agg, model, args, user_indices, user_num, train_sam, nid2index, news_index, device
):
    # sample users
    users = random.sample(user_indices.keys(), user_num)
    nids, user_sample = collect_users_nids(train_sam, users, user_indices, nid2index)

    agg.gen_news_vecs(nids)
    train_ds = TrainDataset(
        args, train_sam, users, user_indices, nid2index, agg, news_index
    )
    train_dl = DataLoader(train_ds, batch_size=16384, shuffle=True, num_workers=0)
    model.train()
    loss = 0

    for cnt, batch_sample in enumerate(train_dl):
        model.user_encoder.load_state_dict(agg.user_encoder.state_dict())
        optimizer = optim.SGD(model.parameters(), lr=args.user_lr)

        candidate_news, candidate_news_vecs, his, his_vecs, label = batch_sample
        candidate_news_vecs = candidate_news_vecs.to(device)
        his_vecs = his_vecs.to(device)
        sample_num = his_vecs.shape[0]

        label = label.to(device)

        # compute gradients for user model and news representations
        candidate_news_vecs.requires_grad = True
        his_vecs.requires_grad = True
        bz_loss, y_hat = model(candidate_news_vecs, his_vecs, label)
        loss += bz_loss.detach().cpu().numpy()

        optimizer.zero_grad()
        bz_loss.backward()

        candaidate_grad = candidate_news_vecs.grad.detach().cpu() * (
            sample_num / user_sample
        )
        candidate_vecs = candidate_news_vecs.detach().cpu().numpy()
        candidate_news = candidate_news.numpy()

        his_grad = his_vecs.grad.detach().cpu() * (sample_num / user_sample)
        his_vecs = his_vecs.detach().cpu().numpy()
        his = his.numpy()

        news_grad = process_news_grad(
            [candidate_news, candidate_vecs, candaidate_grad], [his, his_vecs, his_grad]
        )
        user_grad = process_user_grad(
            model.user_encoder.named_parameters(), sample_num, user_sample
        )
        agg.collect(news_grad, user_grad)

    loss = loss / (cnt + 1)
    agg.update()
    return loss


def validate(args, agg, valid_sam, nid2index, news_index, device):
    agg.gen_news_vecs(list(range(len(news_index))))
    agg.user_encoder.eval()
    user_dataset = UserDataset(args, valid_sam, agg.news_vecs, nid2index)
    user_vecs = []
    user_dl = DataLoader(user_dataset, batch_size=4096, shuffle=False, num_workers=0)
    with torch.no_grad():
        for his in tqdm(user_dl):
            his = his.to(device)
            user_vec = agg.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
    user_vecs = np.concatenate(user_vecs)

    val_scores = evaluation_split(agg.news_vecs, user_vecs, valid_sam, nid2index)
    val_auc, val_mrr, val_ndcg, val_ndcg10 = [
        np.mean(i) for i in list(zip(*val_scores))
    ]

    return val_auc, val_mrr, val_ndcg, val_ndcg10


def test(args, data_path, out_model_path, out_path, device):
    with open(data_path / "test_sam_uid.pkl", "rb") as f:
        test_sam = pickle.load(f)

    with open(data_path / "bert_test_nid2index.pkl", "rb") as f:
        test_nid2index = pickle.load(f)

    test_news_index = np.load(data_path / "bert_test_news_index.npy", allow_pickle=True)

    text_encoder = TextEncoder(bert_type=args.bert_type).to(device)
    user_encoder = UserEncoder().to(device)
    ckpt = torch.load(out_model_path / f"{args.name}-{args.data}.pkl")
    text_encoder.load_state_dict(ckpt["text_encoder"])
    user_encoder.load_state_dict(ckpt["user_encoder"])

    test_news_dataset = NewsDataset(test_news_index)
    news_dl = DataLoader(
        test_news_dataset, batch_size=512, shuffle=False, num_workers=0
    )
    news_vecs = []
    text_encoder.eval()
    for news in tqdm(news_dl):
        news = news.to(device)
        news_vec = text_encoder(news).detach().cpu().numpy()
        news_vecs.append(news_vec)
    news_vecs = np.concatenate(news_vecs)

    user_dataset = UserDataset(args, test_sam, news_vecs, test_nid2index)
    user_vecs = []
    user_dl = DataLoader(user_dataset, batch_size=4096, shuffle=False, num_workers=0)
    user_encoder.eval()
    for his in tqdm(user_dl):
        his = his.to(device)
        user_vec = user_encoder(his).detach().cpu().numpy()
        user_vecs.append(user_vec)
    user_vecs = np.concatenate(user_vecs)

    test_scores = evaluation_split(news_vecs, user_vecs, test_sam, test_nid2index)
    test_auc, test_mrr, test_ndcg, test_ndcg10 = [
        np.mean(i) for i in list(zip(*test_scores))
    ]

    with open(out_path / f"log.txt", "a") as f:
        f.write(
            f"test auc: {test_auc:.4f}, mrr: {test_mrr:.4f}, ndcg5: {test_ndcg:.4f}, ndcg10: {test_ndcg10:.4f}\n"
        )


def predict(args, data_path, out_model_path, out_path, device):
    with open(data_path / "test_sam_uid.pkl", "rb") as f:
        test_sam = pickle.load(f)

    with open(data_path / "bert_test_nid2index.pkl", "rb") as f:
        test_nid2index = pickle.load(f)

    test_news_index = np.load(data_path / "bert_test_news_index.npy", allow_pickle=True)

    text_encoder = TextEncoder(bert_type=args.bert_type).to(device)
    user_encoder = UserEncoder().to(device)
    ckpt = torch.load(out_model_path / f"{args.name}-{args.data}.pkl")
    text_encoder.load_state_dict(ckpt["text_encoder"])
    user_encoder.load_state_dict(ckpt["user_encoder"])

    test_news_dataset = NewsDataset(test_news_index)
    news_dl = DataLoader(
        test_news_dataset, batch_size=512, shuffle=False, num_workers=0
    )
    news_vecs = []
    text_encoder.eval()
    for news in tqdm(news_dl):
        news = news.to(device)
        news_vec = text_encoder(news).detach().cpu().numpy()
        news_vecs.append(news_vec)
    news_vecs = np.concatenate(news_vecs)

    user_dataset = UserDataset(args, test_sam, news_vecs, test_nid2index)
    user_vecs = []
    user_dl = DataLoader(user_dataset, batch_size=4096, shuffle=False, num_workers=0)
    user_encoder.eval()
    for his in tqdm(user_dl):
        his = his.to(device)
        user_vec = user_encoder(his).detach().cpu().numpy()
        user_vecs.append(user_vec)
    user_vecs = np.concatenate(user_vecs)

    pred_lines = []
    for i in tqdm(range(len(test_sam))):
        impr_id, poss, negs, _, _ = test_sam[i]
        user_vec = user_vecs[i]
        news_ids = [test_nid2index[i] for i in poss + negs]
        news_vec = news_vecs[news_ids]
        y_score = np.multiply(news_vec, user_vec)
        y_score = np.sum(y_score, axis=1)

        pred_rank = (np.argsort(np.argsort(y_score)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        pred_lines.append((int(impr_id), ' '.join([impr_id, pred_rank])+ '\n'))

    pred_lines.sort(key=lambda x: x[0])
    pred_lines = [x[1] for x in pred_lines]
    with open(out_path / 'prediction.txt', 'w') as f:
        f.writelines(pred_lines)


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if args.mode == "train":
        wandb.init(
            project=f"{args.name}-{args.data}", config=args, entity=args.wandb_entity
        )

        data_path = Path(args.data_path) / args.data
        out_path = Path(args.out_path) / f"{args.name}-{args.data}"
        out_model_path = out_path / "model"

        out_model_path.mkdir(exist_ok=True, parents=True)

        # load preprocessed data
        with open(data_path / "bert_nid2index.pkl", "rb") as f:
            nid2index = pickle.load(f)

        news_index = np.load(data_path / "bert_news_index.npy", allow_pickle=True)

        with open(data_path / "train_sam_uid.pkl", "rb") as f:
            train_sam = pickle.load(f)

        with open(data_path / "valid_sam_uid.pkl", "rb") as f:
            valid_sam = pickle.load(f)

        with open(data_path / "user_indices.pkl", "rb") as f:
            user_indices = pickle.load(f)

        news_dataset = NewsDataset(news_index)

        agg = Aggregator(args, news_dataset, news_index, device)
        model = Model().to(device)
        best_auc = 0
        for step in range(args.max_train_steps):
            loss = train_on_step(
                agg,
                model,
                args,
                user_indices,
                args.user_num,
                train_sam,
                nid2index,
                news_index,
                device,
            )

            wandb.log({"train loss": loss}, step=step + 1)

            if (step + 1) % args.validation_steps == 0:
                val_auc, val_mrr, val_ndcg, val_ndcg10 = validate(
                    args, agg, valid_sam, nid2index, news_index, device
                )

                wandb.log(
                    {
                        "valid auc": val_auc,
                        "valid mrr": val_mrr,
                        "valid ndcg@5": val_ndcg,
                        "valid ndcg@10": val_ndcg10,
                    },
                    step=step + 1,
                )

                with open(out_path / f"log.txt", "a") as f:
                    f.write(
                        f"[{step}] round auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}\n"
                    )

                if val_auc > best_auc:
                    best_auc = val_auc
                    wandb.run.summary["best_auc"] = best_auc
                    torch.save(
                        {
                            "text_encoder": agg.text_encoder.state_dict(),
                            "user_encoder": agg.user_encoder.state_dict(),
                        },
                        out_model_path / f"{args.name}-{args.data}.pkl",
                    )

                    with open(out_path / f"log.txt", "a") as f:
                        f.write(f"[{step}] round save model\n")

    elif args.mode == "test":
        data_path = Path(args.data_path) / args.data
        out_path = Path(args.out_path) / f"{args.name}-{args.data}"
        out_model_path = out_path / "model"
        test(args, data_path, out_model_path, out_path, device)

    elif args.mode == "predict":
        data_path = Path(args.data_path) / args.data
        out_path = Path(args.out_path) / f"{args.name}-{args.data}"
        out_model_path = out_path / "model"
        predict(args, data_path, out_model_path, out_path, device)

