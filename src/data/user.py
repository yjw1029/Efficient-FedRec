import numpy as np

import torch
from torch.utils.data import Dataset

from data.base import newsample


def train_user_collate_fn(data):
    batch_candidate_news, batch_candidate_news_vecs, batch_his, batch_his_vecs, batch_label, batch_uindex, batch_max_unum = zip(
        *data
    )
    max_unum = batch_max_unum[0]

    batch_candidate_news = np.stack(batch_candidate_news)
    batch_his = np.stack(batch_his)

    batch_candidate_news = torch.LongTensor(batch_candidate_news)
    batch_candidate_news_vecs = torch.FloatTensor(batch_candidate_news_vecs)
    batch_his = torch.LongTensor(batch_his)
    batch_his_vecs = torch.FloatTensor(batch_his_vecs)
    batch_label = torch.LongTensor(batch_label)

    user_mask_matrix = (
        torch.nn.functional.one_hot(
            torch.LongTensor(batch_uindex), num_classes=max_unum
        )
        .t()
        .float()
    )

    data = {
        "batch_candidate_news": batch_candidate_news,
        "batch_candidate_news_vecs": batch_candidate_news_vecs,
        "batch_his": batch_his,
        "batch_his_vecs": batch_his_vecs,
        "batch_label": batch_label,
        "user_mask_matrix": user_mask_matrix,
    }
    return data


class TrainUserDataset(Dataset):
    def __init__(
        self,
        args,
        samples,
        users,
        user_indices,
        nid2index,
        agg,
        news_index,
        *other_args,
        **kwargs
    ):
        self.args = args
        self.agg = agg
        self.nid2index = nid2index
        self.news_index = news_index
        self.samples = []

        for user in users:
            self.samples.extend([samples[i] for i in user_indices[user]])

        self.tmp_user_dict = {u: i for i, u in enumerate(users)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, uid = self.samples[idx]

        uindex = self.tmp_user_dict[uid]
        neg = newsample(neg, self.args.npratio)
        candidate_news = np.array([self.nid2index[n] for n in [pos] + neg])
        candidate_news_vecs = self.agg.get_news_vecs(candidate_news)
        his = np.array([self.nid2index[n] for n in his] + [0] * (self.args.max_his_len - len(his)))
        his_vecs = self.agg.get_news_vecs(his)
        label = 0

        return candidate_news, candidate_news_vecs, his, his_vecs, label, uindex, len(self.tmp_user_dict)
