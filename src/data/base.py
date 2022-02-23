import random
import numpy as np

import torch
from torch.utils.data import Dataset


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)

class TrainBaseDataset(Dataset):
    def __init__(self, args, samples, users, user_indices, nid2index, agg, news_index):
        self.news_index = news_index
        self.nid2index = nid2index
        self.agg = agg
        self.samples = []
        self.args = args
        
        for user in users:
            self.samples.extend([samples[i] for i in user_indices[user]])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, _ = self.samples[idx]
        neg = newsample(neg, self.args.npratio)
        candidate_news = np.array([self.nid2index[n] for n in [pos] + neg])
        candidate_news_vecs = self.agg.get_news_vecs(candidate_news)
        his = np.array([self.nid2index[n] for n in his] + [0] * (self.args.max_his_len - len(his)))
        his_vecs = self.agg.get_news_vecs(his)
        label = 0

        return candidate_news, candidate_news_vecs, his, his_vecs, label


def train_base_collate_fn(data):
    batch_candidate_news, batch_candidate_news_vecs, batch_his, batch_his_vecs, batch_label = zip(*data)
    batch_candidate_news = np.stack(batch_candidate_news)
    batch_his = np.stack(batch_his)

    batch_candidate_news = torch.LongTensor(batch_candidate_news)
    batch_candidate_news_vecs = torch.FloatTensor(batch_candidate_news_vecs)
    batch_his = torch.LongTensor(batch_his)
    batch_his_vecs = torch.FloatTensor(batch_his_vecs)
    batch_label = torch.LongTensor(batch_label)

    batch_data = {
        "batch_candidate_news": batch_candidate_news,
        "batch_candidate_news_vecs": batch_candidate_news_vecs,
        "batch_his": batch_his,
        "batch_his_vecs": batch_his_vecs,
        "batch_label": batch_label,
    }

    return batch_data
