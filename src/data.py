import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)

class TrainDataset(Dataset):
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
        label = np.array(0)

        return candidate_news, candidate_news_vecs, his, his_vecs, label


class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index
        
    def __len__(self):
        return len(self.news_index)
    
    def __getitem__(self, idx):
        return self.news_index[idx]


class NewsPartDataset(Dataset):
    def __init__(self, news_index, nids):
        self.news_index = news_index
        self.nids = nids
        
    def __len__(self):
        return len(self.nids)
    
    def __getitem__(self, idx):
        nid = self.nids[idx]
        return nid, self.news_index[nid]


class UserDataset(Dataset):
    def __init__(self,
                 args,
                 samples,
                 news_vecs,
                 nid2index):
        self.samples = samples
        self.args = args
        self.news_vecs = news_vecs
        self.nid2index = nid2index
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, poss, negs, his, _ = self.samples[idx]
        his = [self.nid2index[n] for n in his] + [0] * (self.args.max_his_len - len(his))
        his = self.news_vecs[his]
        return his


class NewsUpdatorDataset(Dataset):
    def __init__(self, news_index, news_ids, news_grads):
        self.news_index = news_index
        self.news_grads = news_grads
        self.news_ids = news_ids
        
    def __len__(self):
        return len(self.news_ids)
    
    def __getitem__(self, idx):
        nid = self.news_ids[idx]
        return self.news_index[nid], self.news_grads[idx]