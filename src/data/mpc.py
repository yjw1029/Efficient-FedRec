import numpy as np
from torch.utils.data import Dataset


class TrainMPCDataset(Dataset):
    def __init__(
        self,
        args,
        samples,
        users,
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
        self.samples = samples

        self.tmp_user_dict = {u: i for i, u in enumerate(users)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, uid = self.samples[idx]

        uindex = self.tmp_user_dict[uid]
        candidate_news = np.array([self.nid2index[n] for n in [pos] + neg])
        candidate_news_vecs = self.agg.get_news_vecs(candidate_news)
        his = np.array([self.nid2index[n] for n in his] + [0] * (self.args.max_his_len - len(his)))
        his_vecs = self.agg.get_news_vecs(his)
        label = 0

        return candidate_news, candidate_news_vecs, his, his_vecs, label, uindex, len(self.tmp_user_dict)