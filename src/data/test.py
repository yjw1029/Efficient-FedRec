from torch.utils.data import Dataset

# datasets for testing and evaluation
class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index
        
    def __len__(self):
        return len(self.news_index)
    
    def __getitem__(self, idx):
        return self.news_index[idx]

class UserDataset(Dataset):
    def __init__(self, args, samples, news_vecs, nid2index):
        self.args = args
        self.samples = samples
        self.news_vecs = news_vecs
        self.nid2index = nid2index
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, poss, negs, his, _ = self.samples[idx]
        his = [0] * (self.args.max_his_len - len(his)) + [
            self.nid2index[n] for n in his
        ]
        his = self.news_vecs[his]

        return his