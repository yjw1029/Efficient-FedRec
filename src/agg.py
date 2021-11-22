import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import TextEncoder, UserEncoder
import torch.optim as optim

from data import NewsPartDataset

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


class Aggregator:
    def __init__(self, args, news_dataset, news_index, device):
        self.device = device

        self.text_encoder = TextEncoder(bert_type=args.bert_type).to(device)
        self.user_encoder = UserEncoder().to(device)
        
        self.news_optimizer = optim.Adam(self.text_encoder.parameters(), lr=args.news_lr)
        self.user_optimizer = optim.Adam(self.user_encoder.parameters(), lr=args.user_lr)
        
        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False

        for index, layer in enumerate(self.text_encoder.bert.encoder.layer):
            if index in args.trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        
        if -1 in args.trainable_layers:
            for param in self.text_encoder.bert.embeddings.parameters():
                    param.requires_grad = True
        
        self.news_dataset = news_dataset
        self.news_index = news_index
        
        self.time = 0
        self.cnt = 0
        
        self._init_grad_param()
    
    def _init_grad_param(self):        
        self.news_grads = {}
        self.user_optimizer.zero_grad()
        self.news_optimizer.zero_grad()

    def gen_news_vecs(self, nids):
        self.text_encoder.eval()
        news_ds = NewsPartDataset(self.news_index, nids)
        news_dl = DataLoader(news_ds, batch_size=2048, shuffle=False, num_workers=0)
        news_vecs = np.zeros((len(self.news_index), 400), dtype='float32')
        with torch.no_grad():
            for nids, news in news_dl:
                news = news.to(self.device)
                news_vec = self.text_encoder(news).detach().cpu().numpy()
                news_vecs[nids.numpy()] = news_vec
        if np.isnan(news_vecs).any():
            raise ValueError("news_vecs contains nan")
        self.news_vecs = news_vecs
        return news_vecs
    
    def get_news_vecs(self, idx):
        return self.news_vecs[idx]
    
    def update(self):
        self.update_user_grad()
        self.update_news_grad()
        self._init_grad_param()
        self.cnt += 1
    
    def average_update_time(self):
        return self.time / self.cnt
    
    def update_news_grad(self):
        self.text_encoder.train()
        self.news_optimizer.zero_grad()
        
        news_ids, news_grads = [], []
        for nid in self.news_grads:
            news_ids.append(nid)
            news_grads.append(self.news_grads[nid])
        
        news_up_ds = NewsUpdatorDataset(self.news_index, news_ids, news_grads)
        news_up_dl = DataLoader(news_up_ds, batch_size=128, shuffle=False, num_workers=0)
        for news_index, news_grad in news_up_dl:
            news_index = news_index.to(self.device)
            news_grad = news_grad.to(self.device)
            news_vecs = self.text_encoder(news_index)
            news_vecs.backward(news_grad)
        
        self.news_optimizer.step()
        self.news_optimizer.zero_grad()
        
    def update_user_grad(self):
        self.user_optimizer.step()
        self.user_optimizer.zero_grad()
    
    def check_news_vec_same(self, nids, news_vecs):
        assert (self.get_news_vecs(nids) == news_vecs).all(), "News vecs are not the same"
    
    def collect(self, news_grad, user_grad):
        # update user model params
        for name, param in self.user_encoder.named_parameters():
            if param.grad is None:
                param.grad = user_grad[name]
            else:
                param.grad += user_grad[name]
        
        # update news model params
        for nid in news_grad:
            if nid in self.news_grads:
                self.news_grads[nid] += news_grad[nid]
            else:
                self.news_grads[nid] = news_grad[nid]