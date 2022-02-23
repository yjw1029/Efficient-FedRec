import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

import numpy as np
from model.layers import MultiHeadAttention, AdditiveAttention

class TextEncoder(nn.Module):
    def __init__(self,
                 bert_type="bert-base-uncased",
                 word_embedding_dim=400,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = 0.2
        self.bert = BertModel.from_pretrained(bert_type,
                                              hidden_dropout_prob=0,
                                              attention_probs_dropout_prob=0)
        self.additive_attention = AdditiveAttention(self.bert.config.hidden_size,
                                                    self.bert.config.hidden_size//2)
        self.fc = nn.Linear(self.bert.config.hidden_size, word_embedding_dim)

    def forward(self, text):
        # text batch, 2, word
        tokens = text[:,0,:]
        atts = text[:,1,:]
        text_vector = self.bert(tokens, attention_mask=atts)[0]
        text_vector = self.additive_attention(text_vector)
        text_vector = self.fc(text_vector)
        return text_vector


class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200
                ):
        super(UserEncoder, self).__init__()
        self.dropout_rate = 0.2
        self.multihead_attention = MultiHeadAttention(news_embedding_dim,
                                              num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(news_embedding_dim,
                                                    query_vector_dim)
        
    def forward(self, clicked_news_vecs):
        clicked_news_vecs = F.dropout(clicked_news_vecs, p=self.dropout_rate, training=self.training)
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)
        
        user_vector = pos_user_vector
        return user_vector

class PLMNR(nn.Module):
    def __init__(self, args):
        super(PLMNR, self).__init__()
        self.args = args
        self.user_encoder = UserEncoder()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, data, compute_loss=True): 
        user_vector = self.user_encoder(data["batch_his_vecs"])
        
        score = torch.bmm(data["batch_candidate_news_vecs"], user_vector.unsqueeze(-1)).squeeze(dim=-1)
        
        if compute_loss:
            loss = self.criterion(score, data["batch_label"])
            return loss, score
        else:
            return score