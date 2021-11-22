import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True)  + 1e-8)
        
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 300
        self.n_heads = n_heads # 20
        self.d_k = d_k # 20
        self.d_v = d_v # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads) # 300, 400
        
        self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self, Q, K, V, attn_mask=None):
        batch_size, seq_len, _ =  Q.size()
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) 
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        return context 


class AdditiveAttention(nn.Module):
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.user_encoder = UserEncoder()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, candidate_vecs, clicked_news_vecs, targets, compute_loss=True): 
        user_vector = self.user_encoder(clicked_news_vecs)
        
        score = torch.bmm(candidate_vecs, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score