import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def compute_amn(y_true, y_score):
    auc = roc_auc_score(y_true,y_score)
    mrr = mrr_score(y_true,y_score)
    ndcg5 = ndcg_score(y_true,y_score,5)
    ndcg10 = ndcg_score(y_true,y_score,10)
    return auc, mrr, ndcg5, ndcg10

def evaluation_split(news_vecs, user_vecs, samples, nid2index):
    all_rslt = []
    for i in tqdm(range(len(samples))):
        _, poss, negs, _, _ = samples[i]
        user_vec = user_vecs[i]
        y_true = [1] * len(poss) + [0] * len(negs)
        news_ids = [nid2index[i] for i in poss + negs]
        news_vec = news_vecs[news_ids]
        y_score = np.multiply(news_vec, user_vec)
        y_score = np.sum(y_score, axis=1)
        try:
            all_rslt.append(compute_amn(y_true, y_score))
        except Exception as e:
            print(e)
    return np.array(all_rslt)