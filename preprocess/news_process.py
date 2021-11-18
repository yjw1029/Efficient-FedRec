from transformers import BertTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import pickle
import argparse
# config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="../raw/mind",
        help="path to raw mind dataset or parsed ",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../raw/mind/preprocess",
        help="path to save processed dataset, default in ../raw/mind/preprocess",
    )
    parser.add_argument(
        "--npratio",
        type=int,
        default=4
    )
    parser.add_argument(
        "--max_his_len", type=int, default=50
    )
    parser.add_argument("--min_word_cnt", type=int, default=3)
    parser.add_argument("--min_title_len", type=int, default=30)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    adressa_path = Path(args.adressa_path)
    out_path = Path(args.out_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # news preprocess
    nid2index = {"<unk>": 0}
    news_index = [[[0] * args.max_title_len, [0] * args.max_title_len]]

    for l in tqdm(open(data_path / "train" / "news.tsv", "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        tokens = tokenizer(
            title,
            max_length=args.max_title_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        nid2index[nid] = len(nid2index)
        news_index.append([tokens.input_ids, tokens.attention_mask])


    for l in tqdm(open(data_path / "valid" / "news.tsv", "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        tokens = tokenizer(
            title,
            max_length=args.max_title_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        nid2index[nid] = len(nid2index)
        news_index.append([tokens.input_ids, tokens.attention_mask])

    with open(out_path / "bert_nid2index.pkl", "wb") as f:
        pickle.dump(nid2index, f)

    news_index = np.array(news_index)
    np.save(out_path / "bert_news_index", news_index)

    if os.path.exists(data_path / "test"):
        nid2index = {"<unk>": 0}
        news_index = [[[0] * args.max_title_len, [0] * args.max_title_len]]

        for l in tqdm(open(data_path / "test" / "news.tsv", "r", encoding='utf-8')):
            nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
            if nid in nid2index:
                continue
            tokens = tokenizer(
                title,
                max_length=args.max_title_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            nid2index[nid] = len(nid2index)
            news_index.append([tokens.input_ids, tokens.attention_mask])

        with open(out_path / "bert_test_nid2index.pkl", "wb") as f:
            pickle.dump(nid2index, f)

        news_index = np.array(news_index)
        np.save(out_path / "bert_test_news_index", news_index)
