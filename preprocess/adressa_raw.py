# This script is used to construct training, validation and test dataset of adressa.
# We follow existing works[1][2] to split the dataset.
#   [1]
#   [2]

import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adressa_path",
        type=str,
        default="../raw/adressa/raw/one_week",
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../raw/adressa/",
        help="path to save processed dataset, default in ../raw/adressa",
    )
    parser.add_argument(
        "--neg_num",
        type=int,
        default=20,
        help="randomly sample neg_num negative impression for every positive behavior",
    )

    args = parser.parse_args()
    return args


def process_news(adressa_path):
    news_title = {}

    for file in adressa_path.iterdir():
        with open(file, "r") as f:
            for l in tqdm(f):
                event_dict = json.loads(l.strip("\n"))
                if "id" in event_dict and "title" in event_dict:
                    if event_dict["id"] not in news_title:
                        news_title[event_dict["id"]] = event_dict["title"]
                    else:
                        assert news_title[event_dict["id"]] == event_dict["title"]

    nid2index = {k: v for k, v in zip(news_title.keys(), range(1, len(news_title) + 1))}
    return news_title, nid2index


def write_news_files(news_title, nid2index, out_path):
    # Output with MIND format
    news_lines = []
    for nid in tqdm(news_title):
        nindex = nid2index[nid]
        title = news_title[nid]
        news_line = "\t".join([str(nindex), "", "", title, "", "", "", ""]) + "\n"
        news_lines.append(news_line)

    for stage in ["train", "valid", "test"]:
        file_path = out_path / stage
        file_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / stage / "news.tsv", "w", encoding="utf-8") as f:
            f.writelines(news_lines)


class UserInfo:
    def __init__(self, train_day=6, test_day=7):
        self.click_news = []
        self.click_time = []
        self.click_days = []

        self.train_news = []
        self.train_time = []
        self.train_days = []

        self.test_news = []
        self.test_time = []
        self.test_days = []

        self.train_day = train_day
        self.test_day = test_day

    def update(self, nindex, time, day):
        if day == self.train_day:
            self.train_news.append(nindex)
            self.train_time.append(time)
            self.train_days.append(day)
        elif day == self.test_day:
            self.test_news.append(nindex)
            self.test_time.append(time)
            self.test_days.append(day)
        else:
            self.click_news.append(nindex)
            self.click_time.append(time)
            self.click_days.append(day)

    def sort_click(self):
        self.click_news = np.array(self.click_news, dtype="int32")
        self.click_time = np.array(self.click_time, dtype="int32")
        self.click_days = np.array(self.click_days, dtype="int32")

        self.train_news = np.array(self.train_news, dtype="int32")
        self.train_time = np.array(self.train_time, dtype="int32")
        self.train_days = np.array(self.train_days, dtype="int32")

        self.test_news = np.array(self.test_news, dtype="int32")
        self.test_time = np.array(self.test_time, dtype="int32")
        self.test_days = np.array(self.test_days, dtype="int32")

        order = np.argsort(self.train_time)
        self.train_time = self.train_time[order]
        self.train_days = self.train_days[order]
        self.train_news = self.train_news[order]

        order = np.argsort(self.test_time)
        self.test_time = self.test_time[order]
        self.test_days = self.test_days[order]
        self.test_news = self.test_news[order]

        order = np.argsort(self.click_time)
        self.click_time = self.click_time[order]
        self.click_days = self.click_days[order]
        self.click_news = self.click_news[order]


def process_users(adressa_path):
    uid2index = {}
    user_info = defaultdict(UserInfo)

    for file in adressa_path.iterdir():
        with open(file, "r") as f:
            for l in tqdm(f):
                event_dict = json.loads(l.strip("\n"))
                if "id" in event_dict and "title" in event_dict:
                    nindex = nid2index[event_dict["id"]]
                    uid = event_dict["userId"]

                    if uid not in uid2index:
                        uid2index[uid] = len(uid2index)

                    uindex = uid2index[uid]
                    click_time = int(event_dict["time"])
                    day = int(file.name[-1])
                    user_info[uindex].update(nindex, click_time, day)

    return uid2index, user_info


def construct_behaviors(uindex, click_news, train_news, test_news, neg_num):
    p = np.ones(len(news_title) + 1, dtype="float32")
    p[click_news] = 0
    p[train_news] = 0
    p[test_news] = 0
    p[0] = 0
    p /= p.sum()

    train_his_news = [str(i) for i in click_news.tolist()]
    train_his_line = " ".join(train_his_news)

    for nindex in train_news:
        neg_cand = np.random.choice(
            len(news_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
        )

        train_behavior_line = f"null\t{uindex}\tnull\t{train_his_line}\t{cand_news}\n"
        train_lines.append(train_behavior_line)

    test_his_news = [str(i) for i in click_news.tolist() + train_news.tolist()]
    test_his_line = " ".join(test_his_news)
    for nindex in test_news:
        neg_cand = np.random.choice(
            len(news_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
        )

        test_behavior_line = f"null\t{uindex}\tnull\t{test_his_line}\t{cand_news}\n"
        test_lines.append(test_behavior_line)


if __name__ == "__main__":
    args = parse_args()
    adressa_path = Path(args.adressa_path)
    out_path = Path(args.out_path)

    news_title, nid2index = process_news(adressa_path)
    write_news_files(news_title, nid2index, out_path)

    uid2index, user_info = process_users(adressa_path)
    for uid in tqdm(user_info):
        user_info[uid].sort_click()

    train_lines = []
    test_lines = []
    for uindex in tqdm(user_info):
        uinfo = user_info[uindex]
        click_news = uinfo.click_news
        train_news = uinfo.train_news
        test_news = uinfo.test_news
        construct_behaviors(uindex, click_news, train_news, test_news, args.neg_num)

    test_split_lines, valid_split_lines = train_test_split(
        test_lines, test_size=0.2, random_state=2021
    )
    with open(out_path / "train" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(out_path / "valid" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(valid_split_lines)

    with open(out_path / "test" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(test_split_lines)
