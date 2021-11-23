from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="../raw/",
        help="path to raw mind dataset or parsed ",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../data/",
        help="path to save processed dataset, default in ../raw/mind/preprocess",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa"],
        help="decide which dataset for preprocess"
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
    parser.add_argument("--max_title_len", type=int, default=30)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    raw_path = Path(args.raw_path) / args.data
    out_path = Path(args.out_path) / args.data

    user_imprs = defaultdict(list)

    # read user impressions
    for l in tqdm(open(raw_path / "train" / "behaviors.tsv", "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
        his = his.split()
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]
        user_imprs[uid].append([imp_id, his, pos_imp, neg_imp, 0, uid])

    for l in tqdm(open(raw_path / "valid" / "behaviors.tsv", "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
        his = his.split()
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]
        user_imprs[uid].append([imp_id, his, pos_imp, neg_imp, 1, uid])

    if os.path.exists(raw_path / "test"):
        if args.data == "adressa":
            for l in tqdm(open(raw_path / "test" / "behaviors.tsv", "r")):
                imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
                his = his.split()
                imprs = [i.split("-") for i in imprs.split(" ")]
                neg_imp = [i[0] for i in imprs if i[1] == "0"]
                pos_imp = [i[0] for i in imprs if i[1] == "1"]
                user_imprs[uid].append([imp_id, his, pos_imp, neg_imp, 2, uid])
        else:
            # MIND test dataset do not contains labels, need to test on condalab
            for l in tqdm(open(raw_path / "test" / "behaviors.tsv", "r")):
                imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
                his = his.split()
                imprs = imprs.split(" ")
                user_imprs[uid].append([imp_id, his, imprs, [], 2, uid])


    train_samples = []
    valid_samples = []
    test_samples = []
    user_indices = defaultdict(list)

    index = 0
    for uid in tqdm(user_imprs):
        for impr in user_imprs[uid]:
            imp_id, his, poss, negs, is_valid, uid = impr
            his = his[-args.max_his_len:]
            if is_valid == 0:
                for pos in poss:
                    train_samples.append([imp_id, pos, negs, his, uid])
                    user_indices[uid].append(index)
                    index += 1
            elif is_valid == 1:
                valid_samples.append([imp_id, poss, negs, his, uid])
            else:
                test_samples.append([imp_id, poss, negs, his, uid])

    print(len(train_samples), len(valid_samples), len(test_samples))

    with open(out_path / "train_sam_uid.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(out_path / "valid_sam_uid.pkl", "wb") as f:
        pickle.dump(valid_samples, f)

    with open(out_path / "test_sam_uid.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    with open(out_path / "user_indices.pkl", "wb") as f:
        pickle.dump(user_indices, f)

    train_user_samples = 0

    for uid in tqdm(user_indices):
        train_user_samples += len(user_indices[uid])

    print(train_user_samples / len(user_indices))
