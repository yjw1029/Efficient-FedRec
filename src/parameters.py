import os
import sys
import logging
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "../data"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "../output"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa", "feeds"],
        help="decide which dataset for preprocess",
    )

    # job configutation
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "predict"])
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--job_name", type=str, default="Efficient-FedRec-Fast", help="name to choose job config including task, dataset, model etc..")
    parser.add_argument("--project_name", type=str, default="efficient-fedrec", help="Wandb project name.")
    parser.add_argument("--run_name", type=str, default="", help="Wandb run name.")

    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--user_num", type=int, default=50)
    parser.add_argument("--max_his_len", type=float, default=50)
    parser.add_argument(
        "--npratio",
        type=int,
        default=20,
        help="randomly sample npratio negative behaviors for every positive behavior",
    )
    parser.add_argument("--max_train_steps", type=int, default=4000)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--bert_type", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--trainable_layers", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11]
    )
    parser.add_argument("--user_lr", type=float, default=0.00005)
    parser.add_argument("--news_lr", type=float, default=0.00005)

    parser.add_argument("--mpc_t", type=int, default=30)

  
    
    args = parser.parse_args()
    return args


def setuplogger(args, out_path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    fh = logging.FileHandler(out_path / f"log.{args.mode}.txt")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)