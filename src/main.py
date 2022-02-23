import torch
import os
import logging
from pathlib import Path

from task import get_task
from config import job_config
from parameters import parse_args, setuplogger

if __name__ == "__main__":
    args = parse_args()
    out_path = Path(args.out_path) / f"{args.run_name}-{args.data}"
    out_path.mkdir(exist_ok=True, parents=True)

    setuplogger(args, out_path)
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    config = job_config[args.job_name]
    logging.info(f"[-] load config of {args.job_name}")
    logging.info(config)

    if args.mode == "train":
        task_cls = get_task(config["train_task_name"])
    else:
        task_cls = get_task(config["test_task_name"])

    task = task_cls(args, config, device)
    task.start()