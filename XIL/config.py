import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("XIL-XEED")
    parser.add_argument("--config", type=str, default="configs/domainnet.json")
    parser.add_argument("--run_mode", choices=["centroids", "train", "eval"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume_eval_task", type=int, default=999)

    # paths
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--proto_data_path", type=str, default="")
    parser.add_argument("--centroid_output_dir", type=str, default="")

    # experiment/data
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--domain_order", type=str)
    parser.add_argument("--total_cls", type=int)
    parser.add_argument("--total_sessions", type=int)

    return parser


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_args() -> Dict[str, Any]:
    cli = vars(build_parser().parse_args())
    cfg = load_json(cli["config"])
    args = {**cfg, **{k: v for k, v in cli.items() if v is not None}}

    if isinstance(args.get("domain_order"), str):
        args["domain_order"] = [x.strip() for x in args["domain_order"].split(",") if x.strip()]
    if isinstance(args.get("device"), str):
        args["device"] = [torch.device(f"cuda:{d}") for d in args["device"].split(",")]

    args["log_dir"] = make_log_dir(args)
    set_seed(args["seed"], deterministic=True)
    setup_logging(args)
    return args


def make_log_dir(args: Dict[str, Any]) -> str:
    stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    log_dir = Path(args["output_dir"]) / f"XEED_{args['dataset']}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def setup_logging(args: Dict[str, Any]) -> None:
    log_file = Path(args["log_dir"]) / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    for k, v in sorted(args.items()):
        logging.info("%s: %s", k, v)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
