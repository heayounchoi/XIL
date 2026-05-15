from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from .config import load_args
from .data import XILDataManager
from .labels import get_classnames
from .model import XEED
from .trainer import XILTrainer
from .utils import count_parameters


def load_checkpoint_if_needed(model, trainer, checkpoint_path: str):
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned_state = OrderedDict()
    for key, value in state.items():
        cleaned_state[key.replace("module.", "", 1)] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    trainer.all_keys = checkpoint.get("domain_keys", {})

    logging.info("Loaded checkpoint: %s", checkpoint_path)
    logging.info("Missing keys: %s", missing)
    logging.info("Unexpected keys: %s", unexpected)


def main():
    args = load_args()
    data_manager = OMyGapsDataManager(args)
    model = OMyGapsNet(args, data_manager.task_order)
    trainer = OMyGapsTrainer(args, get_classnames(args["dataset"]))
    trainer.task_order = data_manager.task_order

    load_checkpoint_if_needed(model, trainer, args.get("checkpoint", ""))

    for _ in range(data_manager.nb_tasks):
        logging.info("Trainable params: %d", count_parameters(model, trainable=True))
        model = trainer.incremental_step(model, data_manager)


if __name__ == "__main__":
    main()
