from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module, trainable: bool = False) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model
