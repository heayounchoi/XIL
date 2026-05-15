from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import get_classnames
from .utils import load_json, save_json


class ImageListDataset(Dataset):
    def __init__(self, images, labels, transform, domains=None, mode: str = "train"):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.domains = None if domains is None else np.asarray(domains)
        self.transform = transform
        self.mode = mode
        self.classwise_data = self._build_classwise_index()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self._pil_loader(self.images[index]))
        label = int(self.labels[index])
        if self.mode == "test":
            return index, image, label, int(self.domains[index]), str(self.images[index])
        return index, image, label

    @staticmethod
    def _pil_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def _build_classwise_index(self):
        classwise = {}
        for path, label in zip(self.images, self.labels):
            classwise.setdefault(int(label), []).append(str(path))
        return classwise

    def get_classwise_dataset(self, class_label: int):
        paths = self.classwise_data[class_label]
        return ClasswiseImageDataset(class_label, paths, self.transform)


class ClasswiseImageDataset(Dataset):
    def __init__(self, class_label: int, paths: List[str], transform):
        self.class_label = class_label
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with open(self.paths[index], "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img), self.paths[index]


class XILDataManager:
    """
    Expected list files:
        {data_path}/{domain}_train.txt
        {data_path}/{domain}_test.txt

    Each line should be:
        relative/or/absolute/image/path class_index
    """

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_transform = [transforms.Resize(256), transforms.CenterCrop(224)]
    common_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        self.classnames = get_classnames(args["dataset"])
        self.domains = list(args["domain_order"])
        self.class_order = self._load_or_make_class_order()
        self.cls_per_task = self._make_cls_per_task()
        self.task_order = self._load_or_make_task_order()
        self.train_data, self.train_targets = {}, {}
        self.test_data, self.test_targets = {}, {}
        self._load_lists()

    @property
    def nb_tasks(self) -> int:
        return len(self.task_order)

    def get_task_size(self, task_id: int) -> int:
        return len(self.task_order[task_id])

    def get_dataset(self, task_id: int, mode: str):
        if mode == "train":
            return ImageListDataset(
                self.train_data[task_id],
                self.train_targets[task_id],
                transforms.Compose([*self.train_transform, *self.common_transform]),
                mode="train",
            )

        if mode == "proto":
            data, targets = self._load_proto_data(task_id)
            return ImageListDataset(
                data,
                targets,
                transforms.Compose([*self.test_transform, *self.common_transform]),
                mode="proto",
            )

        if mode == "test":
            data = np.concatenate([self.test_data[i] for i in range(task_id + 1)], axis=0)
            targets = np.concatenate([self.test_targets[i] for i in range(task_id + 1)], axis=0)
            domains = np.concatenate([[i] * len(self.test_targets[i]) for i in range(task_id + 1)])
            seen_classes = np.concatenate(self.task_order[: task_id + 1], axis=0)
            mask = np.isin(targets, seen_classes)
            return ImageListDataset(
                data[mask],
                targets[mask],
                transforms.Compose([*self.test_transform, *self.common_transform]),
                domains=domains[mask],
                mode="test",
            )

        if mode == "centroids":
            return self.get_dataset(task_id, "train")

        raise ValueError(f"Unknown mode: {mode}")

    def _load_or_make_class_order(self):
        order_file = Path(f"{self.args['dataset']}_class_order.json")
        if self.args["run_mode"] == "centroids":
            order = list(range(self.args["total_cls"]))
            random.shuffle(order)
            save_json(order, order_file)
            return order
        return load_json(order_file)

    def _make_cls_per_task(self):
        base = self.args["total_cls"] // len(self.domains)
        cls_per_task = [base for _ in self.domains]
        for i in range(self.args["total_cls"] % len(self.domains)):
            cls_per_task[i] += 1
        return cls_per_task

    def _load_or_make_task_order(self):
        order_file = Path(f"{self.args['dataset']}_task_order.json")
        task_order = []
        offset = 0
        for n_cls in self.cls_per_task:
            task_order.append(self.class_order[offset : offset + n_cls])
            offset += n_cls
        if self.args["run_mode"] == "centroids":
            save_json(task_order, order_file)
        else:
            task_order = load_json(order_file)
        return task_order

    def _load_lists(self):
        for task_id, domain in enumerate(self.domains):
            train_entries = self._read_list_file(domain, train=True)
            test_entries = self._read_list_file(domain, train=False)
            selected = set(self.task_order[task_id])
            train_entries = [(p, y) for p, y in train_entries if y in selected]
            self.train_data[task_id], self.train_targets[task_id] = self._split_entries(train_entries)
            self.test_data[task_id], self.test_targets[task_id] = self._split_entries(test_entries)

    def _read_list_file(self, domain: str, train: bool):
        split = "train" if train else "test"
        path = Path(self.args["data_path"]) / f"{domain}_{split}.txt"
        entries = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                img_path, label = parts[0], int(parts[1])
                if not os.path.isabs(img_path):
                    img_path = str(Path(self.args["data_path"]) / img_path)
                entries.append((img_path, label))
        return entries

    @staticmethod
    def _split_entries(entries):
        if not entries:
            return np.asarray([]), np.asarray([])
        images, labels = zip(*entries)
        return np.asarray(images), np.asarray(labels)

    def _load_proto_data(self, task_id: int):
        domain = self.domains[task_id]
        root = Path(self.args["proto_data_path"]) / domain
        if not root.exists():
            raise FileNotFoundError(f"Prototype directory not found: {root}")

        name_to_idx = {name: idx for idx, name in self.classnames.items()}
        entries = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = name_to_idx[class_dir.name]
            for img in sorted(class_dir.iterdir()):
                if img.is_file():
                    entries.append((str(img), label))
        return self._split_entries(entries)
