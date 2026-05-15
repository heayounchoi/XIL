from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import accuracy
from .utils import count_parameters, tensor_to_numpy, unwrap


class Trainer:
    def __init__(self, args, classnames):
        self.args = args
        self.classnames = classnames
        self.cur_task = 0
        self.device = args["device"][0]
        self.gpus = args["device"]
        self.all_keys = {}
        self.task_order = None
        self.curves = self._new_curves()

    @staticmethod
    def _new_curves():
        return {
            "top1": [],
            "top5": [],
            "BiDoT": [],
            "class": [],
            "domain": [],
            "BiDoT_domain": [],
        }

    def incremental_step(self, model, data_manager):
        self.task_order = data_manager.task_order
        train_dataset = data_manager.get_dataset(self.cur_task, "train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
        )

        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model, self.gpus)

        if self.args["run_mode"] == "centroids":
            self.extract_centroids(model, train_dataset, data_manager.domains[self.cur_task])

        if self.args["run_mode"] == "train":
            self.train_one_task(model, train_loader)
            proto_dataset = data_manager.get_dataset(self.cur_task, "proto")
            proto_loader = DataLoader(
                proto_dataset,
                batch_size=self.args["batch_size"],
                shuffle=True,
                num_workers=self.args["num_workers"],
            )
            self.replace_fc_with_prototypes(model, proto_loader, proto_dataset, self.cur_task)
            self.save_checkpoint(model)
            self.evaluate_seen_tasks(model, data_manager)

        if self.args["run_mode"] == "eval":
            self.evaluate_seen_tasks(model, data_manager)

        self.cur_task += 1
        unwrap(model).update_task()
        return model

    def train_one_task(self, model, train_loader):
        model.to(self.device)
        model.train()
        self._freeze_except_current_task(model)

        if self.cur_task == 0:
            lr = self.args["init_lr"]
            weight_decay = self.args["init_weight_decay"]
            epochs = self.args["init_epoch"]
        else:
            lr = self.args["lrate"]
            weight_decay = self.args["weight_decay"]
            epochs = self.args["epochs"]

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in tqdm(range(epochs), desc=f"Task {self.cur_task}"):
            losses, correct, total = 0.0, 0, 0
            model.eval()
            for _, images, targets in train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor_to_numpy(correct) * 100 / total, decimals=2)
            logging.info(
                "Task %d, Epoch %d/%d => Loss %.3f, Train_accy %.2f",
                self.cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )

    def _freeze_except_current_task(self, model):
        cur = unwrap(model).numtask
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            if f"classifier_pool.{cur}" in name or f"prompt_pool.{cur}" in name:
                param.requires_grad_(True)

    @torch.no_grad()
    def replace_fc_with_prototypes(self, model, loader, dataset, domain_id: int):
        model.eval().to(self.device)
        net = unwrap(model)
        features, labels, raw_features = [], [], []

        for _, images, targets in loader:
            images = images.to(self.device)
            concat, raw = net.concat_features(images, domain_id)
            features.append(concat.cpu())
            raw_features.append(raw.cpu())
            labels.append(targets.cpu())

        features = torch.cat(features, dim=0)
        raw_features = torch.cat(raw_features, dim=0)
        labels = torch.cat(labels, dim=0)

        for class_id in np.unique(dataset.labels):
            idx = (labels == int(class_id)).nonzero().squeeze(-1)
            proto = features[idx].mean(0)
            domain_key = raw_features[idx].mean(0)
            self.all_keys.setdefault(domain_id, []).append(domain_key)
            net.proto_classifier_pool[domain_id].weight.data[int(class_id)] = proto

    @torch.no_grad()
    def extract_centroids(self, model, train_dataset, domain_name: str):
        output_dir = Path(self.args["centroid_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        model.eval().to(self.device)
        net = unwrap(model)

        class_transform_dataset = train_dataset
        for class_id in sorted(class_transform_dataset.classwise_data.keys()):
            class_dataset = class_transform_dataset.get_classwise_dataset(class_id)
            loader = DataLoader(class_dataset, batch_size=len(class_dataset), shuffle=False, num_workers=self.args["num_workers"])

            for images, paths in loader:
                images = images.to(self.device)
                feats = net.extract_vector(images).cpu().numpy()
                paths = list(paths)

            picked_paths = self._pick_centroid_paths(feats, paths)
            class_name = self.classnames[int(class_id)]
            class_dir = output_dir / domain_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for src in picked_paths:
                shutil.copy(src, class_dir / os.path.basename(src))

            logging.info("Saved %d centroids for %s/%s", len(picked_paths), domain_name, class_name)

    def _pick_centroid_paths(self, features, paths):
        if len(features) <= self.args["num_centroids"]:
            return paths

        mean_feature = features.mean(axis=0)
        distances = np.linalg.norm(features - mean_feature, axis=1)
        cutoff = np.percentile(distances, self.args["outlier_cutoff"])
        inlier_idx = np.where(distances <= cutoff)[0]
        inlier_features = features[inlier_idx]
        inlier_paths = [paths[i] for i in inlier_idx]

        if len(inlier_features) <= self.args["num_centroids"]:
            return inlier_paths

        kmeans = KMeans(n_clusters=self.args["num_centroids"], random_state=0).fit(inlier_features)
        picked = []
        for centroid in kmeans.cluster_centers_:
            distances = np.linalg.norm(inlier_features - centroid, axis=1)
            picked.append(inlier_paths[int(np.argmin(distances))])
        return picked

    def evaluate_seen_tasks(self, model, data_manager):
        self.curves = self._new_curves()
        for task_id in range(self.cur_task + 1):
            if self.args["run_mode"] == "eval" and task_id < self.args.get("resume_eval_task", 999):
                logging.info("Resume eval: skip task %d", task_id)
                continue

            test_dataset = data_manager.get_dataset(task_id, "test")
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args["test_batch_size"],
                shuffle=False,
                num_workers=self.args["num_workers"],
            )
            acc = self.eval_task(model, test_loader, task_id)
            self._log_eval(acc)

    def eval_task(self, model, loader, task_id: int):
        y_pred, y_true, dom_pred, dom_true = self._predict(model, loader, task_id)
        grouped = accuracy(y_pred.T[0], y_true, dom_pred, dom_true, self.task_order)
        return {
            "grouped": grouped,
            "top1": grouped["total"],
            "top5": np.around((y_pred.T == np.tile(y_true, (5, 1))).sum() * 100 / len(y_true), decimals=2),
        }

    @torch.no_grad()
    def _predict(self, model, loader, task_id: int):
        model.eval().to(self.device)
        net = unwrap(model)
        y_pred, y_true, dom_pred, dom_true = [], [], [], []
        avg_keys = self._average_domain_keys(task_id)

        for _, images, targets, domains, _ in loader:
            images = images.to(self.device)
            raw = net.raw_features(images)
            selected_domains = self._select_domains(raw, avg_keys)
            dom_pred.append(np.asarray(selected_domains))
            dom_true.append(domains.numpy())

            grouped = defaultdict(list)
            for idx, (img, label, selected_domain) in enumerate(zip(images.cpu(), targets, selected_domains)):
                grouped[(int(label), int(selected_domain))].append((idx, img))

            batch_results = {}
            for (_, selected_domain), items in grouped.items():
                indices, group_imgs = zip(*items)
                group_imgs = torch.stack(group_imgs).to(self.device)
                features, _ = net.concat_features(group_imgs, selected_domain)
                logits = net.proto_classifier_pool[selected_domain](features)
                for index, logit in zip(indices, logits):
                    batch_results[index] = logit

            logits = torch.stack([batch_results[i] for i in sorted(batch_results.keys())])
            y_pred.append(torch.topk(logits, k=5, dim=1, largest=True, sorted=True)[1].cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(dom_pred), np.concatenate(dom_true)

    def _average_domain_keys(self, task_id: int):
        avg_keys = {}
        for key, vectors in self.all_keys.items():
            if key <= task_id:
                avg_keys[key] = torch.stack(vectors).to(self.device).mean(dim=0)
        if not avg_keys:
            raise RuntimeError("No domain keys found. Run train first or load a checkpoint with domain_keys.")
        return avg_keys

    @staticmethod
    def _select_domains(features, avg_keys):
        selected = []
        for feature in features:
            distances = {key: torch.norm(feature - avg, p=2) for key, avg in avg_keys.items()}
            selected.append(min(distances, key=distances.get))
        return selected

    def _log_eval(self, acc):
        grouped = acc["grouped"]
        self.curves["top1"].append(acc["top1"])
        self.curves["top5"].append(acc["top5"])
        for key in ["BiDoT", "class", "domain", "BiDoT_domain"]:
            self.curves[key].append(grouped[key])

        logging.info("grouped: %s", grouped)
        logging.info("top1 curve: %s", self.curves["top1"])
        logging.info("top5 curve: %s", self.curves["top5"])
        logging.info("BiDoT curve: %s", self.curves["BiDoT"])

    def save_checkpoint(self, model):
        path = Path(self.args["log_dir"]) / f"task_{self.cur_task}.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "domain_keys": self.all_keys,
                "task": self.cur_task,
                "task_order": self.task_order,
            },
            path,
        )
        logging.info("Saved checkpoint: %s", path)
