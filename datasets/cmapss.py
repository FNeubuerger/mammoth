# Copyright 2024-present
# NASA C-MAPSS Turbofan Engine Degradation dataset for continual learning.
#
# Source: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
#         Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
#         PHM08. Host: NASA Open Data Portal (public domain).
#         https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
#
# C-MAPSS is natively a regression benchmark (predict Remaining Useful Life,
# RUL, in cycles). This adaptation caps RUL at 130 cycles, bins it into three
# health states, and treats each FD00X subset as a domain-incremental task.
# The same three labels are shared across domains; they are not duplicated
# into task-specific classes.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path

FD_SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
N_FEATURES = 24  # 3 operational settings + 21 sensor measurements
WINDOW_SIZE = 30
STRIDE = 5
RUL_CAP = 130
RUL_HEALTHY = 100
RUL_CRITICAL = 30


def _bin_rul(rul: np.ndarray) -> np.ndarray:
    rul = np.minimum(rul, RUL_CAP)
    bins = np.zeros_like(rul, dtype=np.int64)
    bins[(rul < RUL_HEALTHY) & (rul >= RUL_CRITICAL)] = 1
    bins[rul < RUL_CRITICAL] = 2
    return bins


def _load_subset_windows(data_path, subset, train, feat_mean=None, feat_std=None):
    """Load one FD00X subset and return (windows, rul_bins, feat_mean, feat_std)."""
    fname = f"{'train' if train else 'test'}_{subset}.txt"
    raw = np.loadtxt(os.path.join(data_path, fname))
    unit_ids = raw[:, 0].astype(int)
    cycles = raw[:, 1].astype(int)
    # columns: 0=unit, 1=cycle, 2-4=operational settings, 5..=21 sensors
    features = raw[:, 2:2 + N_FEATURES].astype(np.float32)

    if not train:
        rul_final = np.atleast_1d(
            np.loadtxt(os.path.join(data_path, f'RUL_{subset}.txt')).astype(np.float32)
        )

    if feat_mean is None or feat_std is None:
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-8
    features = (features - feat_mean) / feat_std

    windows, rul_bins = [], []
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        unit_feats = features[mask]
        unit_cycles = cycles[mask]
        max_cycle = unit_cycles.max()
        if train:
            unit_rul = (max_cycle - unit_cycles).astype(np.float32)
        else:
            final_rul = rul_final[uid - 1]
            unit_rul = (final_rul + (max_cycle - unit_cycles)).astype(np.float32)

        if len(unit_feats) < WINDOW_SIZE:
            continue
        for start in range(0, len(unit_feats) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            windows.append(unit_feats[start:end])
            rul_bins.append(unit_rul[end - 1])

    windows = np.array(windows, dtype=np.float32)
    rul_bins = _bin_rul(np.array(rul_bins, dtype=np.float32))
    return windows, rul_bins, feat_mean, feat_std


class CMAPSSDataset(Dataset):
    """C-MAPSS sliding windows for one or more FD00X operating domains."""

    def __init__(self, data_path, train=True, subset_stats=None, subsets=None):
        self.data_path = data_path
        self.train = train
        self.subset_stats = subset_stats or {}
        self.subsets = list(FD_SUBSETS if subsets is None else subsets)
        if not self.subsets or len(set(self.subsets)) != len(self.subsets):
            raise ValueError("subsets must be a non-empty sequence of unique FD00X names")
        unknown = set(self.subsets) - set(FD_SUBSETS)
        if unknown:
            raise ValueError(f"Unknown C-MAPSS subsets: {sorted(unknown)}")
        self._load_data()

    def _load_data(self):
        all_windows, all_labels = [], []
        for subset in self.subsets:
            mean, std = self.subset_stats.get(subset, (None, None))
            windows, bins, mean, std = _load_subset_windows(
                self.data_path, subset, self.train, feat_mean=mean, feat_std=std)
            self.subset_stats[subset] = (mean, std)
            all_windows.append(windows)
            all_labels.append(bins)

        windows = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        self.data = torch.from_numpy(windows)
        self.targets = torch.from_numpy(labels)
        print(f"Loaded {len(labels)} C-MAPSS windows from {', '.join(self.subsets)} "
              f"({'train' if self.train else 'test'})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, x


class CMAPSSContinual(ContinualDataset):
    """
    NASA C-MAPSS turbofan degradation under domain-incremental learning.

    Each FD00X subset is a task/domain. The same three RUL health-state
    labels are used in every domain; no task-specific copies are introduced.
    """

    NAME = 'cmapss'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 3
    N_TASKS = 4
    N_CLASSES = 3
    REPORT_CLASS_RECALL = True
    SIZE = (N_FEATURES,)

    def __init__(self, args):
        super().__init__(args)
        self._task_idx = 0

    def get_data_loaders(self):
        if self._task_idx >= len(FD_SUBSETS):
            raise RuntimeError("All C-MAPSS operating domains have already been loaded")
        subset = FD_SUBSETS[self._task_idx]
        train_dataset = CMAPSSDataset(
            data_path=os.path.join(base_path(), 'CMAPSS'),
            train=True,
            subsets=[subset],
        )
        test_dataset = CMAPSSDataset(
            data_path=os.path.join(base_path(), 'CMAPSS'),
            train=False,
            subset_stats=train_dataset.subset_stats,
            subsets=[subset],
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        self.train_loader = train
        self._task_idx += 1
        return train, test

    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        return "cmapsscfc"

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 32

    @staticmethod
    def get_minibatch_size() -> int:
        return CMAPSSContinual.get_batch_size()
