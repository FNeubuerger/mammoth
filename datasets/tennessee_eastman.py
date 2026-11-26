# Copyright 2024-present
# Tennessee Eastman Process dataset for continual fault detection learning

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path


TEST_FAULT_ONSET = 160  # First disturbed row, zero-based, in Braatz d01_te..d21_te.


class TennesseeEastmanDataset(Dataset):
    """Label-pure windows for diagnosis (not onset detection) in the Braatz TEP files."""
    
    def __init__(self, data_path, fault_ids=None, train=True, window_size=50, stride=10,
                 norm_mean=None, norm_std=None, train_block="full", shuffle_time=False):
        """
        Args:
            data_path: Path to TEP dataset files
            fault_ids: List of fault IDs to include (0-21). If None, include all.
            train: Whether this is training or test data
            window_size: Number of timesteps in each sequence window
            stride: Stride for sliding window
            norm_mean, norm_std: Fixed per-channel normalization statistics
                (shape (52,)), computed once from normal-operation (fault 0)
                training data and shared across all fault files/splits. If
                None, they are computed here from fault 0's training file.
            train_block: "full" for final runs. For train-only recipe checks,
                "fit" uses the first 80% of each raw training trajectory and
                "validation" its last 20%. Windows never cross this boundary;
                both blocks use normalization fitted on the normal fit prefix.
            shuffle_time: Order ablation only: independently permute observations
                within each window, keeping all sensor channels together. Fixed
                split-specific generators do not consume training RNG state.
        """
        self.data_path = data_path
        self.fault_ids = fault_ids if fault_ids is not None else list(range(22))
        self.train = train
        self.shuffle_time = shuffle_time
        self.window_size = window_size
        self.stride = stride
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        if train_block not in ("full", "fit", "validation") or (not train and train_block != "full"):
            raise ValueError("train_block must be full, fit or validation (full for test files)")
        self.train_block = train_block
        
        # Load data
        self.windows = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess TEP data."""
        mode = 'train' if self.train else 'test'

        if len(self.fault_ids) == 0 or len(set(self.fault_ids)) != len(self.fault_ids):
            raise ValueError("fault_ids must be a non-empty sequence of unique fault IDs")
        if any(fault_id < 0 or fault_id > 21 for fault_id in self.fault_ids):
            raise ValueError(f"fault_ids must be between 0 and 21; got {self.fault_ids}")
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive integers")

        # Fit normalization once on fault-free training data, then reuse the
        # same transform for all faults and the independent test runs.
        if (self.norm_mean is None) != (self.norm_std is None):
            raise ValueError("Provide both norm_mean and norm_std or neither")
        if self.norm_mean is None:
            ref_path = os.path.join(self.data_path, 'd00.dat')
            ref_data = np.loadtxt(ref_path)
            if ref_data.ndim != 2:
                raise ValueError(f"Expected a 2-D TEP array in {ref_path}; got shape {ref_data.shape}")
            if ref_data.shape[0] == 52 and ref_data.shape[1] != 52:
                ref_data = ref_data.T
            if ref_data.shape[1] != 52 or not np.isfinite(ref_data).all():
                raise ValueError(f"Expected finite TEP samples with 52 features in {ref_path}; got shape {ref_data.shape}")
            if self.train_block != "full":
                ref_data = ref_data[:4 * len(ref_data) // 5]
            self.norm_mean = ref_data.mean(axis=0)
            self.norm_std = ref_data.std(axis=0) + 1e-8
        elif (
            np.asarray(self.norm_mean).shape != (52,)
            or np.asarray(self.norm_std).shape != (52,)
            or not np.isfinite(self.norm_mean).all()
            or not np.isfinite(self.norm_std).all()
            or np.any(np.asarray(self.norm_std) <= 0)
        ):
            raise ValueError("norm_mean and norm_std must be finite vectors of length 52 with positive std values")

        for fault_id in self.fault_ids:
            if self.train:
                filename = f'd{fault_id:02d}.dat'
            else:
                filename = f'd{fault_id:02d}_te.dat'
            
            filepath = os.path.join(self.data_path, filename)

            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Required TEP {'training' if self.train else 'test'} file is missing: {filepath}")

            try:
                data = np.loadtxt(filepath)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not read TEP data file {filepath}: {exc}") from exc

            if data.ndim != 2:
                raise ValueError(f"Expected a 2-D TEP array in {filepath}; got shape {data.shape}")
            if data.shape[0] == 52 and data.shape[1] != 52:
                data = data.T
            if data.shape[1] != 52 or not np.isfinite(data).all():
                raise ValueError(f"Expected finite TEP samples with 52 features in {filepath}; got shape {data.shape}")
            if not self.train and fault_id != 0:
                data = data[TEST_FAULT_ONSET:]
            elif self.train_block != "full":
                cut = 4 * len(data) // 5
                data = data[:cut] if self.train_block == "fit" else data[cut:]
            if len(data) < self.window_size:
                raise ValueError(
                    f"TEP file {filepath} has {len(data)} usable samples, fewer than window_size={self.window_size}"
                )

            # Keep fault-induced shifts by applying the fault-free training
            # statistics to every training and test fault file.
            data = (data - self.norm_mean) / self.norm_std

            num_windows = (len(data) - self.window_size) // self.stride + 1
            for i in range(num_windows):
                start_idx = i * self.stride
                end_idx = start_idx + self.window_size
                self.windows.append(data[start_idx:end_idx])
                self.labels.append(fault_id)

        if not self.windows:
            raise ValueError(f"No TEP windows were created from {self.data_path}")
        
        self.windows = np.array(self.windows, dtype=np.float32)
        if self.shuffle_time:
            rng = np.random.default_rng(0 if self.train else 1)
            for window in self.windows:
                rng.shuffle(window, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Mammoth v2 requires 'data' and 'targets' attributes
        self.data = torch.from_numpy(self.windows)
        self.targets = torch.from_numpy(self.labels)
        
        print(f"Loaded {len(self.windows)} windows for fault IDs {self.fault_ids} ({mode})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            window: (window_size, num_features) tensor
            label: fault class (0-21)
            window: same as first return (for compatibility with Mammoth)
        """
        window = self.data[idx]
        label = self.targets[idx]
        
        return window, label, window


class TennesseeEastmanContinual(ContinualDataset):
    """
    Tennessee Eastman Process for continual fault detection learning.

    Setting: Class-incremental learning, 11 tasks of 2 classes each.

    The original 22-class space (Normal + 21 faults) is grouped pairwise so
    that each task contains two classes, allowing within-task discrimination
    as well as a task-agnostic test across previously learned classes.

    Task layout (default ``pairwise`` grouping):
        Task 0: classes {0, 1}     (Normal,  Fault 1)
        Task 1: classes {2, 3}     (Fault 2, Fault 3)
        ...
        Task 10: classes {20, 21}  (Fault 20, Fault 21)

    Each task involves learning to discriminate two new fault classes while
    retaining the ability to discriminate all previously learned ones.
    """

    NAME = 'tennessee-eastman'
    SETTING = 'class-il'  # Class-incremental learning
    N_CLASSES_PER_TASK = 2  # Two faults per task
    N_TASKS = 11  # 22 classes / 2 per task
    SIZE = (52,)  # 52 process variables (time-series input)
    
    def __init__(self, args, tep_shuffle_time: bool = False):
        super().__init__(args)
        self.shuffle_time = tep_shuffle_time
        self.window_size = getattr(args, 'tep_window_size', 50)
        self.stride = getattr(args, 'tep_stride', 10)
        self._task_idx = 0
        
    def get_data_loaders(self):
        """Get train and test loaders for current task."""
        # Load ALL faults and let store_masked_loaders do the task-based filtering
        # This is the correct way for Mammoth v2 - it will mask based on targets
        train_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # Load ALL faults
            train=True,
            window_size=self.window_size,
            stride=self.stride,
            shuffle_time=self.shuffle_time
        )
        
        test_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # Load ALL faults
            train=False,
            window_size=self.window_size,
            stride=self.stride,
            norm_mean=train_dataset.norm_mean,
            norm_std=train_dataset.norm_std,
            shuffle_time=self.shuffle_time
        )
        
        # Use store_masked_loaders to properly wrap datasets for Mammoth v2
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        
        # Don't append test_loaders here - parent class handles it
        self.train_loader = train
        
        return train, test
    
    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        """
        Return CfC backbone for TEP.
        
        Input: (window_size, 52) - sequence of 52 process variables
        Output: 22 classes (normal + 21 faults)
        """
        return "tepcfc"
    
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
        return TennesseeEastmanContinual.get_batch_size()


class TennesseeEastmanJoint(ContinualDataset):
    """
    Tennessee Eastman with joint training (all faults at once).
    Used as upper bound for comparison with incremental learning.
    """
    
    NAME = 'tennessee-eastman-joint'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 22  # All faults in single task
    N_TASKS = 1
    SIZE = (52,)  # 52 process variables
    N_TASKS = 1
    
    def __init__(self, args):
        super().__init__(args)
        self.window_size = getattr(args, 'tep_window_size', 50)
        self.stride = getattr(args, 'tep_stride', 10)
        
    def get_data_loaders(self):
        """Get train and test loaders with all faults."""
        train_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # All faults
            train=True,
            window_size=self.window_size,
            stride=self.stride
        )
        
        test_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # All faults
            train=False,
            window_size=self.window_size,
            stride=self.stride,
            norm_mean=train_dataset.norm_mean,
            norm_std=train_dataset.norm_std
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        
        return train_loader, test_loader
    
    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        return "tepcfc"
    
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
        return TennesseeEastmanJoint.get_batch_size()
