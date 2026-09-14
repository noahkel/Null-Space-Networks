from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class EllipsesGTInitDataset(Dataset):
    """
    Loads (ground_truth, init_reconstruction, sinogram) triples from a data
    directory produced by create_phantom_data.py. The init reconstruction is
    the truncated-SVD pseudoinverse, stored under ``pinv/``.
    """

    def __init__(
        self,
        root: Path,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.root = Path(root)
        self.gt_dir = self.root / "gt"
        self.init_dir = self.root / "pinv"
        self.sino_dir = self.root / "sino"

        self.files = sorted(f.name for f in self.gt_dir.glob("*.npy"))

        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fname = self.files[idx]

        x_gt = np.load(self.gt_dir / fname)
        x_init = np.load(self.init_dir / fname)
        y_delta = np.load(self.sino_dir / fname)

        # (H, W) -> (1, H, W)
        x_gt = torch.from_numpy(x_gt).unsqueeze(0).to(self.dtype)
        x_init = torch.from_numpy(x_init).unsqueeze(0).to(self.dtype)
        y_delta = torch.from_numpy(y_delta).unsqueeze(0).to(self.dtype)

        if self.device is not None:
            x_gt = x_gt.to(self.device, non_blocking=True)
            x_init = x_init.to(self.device, non_blocking=True)
            y_delta = y_delta.to(self.device, non_blocking=True)

        return x_gt, x_init, y_delta


def get_ellipse_dataloader(
    batch_size: int,
    data_root: Path = Path("ellipses_out"),
    split: str = "train",          # "train", "val" or "test"
    n_train: int = 3500,
    n_val: int = 500,
    n_test: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> DataLoader:
    """Train/val/test loader over a data directory. The split is by index and
    therefore deterministic: the first n_train samples train, the next n_val
    validate, the next n_test test.

    Validation and test are kept apart on purpose. Training keeps the checkpoint
    with the lowest validation loss; if that loss were measured on the test set,
    every number later reported on the test set would come from the checkpoint
    chosen to look best there."""
    dataset = EllipsesGTInitDataset(root=Path(data_root), device=device)

    indices = np.arange(len(dataset))
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    if split not in bounds:
        raise ValueError("split must be 'train', 'val' or 'test'")
    lo, hi = bounds[split]
    if hi > len(dataset):
        raise ValueError(f"split '{split}' needs samples [{lo}, {hi}) but the "
                         f"dataset has {len(dataset)}")
    subset = Subset(dataset, indices[lo:hi])
    do_shuffle = shuffle and split == "train"

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        drop_last=False,
    )
