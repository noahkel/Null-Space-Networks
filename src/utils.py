import torch
import numpy as np
from pathlib import Path
from src.unet import UNet
from src.wrappers import RESNET, NSN, DPNSN, DPNSN_RES
from typing import List, Union, Dict, Optional
from src.radon import _RadonBase

import torch.nn as nn
import math
try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def rel_l2_np(x: np.ndarray, y: np.ndarray) -> float:
    # rel-L2(x, y) = ||x - y||_2 / max(||y||_2, floor),   floor = 1e-3·sqrt(|y|)
    # (relative Euclidean error; the floor guards near-zero references).
    num = np.linalg.norm(x - y)
    den = np.linalg.norm(y)
    # Clamp denominator to prevent blow-up on near-zero GT images.
    # Sub-pixel ellipses from single_ellipse_generator produce all-zero discrete
    # phantoms (ODL midpoint sampling); without this guard rel_l2 hits ~1e12.
    # For 128×128: floor = 1e-3 * 128 ≈ 0.128, capping per-sample rel_l2 at ~30.
    den = max(den, 1e-3 * float(np.sqrt(y.size)))
    return float(num / den)


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    # PSNR = 20·log10(range) - 10·log10(MSE),   range = max(y) - min(y),
    # MSE = mean((x - y)^2).   Higher is better.
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float("inf")
    data_range = float(y.max() - y.min())
    if data_range <= 0.0:
        data_range = 1.0
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    data_range = float(y.max() - y.min())
    if data_range <= 0.0:
        data_range = 1.0
    return float(sk_ssim(y, x, data_range=data_range))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error between reconstruction x and reference y."""
    return float(np.mean(np.abs(x - y)))


def max_abs_err(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum absolute (L-inf) pixel error between x and y."""
    return float(np.max(np.abs(x - y))) if x.size else float("nan")


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Plain root-mean-square error between reconstruction x and reference y
    (unnormalised, in image units)."""
    return float(np.sqrt(np.mean((x - y) ** 2)))


def nrmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root-mean-square error normalised by the reference data range
    (y.max - y.min). Complements rel_l2 (which normalises by ‖y‖)."""
    # NRMSE = sqrt(mean((x - y)^2)) / (max(y) - min(y))   (range-normalised RMSE;
    # complements rel_l2 which normalises by ||y||).
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    data_range = float(y.max() - y.min())
    if data_range <= 0.0:
        data_range = 1.0
    return rmse / data_range


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is (B, 1, H, W)."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The training objective: plain MSE against the ground truth."""
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def decompose_error(
    e: torch.Tensor,
    radon: "_RadonBase",
    iters: int = 50,
    tol: float = 1e-6,
) -> tuple:
    """
    Orthogonal decomposition of image-space error e:

      e_ran = A_la^+ A_la e            — projection onto range(A_la^T) = row(A_la)
      e_nul = (I - A_la^+ A_la) e = e - e_ran   — component in null(A_la)
    The two are orthogonal, so the energy splits exactly:
      ||e||_2^2 = ||e_ran||_2^2 + ||e_nul||_2^2.

    Delegates to radon.decompose_error: MatrixRadonAdapter uses exact SVD,
    other adapters use CG.  Returns (e_ran, e_nul) as detached CPU tensors.
    """
    e_ran, e_nul = radon.decompose_error(e, iters=iters, tol=tol)
    return e_ran.detach().cpu(), e_nul.detach().cpu()


def build_models(
    which: List[str],
    radon: _RadonBase,
) -> Dict[str, nn.Module]:
    models: Dict[str, nn.Module] = {}
    for name in which:
        name = name.lower()
        if name == "resnet":
            models[name] = RESNET(unet=UNet(in_channels=1, out_channels=1))
        elif name == "nsn":
            models[name] = NSN(unet=UNet(in_channels=1, out_channels=1), radon=radon)
        else:
            raise ValueError(
                f"Unknown model '{name}'. Use one of: resnet, nsn, dpnsn, dpnsn_res")
    return models