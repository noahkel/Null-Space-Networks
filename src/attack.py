#!/usr/bin/env python3
"""Adversarial attack suite for limited-angle Radon reconstruction models.

This is the *attack/compute* half of the attack/visualisation split. It owns:

  * the attack primitives (norm projections, gradient normalisation),
  * the attack objectives and the PGD attack that maximises them,
  * the model adapter that turns a sinogram into a prediction via A^+,
  * per-sample metric evaluation and aggregation,
  * the suite orchestration that attacks every trained model.

Each run writes its numeric artifacts to disk (per_sample_metrics.csv, attack_output.npz
with the adversarial sinogram + perturbation, examples.npz/.json,
transfer.npz/.json, summary.json, lipschitz.json)
``visualise.py`` rebuilds every figure from those artifacts, so plots can be
regenerated without re-running the attack.

The shared on-disk contract lives in ``src/artifacts.py`` 

The thin top-level ``attack.py`` is the CLI entry point for this module.
"""
import argparse
import csv
import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.ellipse_dataloader import get_ellipse_dataloader
from src.radon import MatrixRadonAdapter
from src.utils import (
    build_models,
    decompose_error,
    mae,
    max_abs_err,
    nrmse,
    psnr,
    rel_l2_np,
    rmse,
    set_seed,
    ssim,
    to_4d,
)
from src.artifacts import (
    write_rows_bundle,
    write_transfer_bundle,
)

F64 = False
SPARSE = False
SEED = 42

SUITE_STEPS = 50
SUITE_WORST = 3
SUITE_EXAMPLES = 10
SUITE_TRANSFER_SAMPLES = 5
SUITE_RESTARTS = 2

NUM_WORKERS = 4
BATCH_SIZE = 32

# Must match train.py: the test split is the one neither training nor
# checkpoint selection has seen.
N_TRAIN = 3500
N_VAL = 500
N_TEST = 1000
SPLIT = "test"

# The only init reconstruction: the truncated-SVD pseudoinverse A_la^+.
# Kept as a name because it is also the on-disk directory the artifacts and
# the trained checkpoints live under.
INIT_NAME = "pinv"


SUCCESS_MSE_FACTOR = 2.0


def stage_seed(*parts: str) -> int:
    """The seed of one stage of a run, e.g. stage_seed("suite", attack_name).

    Every stage reseeds before it draws, so its random starts do not depend on
    which stages ran before it: adding an attack to the suite leaves the numbers
    of the others unchanged. Derived with crc32 because Python's hash() of a
    string changes from one interpreter to the next."""
    return (SEED + zlib.crc32("/".join(parts).encode("utf-8"))) % 2 ** 31


# --------------------------------------------------------------------------- #
# Small tensor helpers.
# --------------------------------------------------------------------------- #
def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()

def l2_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)

Budget = Union[float, torch.Tensor]

def as_eps_batch(eps: Budget, x: torch.Tensor) -> torch.Tensor:
    """The one representation of a budget: a broadcastable [B,1,1,1] tensor. """
    if torch.is_tensor(eps):
        vec = eps.to(device=x.device, dtype=x.dtype).reshape(-1)
    else:
        vec = torch.full((x.shape[0],), float(eps), device=x.device, dtype=x.dtype)
    return vec.clamp_min(0.0).view(-1, 1, 1, 1)

def proj_l2_ball(delta: torch.Tensor, eps: Budget) -> torch.Tensor:
    # Π_{||·||≤ε}(δ) = δ · min(1, ε / ||δ||_2)   (per sample; radial shrink to the ball).
    eps_b = as_eps_batch(eps, delta)
    norms = l2_norm_batch(delta).clamp_min(1e-12).view(-1, 1, 1, 1)
    return delta * torch.minimum(torch.ones_like(norms), eps_b / norms)

def project_delta(delta: torch.Tensor, eps: Budget,
                  projector: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Projection onto the feasible set S = range(U_k) ∩ {||δ||_2 ≤ ε}."""
    return projector(proj_l2_ball(projector(delta), eps))

def suite_eps_batch(y_clean: torch.Tensor, eps_nominal: float) -> torch.Tensor:
    """Per-sample budget eps_i for one batch.

    eps_i = eps_nominal * ||y_i||_2 — per sample, so a bright and a faint
    sinogram are attacked at the same *relative* strength."""
    return eps_nominal * l2_norm_batch(y_clean)

def suite_step_size(eps: Budget, steps: int = SUITE_STEPS) -> Budget:
    """PGD step alpha: the classic 2.5*eps/steps, in the same units as the
    budget. Since the budget is per sample, so is the step."""
    return 2.5 * eps / max(steps, 1)

def per_sample_loss(loss_map: torch.Tensor) -> torch.Tensor:
    """Mean over every axis but the batch axis: one score per sample."""
    return loss_map.reshape(loss_map.shape[0], -1).mean(dim=1)

def reduce_loss(loss_map: torch.Tensor) -> torch.Tensor:
    if loss_map.ndim <= 1:
        return loss_map.mean()
    return per_sample_loss(loss_map).mean()

def random_start(y_clean: torch.Tensor, eps: Budget,
                 projector: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """A random point of S = range(P) ∩ B_eps, per sample.

    The direction is a projected Gaussian, which is isotropic within range(P),
    normalised to unit length; the radius is uniform on [0, eps_i]. Drawing the
    Gaussian and merely clipping it to the ball instead would start every sample
    at radius min(||g||, eps) -- a number set by the sinogram dimension, not by
    the budget -- so all restarts would begin on the same sphere."""
    d = projector(torch.randn_like(y_clean))
    d = d / l2_norm_batch(d).clamp_min(1e-12).view(-1, 1, 1, 1)
    radius = torch.rand(y_clean.shape[0], device=y_clean.device,
                        dtype=y_clean.dtype).view(-1, 1, 1, 1)
    return d * radius * as_eps_batch(eps, y_clean)

def per_example_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).reshape(x.shape[0], -1).mean(dim=1)

def confidence_interval_95(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    # 95% CI half-width for the mean: 1.96 · s / sqrt(n),  s = sample std (ddof=1).
    half_width = float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, half_width

@dataclass
class AttackResult:
    y_adv: torch.Tensor
    delta: torch.Tensor
    runtime_sec: float

# --------------------------------------------------------------------------- #
# Model adapter.
# --------------------------------------------------------------------------- #
class ModelAttackAdapter:
    """Sinogram -> prediction, as one differentiable chain.

    The chain is  y -> P_ran y -> x_init = A_la^+ P_ran y -> model(x_init).
    Keeping the initial reconstruction outside the model means the same attack
    code drives both architectures with no branching inside the attack."""

    def __init__(
        self,
        model: nn.Module,
        radon: MatrixRadonAdapter,
        projector: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.model = model
        self.radon = radon
        self.projector = projector

    def forward(self, y_adv: torch.Tensor, project: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sinogram -> (prediction, init reconstruction, projected sinogram).

        ``project`` is off inside the PGD loop, where the caller has already
        projected onto range(A_la) and needs the graph to start at the
        perturbed sinogram itself."""
        if project:
            y_adv = self.projector(y_adv)
        x_init = self.radon.backward_la(y_adv)
        return self.model(x_init), x_init, y_adv

# --------------------------------------------------------------------------- #
# Attack objective + algorithms.
# --------------------------------------------------------------------------- #
def attack_objective(
    pred: torch.Tensor,
    x_gt: torch.Tensor,
    objective: str,
    radon=None,
    target: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Attack loss to be *maximised*.

    ``reduction="mean"`` returns the batch mean that gradients are taken of;
    ``reduction="none"`` returns one score per sample, which is what selecting
    the best restart per sample needs.

    The plain "mse" objectives reward total reconstruction
    error. On a data-consistent model NSN the cheapest way to grow that
    error is to inject error into the *range* (measured) component, which the
    network reproduces by design — so the attack looks strong but is structurally
    trivial and not comparable to what the same attack does to ResNet.

    The "null" objectives instead reward only the *null-space* component of the
    error, ‖P_null (pred - target)‖². P_null is the image-domain projector onto
    null(A_la) (radon.proj_null_image, differentiable). This forces the optimiser
    to corrupt exactly the component the network is responsible for — the part
    that can hallucinate/break structure the way a ResNet attack does — rather
    than taking the free range-space channel.

      null        : ‖P_null (pred - x_gt)‖²            (null-space error vs GT)
    """
    #   mse         :  mean(e^2)
    #   range       :  mean( (P_ran e)^2 )             (measured / data-consistent channel)
    #   null        :  mean( (P_N e)^2 )               (structural / learned channel)
    #   zero        : -mean(pred^2)                    (targeted: drive pred -> 0)
    #   target      : -mean((pred - t)^2)              (targeted: drive pred -> t)
    if reduction == "mean":
        red = reduce_loss
    elif reduction == "none":
        red = per_sample_loss
    else:
        raise ValueError(f"Unknown reduction '{reduction}' (mean or none)")

    if objective == "mse":
        return red((pred - x_gt) ** 2)

    if objective == "range":
        # Null-space *complement*: reward only the range (measured) error component
        if radon is None:
            raise ValueError("Objective 'range' requires a radon operator.")
        err = pred - x_gt
        err_range = err - radon.proj_null_image(err)
        return red(err_range ** 2)

    if objective == "null":
        # Null-space: reward only the null-space error component
        if radon is None:
            raise ValueError("Objective 'null' requires a radon operator.")
        err = pred - x_gt
        return red(radon.proj_null_image(err) ** 2)

    if objective == "zero":
        # Targeted attack: drive the reconstruction toward the zero image
        return -red(pred ** 2)

    if objective == "target":
        # General targeted attack: drive the reconstruction toward an arbitrary supplied image
        if target is None:
            raise ValueError("Objective 'target' requires a target image tensor.")
        return -red((pred - target.detach()) ** 2)

    raise ValueError(f"Unknown objective '{objective}'")

def pgd_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: Budget,
    alpha: Budget,
    objective: str,
    target: Optional[torch.Tensor] = None,
) -> AttackResult:
    """Projected gradient ascent on the sinogram perturbation -- the one attack
    the suite runs.

    The feasible set is S = range(U_k) intersect {||delta||_2 <= eps}: the
    perturbation must itself be a measurement the scanner could have taken
    (anything else is not a perturbation an attacker could make) and stay
    inside the L2 budget. Each step is

        delta <- Pi_S( delta + alpha * normalize(grad_delta loss) )

    started from a random point of S (see random_start). Of the restarts, the
    best one is kept *per sample*: a batch-level choice would hand every sample
    the restart that is best on average, which is weaker for any sample whose
    own best came from another restart, and would understate how attackable the
    model is.

    ``objective`` is maximised; see attack_objective. ``target`` supplies the
    reference image for the targeted 'target' objective and is ignored by the
    others.
    """
    start = time.perf_counter()
    radon = adapter.radon
    alpha_b = as_eps_batch(alpha, y_clean)
    best_delta = torch.zeros_like(y_clean)
    best_score = torch.full((y_clean.shape[0],), -float("inf"),
                            device=y_clean.device, dtype=y_clean.dtype)

    def loss_of(pred, reduction="mean"):
        return attack_objective(pred, x_gt, objective, radon=radon, target=target,
                                reduction=reduction)

    for _ in range(SUITE_RESTARTS):
        delta = random_start(y_clean, eps, adapter.projector)

        for _ in range(SUITE_STEPS):
            y_adv = (y_clean + delta).detach().requires_grad_(True)
            pred, _, _ = adapter.forward(y_adv, project=False)
            grad = torch.autograd.grad(loss_of(pred), y_adv)[0]
            with torch.no_grad():
                g = grad / l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)
                delta = project_delta(delta + alpha_b * g, eps, adapter.projector)

        with torch.no_grad():
            pred, _, _ = adapter.forward(y_clean + delta, project=False)
            score = loss_of(pred, reduction="none")
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best_delta = torch.where(better.view(-1, 1, 1, 1), delta, best_delta)

    best_delta = best_delta.detach()
    return AttackResult(y_adv=(y_clean + best_delta).detach(), delta=best_delta,
                        runtime_sec=time.perf_counter() - start)

# --------------------------------------------------------------------------- #
# Data / model setup.
# --------------------------------------------------------------------------- #
def load_summary(data_root: str) -> Dict:
    summary_path = Path(data_root) / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_radon(summary: Dict, device: torch.device,
                dtype: torch.dtype = torch.float32, dense: bool = True):
    return MatrixRadonAdapter(
        resolution=int(summary["img_size"]),
        angles=np.asarray(summary["angles"], dtype=np.float64),
        det_count=int(summary["det_count"]),
        dx=float(summary["dx"]),
        estimate_norm=False,
        device=device,
        dtype=dtype,
        dense=dense,
        phi=tuple(summary["phi"]),  # already in radians
        svd_threshold=float(summary["svd_threshold"]),
        cache_dir="radon_cache",
    )

def load_model_checkpoint(
    model_name: str,
    radon,
    device: torch.device,
    model_dir: Optional[str] = None,
) -> nn.Module:
    base = Path(model_dir) if model_dir else Path(".")
    ckpt_path = base / f"init_{INIT_NAME}" / "checkpoints" / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_name}'. Searched:\n  {ckpt_path}"
        )

    model = build_models([model_name], radon=radon)[model_name].to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# --------------------------------------------------------------------------- #
# Per-sample metrics.
# --------------------------------------------------------------------------- #
def target_distance_metrics(
    x_gt: torch.Tensor,
    target: torch.Tensor,
    clean_pred: torch.Tensor,
    adv_pred: torch.Tensor,
    clean_init: torch.Tensor,
    adv_init: torch.Tensor,
    radon=None,
) -> Dict[str, float]:
    """How close a targeted attack brought one sample to its target t.

    Every distance is taken relative to ||x_gt - t||, the distance the attack
    has to cover:

        tgt_dist(x) = ||x - t|| / ||x_gt - t||,

    1 at the ground truth and 0 at the target, so the attack towards the zero
    image and the attack towards another sample read on one scale. The
    denominator has the floor of rel_l2_np. tgt_closed is the fraction of the
    clean reconstruction's distance the attack closed, and tgt_success whether
    the attacked reconstruction ends closer to the target than to the ground
    truth. With a radon operator the distance is also split into its range and
    null-space component: for the NSN the range component moves only through
    A^+ delta, whatever the network does. Arguments are single samples
    (1, 1, H, W)."""
    gt_np = to_numpy_img(x_gt)
    t_np = to_numpy_img(target)
    span = max(float(np.linalg.norm((gt_np - t_np).ravel())), 1e-3 * math.sqrt(gt_np.size))

    def dist(x: torch.Tensor) -> float:
        return float(np.linalg.norm((to_numpy_img(x) - t_np).ravel())) / span

    adv_np = to_numpy_img(adv_pred)
    row = {
        "tgt_span": span,
        "tgt_dist_clean": dist(clean_pred),
        "tgt_dist_adv": dist(adv_pred),
        "tgt_dist_init_clean": dist(clean_init),
        "tgt_dist_init_adv": dist(adv_init),
        "tgt_success": float(np.linalg.norm((adv_np - t_np).ravel())
                             < np.linalg.norm((adv_np - gt_np).ravel())),
    }
    row["tgt_closed"] = 1.0 - row["tgt_dist_adv"] / max(row["tgt_dist_clean"], 1e-12)
    if radon is not None:
        for cond, pred in (("clean", clean_pred), ("adv", adv_pred)):
            d_ran, d_nul = decompose_error(pred - target, radon)
            row[f"tgt_dist_{cond}_ran"] = float(torch.linalg.norm(d_ran.reshape(-1))) / span
            row[f"tgt_dist_{cond}_nul"] = float(torch.linalg.norm(d_nul.reshape(-1))) / span
    return row


def evaluate_batch(
    x_gt: torch.Tensor,
    clean_init: torch.Tensor,
    clean_y: torch.Tensor,
    clean_pred: torch.Tensor,
    adv_init: torch.Tensor,
    adv_y: torch.Tensor,
    adv_pred: torch.Tensor,
    delta: torch.Tensor,
    success_mse_factor: float,
    radon=None,
    target: Optional[torch.Tensor] = None,
) -> List[Dict[str, float]]:
    """Per-sample metrics for one batch: clean and adversarial reconstruction
    quality, the size of the perturbation, and the range/null decomposition of
    both error fields (when a radon operator is supplied). For a targeted attack
    ``target`` is its target image per sample, and the distance to it is scored
    as well (target_distance_metrics)."""
    rows: List[Dict[str, float]] = []
    batch_size = x_gt.shape[0]

    gt_adv = x_gt
    clean_mse_batch = per_example_mse(clean_pred, x_gt)
    adv_mse_batch = per_example_mse(adv_pred, gt_adv)
    delta_l2_batch = l2_norm_batch(delta)
    delta_linf_batch = delta.reshape(delta.shape[0], -1).abs().max(dim=1).values
    sino_shift_batch = delta.abs().reshape(delta.shape[0], -1).mean(dim=1)

    for i in range(batch_size):
        gt_np = to_numpy_img(x_gt[i])
        gt_adv_np = to_numpy_img(gt_adv[i])
        clean_pred_np = to_numpy_img(clean_pred[i])
        adv_pred_np = to_numpy_img(adv_pred[i])
        clean_init_np = to_numpy_img(clean_init[i])
        adv_init_np = to_numpy_img(adv_init[i])
        clean_y_np = to_numpy_img(clean_y[i])
        adv_y_np = to_numpy_img(adv_y[i])

        clean_rel_l2 = rel_l2_np(clean_pred_np, gt_np)
        adv_rel_l2 = rel_l2_np(adv_pred_np, gt_adv_np)
        init_shift = rel_l2_np(adv_init_np, clean_init_np)
        pred_shift = rel_l2_np(adv_pred_np, clean_pred_np)

        # Image-comparison metrics for the *initialisation* reconstruction
        # (the FBP/pinv output, i.e. the network input before the NSN).
        # These quantify how much the attack already corrupts the recon that
        # is fed into the network, separately from the final prediction.
        clean_init_rel_l2 = rel_l2_np(clean_init_np, gt_np)
        adv_init_rel_l2 = rel_l2_np(adv_init_np, gt_adv_np)

        clean_mse = float(clean_mse_batch[i].item())
        adv_mse = float(adv_mse_batch[i].item())
        clean_sino_l2 = float(np.linalg.norm(clean_y_np.reshape(-1)))
        delta_l2_i = float(delta_l2_batch[i].item())
        gt_l2_i = float(np.linalg.norm(gt_np.ravel()))

        row: Dict[str, float] = {
            "gt_norm": gt_l2_i,
            "clean_mse": clean_mse,
            "adv_mse": adv_mse,
            "mse_ratio": adv_mse / max(clean_mse, 1e-12),
            "clean_rel_l2": clean_rel_l2,
            "adv_rel_l2": adv_rel_l2,
            "rel_l2_ratio": adv_rel_l2 / max(clean_rel_l2, 1e-12),
            "clean_psnr": psnr(clean_pred_np, gt_np),
            "adv_psnr": psnr(adv_pred_np, gt_adv_np),
            "clean_ssim": ssim(clean_pred_np, gt_np),
            "adv_ssim": ssim(adv_pred_np, gt_adv_np),
            "clean_mae": mae(clean_pred_np, gt_np),
            "adv_mae": mae(adv_pred_np, gt_adv_np),
            "clean_nrmse": nrmse(clean_pred_np, gt_np),
            "adv_nrmse": nrmse(adv_pred_np, gt_adv_np),
            "clean_rmse": rmse(clean_pred_np, gt_np),
            "adv_rmse": rmse(adv_pred_np, gt_adv_np),
            "clean_max_err": max_abs_err(clean_pred_np, gt_np),
            "adv_max_err": max_abs_err(adv_pred_np, gt_adv_np),
            # Init-reconstruction metrics (network input, before the NSN)
            "clean_init_rel_l2": clean_init_rel_l2,
            "adv_init_rel_l2": adv_init_rel_l2,
            "init_rel_l2_ratio": adv_init_rel_l2 / max(clean_init_rel_l2, 1e-12),
            "clean_init_psnr": psnr(clean_init_np, gt_np),
            "adv_init_psnr": psnr(adv_init_np, gt_adv_np),
            "clean_init_ssim": ssim(clean_init_np, gt_np),
            "adv_init_ssim": ssim(adv_init_np, gt_adv_np),
            "clean_init_mae": mae(clean_init_np, gt_np),
            "adv_init_mae": mae(adv_init_np, gt_adv_np),
            "pred_shift_rel_l2": pred_shift,
            "init_shift_rel_l2": init_shift,
            "delta_l2": delta_l2_i,
            "delta_linf": float(delta_linf_batch[i].item()),
            "delta_mean_abs": float(sino_shift_batch[i].item()),
            "delta_rel_l2": delta_l2_i / max(clean_sino_l2, 1e-12),
            "clean_sino_l2": clean_sino_l2,
            "adv_sino_l2": float(np.linalg.norm(adv_y_np.reshape(-1))),
            "success_mse": float(adv_mse >= success_mse_factor * max(clean_mse, 1e-12)),
        }

        if radon is not None:
            e_ran_c, e_nul_c = decompose_error(clean_pred[i: i + 1] - x_gt[i: i + 1], radon)
            e_ran_a, e_nul_a = decompose_error(adv_pred[i: i + 1] - gt_adv[i: i + 1], radon)
            clean_e_l2 = max(float(np.linalg.norm((clean_pred_np - gt_np).ravel())), 1e-12)
            adv_e_l2 = max(float(np.linalg.norm((adv_pred_np - gt_adv_np).ravel())), 1e-12)
            clean_e_ran_l2 = float(np.linalg.norm(e_ran_c.numpy().ravel()))
            clean_e_nul_l2 = float(np.linalg.norm(e_nul_c.numpy().ravel()))
            adv_e_ran_l2 = float(np.linalg.norm(e_ran_a.numpy().ravel()))
            adv_e_nul_l2 = float(np.linalg.norm(e_nul_a.numpy().ravel()))
            row.update({
                "clean_e_ran_l2": clean_e_ran_l2,
                "clean_e_nul_l2": clean_e_nul_l2,
                "clean_e_ran_frac": clean_e_ran_l2 / max(clean_e_l2, 1e-12),
                "clean_e_nul_frac": clean_e_nul_l2 / max(clean_e_l2, 1e-12),
                "adv_e_ran_l2": adv_e_ran_l2,
                "adv_e_nul_l2": adv_e_nul_l2,
                "adv_e_ran_frac": adv_e_ran_l2 / max(adv_e_l2, 1e-12),
                "adv_e_nul_frac": adv_e_nul_l2 / max(adv_e_l2, 1e-12),
            })

            #   ||proj_ran(A_la x_hat) - y|| / ||y||   on the measured angles.
            # Damaging adversarial reconstructions stay measurement-consistent, i.e. the
            # error lives in the small-singular-value / null subspace the data
            # cannot constrain. A data-consistent model (NSN) should keep this
            # near zero; an unconstrained ResNet need not.
            # forward_la is the measurement operator; proj_ran keeps only the LA
            # rows so the result is identical to using the full-angle forward, but
            # this makes the measured-angle intent explicit.
            with torch.no_grad():
                # data-consistency residual  =  ||P_ran(A_la x̂) - y|| / ||y||
                def _consistency(pred_t, y_t):
                    y_hat = radon.proj_ran(radon.forward_la(pred_t))
                    num = float(torch.linalg.norm((y_hat - y_t).reshape(-1)).item())
                    den = float(torch.linalg.norm(y_t.reshape(-1)).item())
                    return num / max(den, 1e-12)
                clean_consistency = _consistency(clean_pred[i: i + 1], clean_y[i: i + 1])
                adv_consistency = _consistency(adv_pred[i: i + 1], adv_y[i: i + 1])
                # ...and the same residual against the *clean* measurement. The
                # line above scores the attacked reconstruction against the
                # attacked sinogram, which a hard-data-consistent model (NSN)
                # drives to ~0 by construction whatever the attack does — it is a
                # tautology, not a robustness result (every NSN row of job 20585
                # reads ~3e-8). Measured against the true y it is not: it says how
                # far the attack pushed the reconstruction off the *real* data,
                # which is the quantity TODOs.txt item 4 is actually asking about.
                adv_consistency_vs_clean = _consistency(adv_pred[i: i + 1], clean_y[i: i + 1])
            row.update({
                "clean_consistency_rel": clean_consistency,
                "adv_consistency_rel": adv_consistency,
                "adv_consistency_vs_clean_rel": adv_consistency_vs_clean,
            })
            # Per-metric range/null decomposition. ‖e‖² = ‖e_ran‖² + ‖e_nul‖², but
            # SSIM/PSNR/MAE/… are non-additive and cannot be split from the L2 norms
            # above. Instead we rebuild the reconstruction that carries *only* the
            # range (x_gt + e_ran) resp. null (x_gt + e_nul) component of the error and
            # score it with the same image metrics as the full prediction. This shows
            # how much each error subspace degrades each metric on its own — e.g. how
            # much of the SSIM/PSNR drop is structural (null) vs data-consistent (range).
            for cond, ref_np, e_ran_t, e_nul_t in (("clean", gt_np, e_ran_c, e_nul_c),
                                                   ("adv", gt_adv_np, e_ran_a, e_nul_a)):
                for sub, e_t in (("ran", e_ran_t), ("nul", e_nul_t)):
                    part = ref_np + e_t.numpy().reshape(ref_np.shape)
                    row.update({
                        f"{cond}_rel_l2_{sub}": rel_l2_np(part, ref_np),
                        f"{cond}_psnr_{sub}": psnr(part, ref_np),
                        f"{cond}_ssim_{sub}": ssim(part, ref_np),
                        f"{cond}_mae_{sub}": mae(part, ref_np),
                        f"{cond}_nrmse_{sub}": nrmse(part, ref_np),
                        f"{cond}_rmse_{sub}": rmse(part, ref_np),
                        f"{cond}_max_err_{sub}": max_abs_err(part, ref_np),
                    })

            # Decompose the *init-reconstruction* error too, so we can see how the
            # attack distributes range vs null energy in the network input,
            # before the NSN is applied.
            e_ran_ic, e_nul_ic = decompose_error(clean_init[i: i + 1] - x_gt[i: i + 1], radon)
            e_ran_ia, e_nul_ia = decompose_error(adv_init[i: i + 1] - gt_adv[i: i + 1], radon)
            clean_ie_l2 = max(float(np.linalg.norm((clean_init_np - gt_np).ravel())), 1e-12)
            adv_ie_l2 = max(float(np.linalg.norm((adv_init_np - gt_adv_np).ravel())), 1e-12)
            clean_ie_ran_l2 = float(np.linalg.norm(e_ran_ic.numpy().ravel()))
            clean_ie_nul_l2 = float(np.linalg.norm(e_nul_ic.numpy().ravel()))
            adv_ie_ran_l2 = float(np.linalg.norm(e_ran_ia.numpy().ravel()))
            adv_ie_nul_l2 = float(np.linalg.norm(e_nul_ia.numpy().ravel()))
            row.update({
                "clean_init_e_ran_l2": clean_ie_ran_l2,
                "clean_init_e_nul_l2": clean_ie_nul_l2,
                "clean_init_e_ran_frac": clean_ie_ran_l2 / max(clean_ie_l2, 1e-12),
                "clean_init_e_nul_frac": clean_ie_nul_l2 / max(clean_ie_l2, 1e-12),
                "adv_init_e_ran_l2": adv_ie_ran_l2,
                "adv_init_e_nul_l2": adv_ie_nul_l2,
                "adv_init_e_ran_frac": adv_ie_ran_l2 / max(adv_ie_l2, 1e-12),
                "adv_init_e_nul_frac": adv_ie_nul_l2 / max(adv_ie_l2, 1e-12),
            })

        if target is not None:
            s = slice(i, i + 1)
            row.update(target_distance_metrics(
                x_gt[s], target[s], clean_pred[s], adv_pred[s],
                clean_init[s], adv_init[s], radon=radon))

        rows.append(row)

    return rows

def _image_metrics(pred_np: np.ndarray, ref_np: np.ndarray) -> Dict[str, float]:
    """rel-L2 / PSNR / SSIM of one image against a reference (single sample)."""
    return {"rel_l2": rel_l2_np(pred_np, ref_np), "psnr": psnr(pred_np, ref_np),
            "ssim": ssim(pred_np, ref_np)}

def _component_metrics(gt_np: np.ndarray, e_component_np: np.ndarray) -> Dict[str, float]:
    """Score the component-only reconstruction (gt + e_component) against gt, so
    the SSIM/PSNR of the range- or null-space error can be read on its own. Same
    construction as the ``*_{ran,nul}`` columns in per_sample_metrics.csv."""
    return _image_metrics(gt_np + e_component_np.reshape(gt_np.shape), gt_np)

def summarize_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {"num_examples": len(rows)}
    if not rows:
        return metrics

    keys = [
        "gt_norm",
        "clean_mse",
        "adv_mse",
        "mse_ratio",
        "clean_rel_l2",
        "adv_rel_l2",
        "rel_l2_ratio",
        "clean_psnr",
        "adv_psnr",
        "clean_ssim",
        "adv_ssim",
        "clean_mae",
        "adv_mae",
        "clean_nrmse",
        "adv_nrmse",
        "clean_rmse",
        "adv_rmse",
        "clean_max_err",
        "adv_max_err",
        "clean_init_rel_l2",
        "adv_init_rel_l2",
        "init_rel_l2_ratio",
        "clean_init_psnr",
        "adv_init_psnr",
        "clean_init_ssim",
        "adv_init_ssim",
        "clean_init_mae",
        "adv_init_mae",
        "pred_shift_rel_l2",
        "init_shift_rel_l2",
        "delta_l2",
        "delta_linf",
        "delta_mean_abs",
        "delta_rel_l2",
        "success_mse",
    ]

    decomp_keys = [
        "clean_e_ran_l2", "clean_e_nul_l2", "clean_e_ran_frac", "clean_e_nul_frac",
        "adv_e_ran_l2", "adv_e_nul_l2", "adv_e_ran_frac", "adv_e_nul_frac",
        "clean_init_e_ran_l2", "clean_init_e_nul_l2",
        "clean_init_e_ran_frac", "clean_init_e_nul_frac",
        "adv_init_e_ran_l2", "adv_init_e_nul_l2",
        "adv_init_e_ran_frac", "adv_init_e_nul_frac",
        "clean_consistency_rel", "adv_consistency_rel", "adv_consistency_vs_clean_rel",
    ]
    # Per-metric range/null decomposition emitted by evaluate_batch: clean/adv ×
    # range/null × {rel_l2,psnr,ssim,mae,nrmse,max_err}. Aggregated like everything
    # else; absent (and silently skipped) when the attack runs without a radon op.
    decomp_keys += [
        f"{cond}_{metric}_{sub}"
        for cond in ("clean", "adv")
        for metric in ("rel_l2", "psnr", "ssim", "mae", "nrmse", "rmse", "max_err")
        for sub in ("ran", "nul")
    ]
    # Distance to the target, present for the targeted attacks only.
    decomp_keys += [
        "tgt_span", "tgt_dist_clean", "tgt_dist_adv", "tgt_dist_init_clean",
        "tgt_dist_init_adv", "tgt_closed", "tgt_success",
        "tgt_dist_clean_ran", "tgt_dist_clean_nul", "tgt_dist_adv_ran", "tgt_dist_adv_nul",
    ]
    keys = keys + [k for k in decomp_keys if k in rows[0]]
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean, half_width = confidence_interval_95(values)
        metrics[f"{key}_mean"] = mean
        metrics[f"{key}_ci95"] = half_width
        metrics[f"{key}_median"] = float(np.median(values))
        metrics[f"{key}_q25"] = float(np.percentile(values, 25))
        metrics[f"{key}_q75"] = float(np.percentile(values, 75))

    return metrics

# Subspace the local gain is measured in. "null" restricts both the input and
# the output of the linearised correction to N(A_la) — the channel both
# architectures are free to act in, and so the only one in which the number
# compares the *learned* maps rather than the architectures. "range" restricts
# them to the orthogonal complement N(A_la)^perp, where the NSN's correction is
# identically zero by construction. "full" leaves the correction unrestricted,
# which is the plain local Lipschitz constant of the learned correction.
# "cross" takes the input from N(A_la)^perp and the output in N(A_la): the gain
# from the measured part of the input into the unmeasured part of the output,
# which is the one a null-space attack exploits, since x_init = A^+ y always
# lies in N(A_la)^perp.
LIPSCHITZ_RESTRICTIONS = ("null", "range", "full", "cross")


def image_projector(radon, restriction: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Image-space projector for one of the symmetric restrictions (null, range,
    full), which confine input and output to the same subspace.

    The range projector is built as I - P_null rather than from the SVD factors
    directly, so the two are exactly complementary (P_ran + P_null = I to
    floating point) and the two restricted estimates cannot silently disagree
    about where the split lies.
    """
    if restriction == "null":
        return radon.proj_null_image
    if restriction == "range":
        return lambda v: v - radon.proj_null_image(v)
    if restriction == "full":
        return lambda v: v
    raise ValueError(f"unknown symmetric Lipschitz restriction {restriction!r}, "
                     f"expected one of ('null', 'range', 'full')")


def restriction_projectors(radon, restriction: str) -> Tuple[
        Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """(input projector, output projector) for any of ``LIPSCHITZ_RESTRICTIONS``."""
    if restriction == "cross":
        return image_projector(radon, "range"), image_projector(radon, "null")
    if restriction not in LIPSCHITZ_RESTRICTIONS:
        raise ValueError(f"unknown Lipschitz restriction {restriction!r}, "
                         f"expected one of {LIPSCHITZ_RESTRICTIONS}")
    proj = image_projector(radon, restriction)
    return proj, proj


def parse_lipschitz_restrictions(spec: str) -> List[str]:
    """Parse a comma-separated ``--lipschitz-restrictions`` spec, preserving the
    order of ``LIPSCHITZ_RESTRICTIONS`` and dropping duplicates."""
    wanted = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [token for token in wanted if token not in LIPSCHITZ_RESTRICTIONS]
    if unknown:
        raise ValueError(f"unknown Lipschitz restriction(s) {unknown}, "
                         f"expected any of {list(LIPSCHITZ_RESTRICTIONS)}")
    return [r for r in LIPSCHITZ_RESTRICTIONS if r in wanted]


def estimate_lipschitz(
    model: nn.Module,
    clean_cache: List[Tuple],
    radon,
    n_samples: int,
    n_iters: int,
    proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    out_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Operator-norm (local Lipschitz) estimate of the *learned correction*,
    restricted to whichever subspace ``proj`` projects onto.

    Linearise the correction  g(x) = f(x) - x  (= P_null(UNet(x)) for the NSN,
    UNet(x) for the ResNet) around the clean init x0, restrict the input with
    P_in = ``proj`` and the output with P_out = ``out_proj`` (default: the same
    projector), and estimate the largest singular value of
    M = P_out . J_g . P_in  by power iteration:

        d <- P_in d / ||.|| ;   repeat:  u = M d ,  d = M^T u / ||.|| ;   sigma ~ ||M d||.

    Attack-independent: it measures how strongly an input perturbation in that
    subspace can be amplified into output error in the same subspace, which is
    what governs worst-case robustness of the learned channel there.

    ``proj`` defaults to ``radon.proj_null_image``, the null-restricted gain.
    That is the architecture-comparable one: with the identity projector
    (``image_projector(radon, "full")``) the estimate is dominated by the fact
    that the NSN's correction is zero on N(A_la)^perp while the ResNet's is not,
    so it mostly reports the architectural constraint rather than what either
    network has learned; with the range projector it reports that constraint
    alone, and is zero for the NSN by construction.

    ``clean_cache`` entries only need to supply (x_gt, x_init, ...) as their
    first two elements. ``n_samples`` clean reconstructions are linearised and
    ``n_iters`` power iterations are run at each. The per-point estimates are
    returned under ``values`` as well, in the order of the cache.
    """
    proj = radon.proj_null_image if proj is None else proj
    out_proj = proj if out_proj is None else out_proj
    samples: List[float] = []

    for entry in clean_cache:
        x_init = entry[1]
        for b in range(x_init.shape[0]):
            if len(samples) >= n_samples:
                break
            x0 = x_init[b: b + 1].detach()

            def G(x: torch.Tensor) -> torch.Tensor:
                # learned correction, output restricted to the chosen subspace
                return out_proj(model(x) - x)
            d = proj(torch.randn_like(x0))
            d = d / (torch.linalg.norm(d.reshape(-1)) + 1e-12)
            for _ in range(n_iters):
                _, u = torch.autograd.functional.jvp(G, x0, d, strict=False)
                _, w = torch.autograd.functional.vjp(G, x0, out_proj(u), strict=False)
                w = proj(w)
                nw = torch.linalg.norm(w.reshape(-1))
                if nw < 1e-12:
                    break
                d = w / nw
            _, u = torch.autograd.functional.jvp(G, x0, d, strict=False)
            samples.append(float(torch.linalg.norm(out_proj(u).reshape(-1)).item()))
        if len(samples) >= n_samples:
            break

    if not samples:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan"), "n": 0,
                "values": []}
    return {
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "std": float(np.std(samples)),
        "n": len(samples),
        "values": samples,
    }


# --------------------------------------------------------------------------- #
# Attack suite: one command over a model directory + data directory.
# Produces attacks_n<noise>/init_<init>/<attack>/ for five PGD attacks — total
# error, null-space, range (null-complement), and two targeted attacks (toward
# the zero image / all-zero sinogram, and toward a different sample's ground
# truth) — with per-model metrics and example/attack-output arrays, cross-model
# perturbation-transfer stacks and an optional Lipschitz
# estimate. Every artifact is consumed by visualise.py.
# --------------------------------------------------------------------------- #

# attack dir name -> PGD objective.
# Two of the attacks are *targeted*: they steer the reconstruction toward a
# fixed reference image rather than merely inflating the error.
#   adversarial_target_zero   -> objective 'zero'   : target = all-zero sinogram,
#                                i.e. the zero image recon(0) = 0.
#   adversarial_target_sample -> objective 'target' : target = a *different*
#                                sample's ground truth (a random other item in
#                                the batch), supplied per batch at attack time.
_SUITE_OBJECTIVE = {
    "adversarial": "mse",                 # total reconstruction error
    "adversarial_null": "null",           # null-space (structural / learned) error
    "adversarial_range": "range",         # null-space complement = range (measured) error
    "adversarial_target_zero": "zero",    # targeted: drive the prediction to 0 (zero sinogram)
    "adversarial_target_sample": "target",  # targeted: drive toward another sample's GT
}
_SUITE_ATTACKS = ["adversarial", "adversarial_null", "adversarial_range",
                  "adversarial_target_zero", "adversarial_target_sample"]

# Suite attacks that need a per-batch target image threaded into the attack.
_SUITE_TARGETED_ATTACKS = {"adversarial_target_sample"}


def make_other_sample_target(x_gt: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Targeted-attack reference: for each item in the batch pick a *different*
    sample's ground truth. Returns a tensor shaped like ``x_gt`` whose i-th image
    is x_gt[j] for some random j != i (a derangement of the batch indices).

    With a batch of one there is no other sample, so the single image is returned
    unchanged (the targeted loss then degenerates to pushing pred toward its own
    GT — harmless, and this case is avoided by the default batch size)."""
    b = x_gt.shape[0]
    if b == 1:
        return x_gt.clone()
    # Drawn on the CPU, so a CPU generator serves a batch on any device.
    arange = torch.arange(b)
    # Draw a random derangement (a permutation with no fixed point) by rejection
    # sampling so every target is guaranteed to be a *different* sample. A random
    # permutation has no fixed point with probability ~1/e ≈ 0.37, so 20 attempts
    # miss only with negligible probability; the cyclic-shift fallback is a
    # guaranteed derangement for that rare case.
    perm = None
    for _ in range(20):
        cand = torch.randperm(b, generator=generator)
        if not bool((cand == arange).any()):
            perm = cand
            break
    if perm is None:
        shift = int(torch.randint(1, b, (1,), generator=generator).item())
        perm = (arange + shift) % b
    return x_gt[perm.to(x_gt.device)]


# Attacks scored by the distance to their target: the image the attack steers
# the reconstruction towards.
_SUITE_SCORED_BY_TARGET = {"adversarial_target_zero", "adversarial_target_sample"}


def suite_targets(attack_name: str, input_cache: List[Tuple]) -> List[Optional[torch.Tensor]]:
    """The target image of every batch of the input cache for one attack, None
    for an untargeted one.

    Drawn once per attack from a generator of its own, so both models are
    attacked towards the same targets and their results pair up sample by
    sample."""
    if attack_name == "adversarial_target_zero":
        return [torch.zeros_like(entry[0]) for entry in input_cache]
    if attack_name in _SUITE_TARGETED_ATTACKS:
        gen = torch.Generator().manual_seed(stage_seed("targets", attack_name))
        return [make_other_sample_target(entry[0], generator=gen) for entry in input_cache]
    return [None] * len(input_cache)

def detect_suite_models(model_dir: Optional[str]) -> List[str]:
    """Return the model names whose checkpoints exist under ``model_dir``."""
    base = Path(model_dir) if model_dir else Path(".")
    ckpt_dir = base / f"init_{INIT_NAME}" / "checkpoints"
    return [m for m in ("resnet", "nsn") if (ckpt_dir / f"{m}_best.pt").exists()]

@dataclass
class RunSetup:
    """What every entry point resolves identically from ``--data-root``.

    The attack suite and the epoch study each repeated the same nine lines of
    device/seed/summary/radon resolution. Keeping it in one place means a change
    to how the radon operator is built (dtype, dense vs sparse) cannot silently
    apply to only one of them."""
    device: torch.device
    summary: Dict
    radon: object
    noise_rel: float
    out_root: Path

def prepare_run(args) -> RunSetup:
    """Resolve the data set, operator and output root shared by all run modes."""
    if not args.data_root:
        raise ValueError("requires --data-root (used to infer dataset type and init methods).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = load_summary(args.data_root)
    noise_rel = float(summary.get("noise_sigma_rel") or 0.0)
    if not (Path(args.data_root) / INIT_NAME).is_dir():
        raise FileNotFoundError(
            f"No '{INIT_NAME}' init-reconstruction folder found in {args.data_root}.")
    return RunSetup(
        device=device,
        summary=summary,
        radon=build_radon(summary, device=device, dtype=torch.float64 if F64 else torch.float32, dense=not SPARSE),
        noise_rel=noise_rel,
        out_root=Path(args.out_dir or f"attacks_n{noise_rel}"),
    )

def build_init_inputs(args, radon, max_samples: int, device):
    """Range projector and the shared input cache — identical in both run modes,
    so it lives once.

    Returns (projector, input_cache). The cache is what makes every model see
    byte-identical inputs, which is the basis for comparing them."""
    loader = get_ellipse_dataloader(
        batch_size=BATCH_SIZE,
        split=SPLIT, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
        shuffle=False, num_workers=NUM_WORKERS, data_root=args.data_root,
    )
    proj = lambda y: radon.proj_ran(y)
    return proj, build_input_cache(proj, loader, max_samples, device)

def build_input_cache(projector, loader, max_samples: int, device) -> List[Tuple]:
    """Cache the model-independent (x_gt, x_init, y_clean) inputs once so every
    model in the suite is attacked and evaluated on identical data."""
    cache: List[Tuple] = []
    n = 0
    with torch.no_grad():
        for x_gt, x_init, y_delta in loader:
            x_gt = to_4d(x_gt).to(device)
            x_init = to_4d(x_init).to(device)
            y_delta = to_4d(y_delta).to(device)
            y_clean = projector(y_delta)
            cache.append((x_gt, x_init, y_clean))
            n += x_gt.shape[0]
            if n >= max_samples:
                break
    return cache

def build_example_row(radon, x_gt, clean_init, adv_init, clean_pred, adv_pred,
                      y_clean, adv_y, delta, i: int,
                      target: Optional[torch.Tensor] = None) -> Dict:
    """Assemble one example-image row (GT, inits, preds, sinos and range/null
    error decompositions) for the saved examples bundle (rendered later by
    visualise.save_examples). For a targeted attack the row also carries the
    target image and the distances to it."""
    e_ran_clean, e_nul_clean = decompose_error(clean_pred[i:i + 1] - x_gt[i:i + 1], radon)
    e_ran_adv, e_nul_adv = decompose_error(adv_pred[i:i + 1] - x_gt[i:i + 1], radon)
    e_ran_ic, e_nul_ic = decompose_error(clean_init[i:i + 1] - x_gt[i:i + 1], radon)
    e_ran_ia, e_nul_ia = decompose_error(adv_init[i:i + 1] - x_gt[i:i + 1], radon)
    gt_np = to_numpy_img(x_gt[i])
    cp_np = to_numpy_img(clean_pred[i])
    ap_np = to_numpy_img(adv_pred[i])
    ci_np = to_numpy_img(clean_init[i])
    ai_np = to_numpy_img(adv_init[i])
    row = {
        "x_gt": gt_np,
        "clean_init": ci_np,
        "adv_init": ai_np,
        "clean_pred": cp_np,
        "adv_pred": ap_np,
        "clean_y": to_numpy_img(y_clean[i]),
        "adv_y": to_numpy_img(adv_y[i]),
        "delta": to_numpy_img(delta[i]),
        # Per-sample metrics for *this* example, so the figure shows the error
        # of the exact sample being plotted (not an aggregate). rel-L2 / PSNR /
        # SSIM for the prediction and the init reconstruction, clean vs adv.
        "m_clean_pred": _image_metrics(cp_np, gt_np),
        "m_adv_pred": _image_metrics(ap_np, gt_np),
        "m_clean_init": _image_metrics(ci_np, gt_np),
        "m_adv_init": _image_metrics(ai_np, gt_np),
        "e_ran_clean": e_ran_clean.squeeze().numpy(),
        "e_nul_clean": e_nul_clean.squeeze().numpy(),
        "e_ran_adv": e_ran_adv.squeeze().numpy(),
        "e_nul_adv": e_nul_adv.squeeze().numpy(),
        "e_ran_init_clean": e_ran_ic.squeeze().numpy(),
        "e_nul_init_clean": e_nul_ic.squeeze().numpy(),
        "e_ran_init_adv": e_ran_ia.squeeze().numpy(),
        "e_nul_init_adv": e_nul_ia.squeeze().numpy(),
    }
    # Per-panel metrics for the range/null decomposition figures: SSIM/PSNR/rel-L2
    # of the component-only reconstruction (gt + e_ran resp. gt + e_nul).
    row["m_ran_clean"] = _component_metrics(gt_np, row["e_ran_clean"])
    row["m_nul_clean"] = _component_metrics(gt_np, row["e_nul_clean"])
    row["m_ran_adv"] = _component_metrics(gt_np, row["e_ran_adv"])
    row["m_nul_adv"] = _component_metrics(gt_np, row["e_nul_adv"])
    row["m_ran_init_clean"] = _component_metrics(gt_np, row["e_ran_init_clean"])
    row["m_nul_init_clean"] = _component_metrics(gt_np, row["e_nul_init_clean"])
    row["m_ran_init_adv"] = _component_metrics(gt_np, row["e_ran_init_adv"])
    row["m_nul_init_adv"] = _component_metrics(gt_np, row["e_nul_init_adv"])
    # Reference for the NSN range-shift identity: Delta e_ran should equal
    # proj_ran(A_la^+ delta). A_la^+ is linear, so the identity holds exactly.
    e_ran_init_d, _ = decompose_error(radon.backward_la(delta[i:i + 1]), radon)
    row["proj_ran_init_delta"] = e_ran_init_d.squeeze().numpy()
    if target is not None:
        s = slice(i, i + 1)
        row["target"] = to_numpy_img(target[i])
        d = target_distance_metrics(x_gt[s], target[s], clean_pred[s], adv_pred[s],
                                    clean_init[s], adv_init[s])
        row["tgt_dist_clean"], row["tgt_dist_adv"] = d["tgt_dist_clean"], d["tgt_dist_adv"]
    return row

def _stack_chunks(chunks: List[torch.Tensor]) -> np.ndarray:
    """Concatenate per-batch [B,1,H,W] tensor chunks and drop the channel axis,
    giving a single [N,H,W] numpy array for the .npz attack_output archive."""
    return torch.cat(chunks, dim=0)[:, 0].numpy()

def run_suite(args, radon, summary: Dict,
              noise_rel: float, eps_nominal: float,
              attacks_root: Path) -> bool:
    """Run the five-attack suite and write every artifact to disk. Returns False
    (and skips) when no model checkpoints exist.

    This function only *computes and saves*; it never plots. The figures are
    produced afterwards by ``visualise.py`` from the artifacts written here."""
    device = radon.device
    model_names = detect_suite_models(args.model_dir)
    if not model_names:
        print(f"[suite] no checkpoints found under '{args.model_dir}', skipping.")
        return False
    print(f"\n[suite] ===== models={model_names} =====")

    projector, input_cache = build_init_inputs(args, radon, args.max_samples, device)

    # Load every model + adapter once (reused for both attacking and transfer).
    models: Dict[str, nn.Module] = {}
    adapters: Dict[str, ModelAttackAdapter] = {}
    for name in model_names:
        m = load_model_checkpoint(model_name=name, radon=radon, device=device,
                                  model_dir=args.model_dir)
        models[name] = m
        adapters[name] = ModelAttackAdapter(model=m, radon=radon, projector=projector)

    out_root = attacks_root / f"init_{INIT_NAME}"
    out_root.mkdir(parents=True, exist_ok=True)

    if getattr(args, "lipschitz_only", False):
        write_lipschitz(args, models, input_cache, radon, out_root)
        print(f"[suite] Lipschitz only, attacks skipped -> {out_root}")
        return True

    for attack_name in _SUITE_ATTACKS:
        attack_dir = out_root / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        objective = _SUITE_OBJECTIVE[attack_name]
        print(f"\n[suite] === {attack_name}  (objective={objective}) ===")

        summary_by_model: Dict[str, Dict] = {}
        transfer_pert: Dict[str, torch.Tensor] = {}  # first-batch perturbation per source
        targets = suite_targets(attack_name, input_cache)

        for model_name in model_names:
            adapter = adapters[model_name]
            model = models[model_name]
            # The same seed for every model: both start from the same random
            # points, so the only difference between their attacks is the model.
            set_seed(stage_seed("suite", attack_name))
            rows: List[Dict[str, float]] = []
            example_rows: List[Dict] = []
            worst: List[Tuple[float, Dict]] = []
            delta_chunks: List[torch.Tensor] = []
            yadv_chunks: List[torch.Tensor] = []
            processed = 0

            for bi, (x_gt, clean_init, y_clean) in enumerate(input_cache):
                with torch.no_grad():
                    clean_pred = model(clean_init)
                eps_batch = suite_eps_batch(y_clean, eps_nominal)

                # Targeted attacks steer the recon toward a fixed reference:
                # 'zero' targets the zero image internally, while 'target' needs
                # the per-batch reference, a random *other* sample's ground
                # truth. Both are scored by the distance to their target.
                target = targets[bi]
                result = pgd_attack(
                    adapter=adapter, x_gt=x_gt, y_clean=y_clean,
                    clean_pred=clean_pred, eps=eps_batch,
                    alpha=suite_step_size(eps_batch),
                    objective=objective,
                    target=target if attack_name in _SUITE_TARGETED_ATTACKS else None)
                with torch.no_grad():
                    adv_pred, adv_init, y_adv = adapter.forward(result.y_adv)
                delta = result.delta

                if bi == 0:
                    transfer_pert[model_name] = delta.detach()

                delta_chunks.append(delta.detach().cpu())
                yadv_chunks.append(y_adv.detach().cpu())

                rows.extend(evaluate_batch(
                    x_gt=x_gt, clean_init=clean_init, clean_y=y_clean, clean_pred=clean_pred,
                    adv_init=adv_init, adv_y=y_adv, adv_pred=adv_pred, delta=delta,
                    success_mse_factor=SUCCESS_MSE_FACTOR, radon=radon,
                    target=target,
                ))

                # One example-image dict for sample j, shared by the first-K
                # saved examples and the worst-case capture so both use
                # identical panel content.
                def make_example_row(j):
                    return build_example_row(
                        radon, x_gt, clean_init, adv_init, clean_pred, adv_pred,
                        y_clean, y_adv, delta, j, target=target)

                slots = SUITE_EXAMPLES - len(example_rows)
                for j in range(min(x_gt.shape[0], max(slots, 0))):
                    example_rows.append(make_example_row(j))

                # Worst-case capture: keep the args.suite_worst samples the attack
                # degraded most (largest adversarial/clean rel-L2 ratio), so the
                # presentation shows where the attack does the most damage — not
                # just the first few samples. Rows are built on demand for
                # candidates that make the running top-K.
                if SUITE_WORST > 0:
                    batch_rows = rows[-x_gt.shape[0]:]
                    cur_min = min((w[0] for w in worst), default=float("-inf"))
                    for j in range(x_gt.shape[0]):
                        score = float(batch_rows[j].get(
                            "rel_l2_ratio", batch_rows[j].get("adv_rel_l2", 0.0)))
                        if len(worst) < SUITE_WORST or score > cur_min:
                            ex = make_example_row(j)
                            ex["worst_score"] = score
                            worst.append((score, ex))
                            worst.sort(key=lambda t: t[0], reverse=True)
                            del worst[SUITE_WORST:]
                            cur_min = worst[-1][0]

                processed += x_gt.shape[0]
                if processed >= args.max_samples:
                    break

            metrics = summarize_metrics(rows)
            metrics.update({"model_name": model_name, "attack_name": attack_name,
                            "objective": objective, "eps": eps_nominal})
            summary_by_model[model_name] = metrics

            model_out = attack_dir / model_name
            model_out.mkdir(parents=True, exist_ok=True)
            if rows:
                fieldnames = list(rows[0].keys())
                with open(model_out / "per_sample_metrics.csv", "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            # Example-image arrays for the qualitative figures (rebuilt by
            # visualise.save_examples).
            write_rows_bundle(model_out / "examples.npz",
                              model_out / "examples.json", example_rows)
            # Worst-case example bundle (samples the attack degraded most).
            if worst:
                worst_rows = [ex for _, ex in sorted(worst, key=lambda t: t[0], reverse=True)]
                write_rows_bundle(model_out / "worst.npz",
                                  model_out / "worst.json", worst_rows)
            # Full attack output: adversarial sinogram + perturbation per sample
            # (clean sinogram recoverable as y_adv - delta).
            if delta_chunks:
                np.savez_compressed(model_out / "attack_output.npz",
                                    delta=_stack_chunks(delta_chunks),
                                    y_adv=_stack_chunks(yadv_chunks))
            print(f"  [{model_name}] n={len(rows)} "
                  f"adv_rel_l2={metrics.get('adv_rel_l2_mean', float('nan')):.4f} "
                  f"adv_rmse={metrics.get('adv_rmse_mean', float('nan')):.4f} "
                  f"e_nul(med)={metrics.get('adv_e_nul_l2_median', float('nan')):.4f} "
                  f"e_ran(med)={metrics.get('adv_e_ran_l2_median', float('nan')):.4f}")

        with open(attack_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"attack": attack_name, "objective": objective, "eps": eps_nominal,
                       "noise_sigma_rel": noise_rel, "models": summary_by_model}, f, indent=2)

        # The aggregate figures (scatter / bars / consistency) and the per-attack
        # example figures are rendered later by visualise.render_tree, from the
        # per_sample_metrics.csv and examples bundle written above. Here we only
        # persist the cross-model transfer image stacks it cannot recompute
        # without the models.
        if not input_cache:
            continue
        x_gt0, clean_init0, y_clean0 = input_cache[0]
        B0 = x_gt0.shape[0]
        T = min(SUITE_TRANSFER_SAMPLES, B0)
        n_ex = min(SUITE_EXAMPLES, B0)

        # Reconstruction of every (source δ, target model) pair on the first batch;
        # store enough samples for both the transfer grid (T) and the per-example
        # cross-model figures (n_ex).
        K = max(T, n_ex)
        preds: Dict[Tuple[str, str], torch.Tensor] = {}
        clean_preds0: Dict[str, torch.Tensor] = {}
        for target in model_names:
            with torch.no_grad():
                clean_preds0[target] = models[target](clean_init0)
            for source, pert in transfer_pert.items():
                with torch.no_grad():
                    y_t = projector(y_clean0 + pert)
                    pred_t, _, _ = adapters[target].forward(y_t)
                preds[(source, target)] = pred_t

        gt_stack = np.stack([to_numpy_img(x_gt0[k]) for k in range(K)])
        recon: Dict[str, np.ndarray] = {}
        for target in model_names:
            recon[f"clean__{target}"] = np.stack(
                [to_numpy_img(clean_preds0[target][k]) for k in range(K)])
        for (source, target), pred_t in preds.items():
            recon[f"pred__{source}__{target}"] = np.stack(
                [to_numpy_img(pred_t[k]) for k in range(K)])
        write_transfer_bundle(
            attack_dir / "transfer.npz", attack_dir / "transfer.json",
            model_names=model_names, attack_name=attack_name,
            eps=eps_nominal, T=T, n_ex=n_ex, gt_stack=gt_stack, recon=recon)


    write_lipschitz(args, models, input_cache, radon, out_root)

    print(f"[suite] done -> {out_root}")
    return True


def write_lipschitz(args, models: Dict[str, nn.Module], input_cache: List[Tuple],
                    radon, out_root: Path) -> None:
    """One estimate per model per subspace, merged into ``out_root/lipschitz.json``.

    The null-restricted gain is the comparable one, range and full say how much
    of it is the architecture, and cross is the gain a null-space attack
    exploits. Restrictions not estimated in this call keep the values already in
    the file, so a --lipschitz-only run can add one restriction to a finished run."""
    path = out_root / "lipschitz.json"
    lip_res: Dict[str, Dict[str, Dict[str, float]]] = {}
    if path.exists():
        lip_res = json.loads(path.read_text(encoding="utf-8"))
        # the older flat schema {mean, ...} per model was the null-restricted gain
        lip_res = {m: ({"null": e} if "mean" in e else e) for m, e in lip_res.items()}
    for name, model in models.items():
        entry = lip_res.setdefault(name, {})
        for restriction in parse_lipschitz_restrictions(args.lipschitz_restrictions):
            in_proj, out_proj = restriction_projectors(radon, restriction)
            # Same start vectors for every model, independent of the attacks before.
            set_seed(stage_seed("lipschitz", restriction))
            r = estimate_lipschitz(
                model=model, clean_cache=input_cache, radon=radon,
                n_samples=args.lipschitz_samples, n_iters=args.lipschitz_iters,
                proj=in_proj, out_proj=out_proj)
            entry[restriction] = r
            print(f"[suite][lipschitz] {name} [{restriction}] mean={r['mean']:.4g} "
                  f"max={r['max']:.4g} (n={r['n']})")
    if lip_res:
        # Plotted later by visualise.render_tree from this json.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lip_res, f, indent=2)


# --------------------------------------------------------------------------- #
# Cross-attack / cross-model aggregation.
#
# Every (init, attack, model) run already writes its own per_sample_metrics.csv
# and a per-attack summary.json (with <metric>_mean / _median / _ci95 / _q25 /
# _q75 from summarize_metrics). To *compare the attacks consistently* we collect
# the same curated set of mean+median error metrics for every run into one flat
# table (aggregate_summary.csv) and a nested json (aggregate_summary.json).
#
# Like visualise.py, this is rebuilt purely from the on-disk summary.json
# artifacts, so it can be regenerated without re-running any attack.
# --------------------------------------------------------------------------- #

# Curated metrics reported for every attack so the comparison is apples-to-apples.
# Each base name is emitted as both <name>_mean and <name>_median (both are always
# present in a summarize_metrics() summary). Missing keys degrade to NaN so an
# attack run without a radon operator (no range/null decomposition) still tabulates.
_AGGREGATE_METRICS = [
    "clean_rel_l2", "adv_rel_l2", "rel_l2_ratio",
    "clean_psnr", "adv_psnr", "clean_ssim", "adv_ssim",
    "adv_rmse", "adv_mae", "mse_ratio",
    "adv_e_nul_l2", "adv_e_ran_l2", "adv_e_nul_frac",
    "clean_consistency_rel", "adv_consistency_rel", "adv_consistency_vs_clean_rel",
    "delta_rel_l2", "success_mse",
    # targeted attacks only; NaN for the others
    "tgt_dist_clean", "tgt_dist_adv", "tgt_closed", "tgt_success",
]

def aggregate_from_disk(attacks_root) -> List[Dict[str, float]]:
    """Collect a curated mean+median metric row per (init, attack, model) from the
    per-attack summary.json files written by the suite.

    Walks ``attacks_root/init_<init>/<attack>/summary.json`` (the same layout
    visualise.py consumes) and returns one flat record per model. Reconstructable
    from artifacts alone — no torch, models or radon operator required."""
    root = Path(attacks_root)
    records: List[Dict[str, float]] = []
    for init_dir in sorted(root.glob("init_*")):
        if not init_dir.is_dir():
            continue
        init_name = init_dir.name[len("init_"):]
        for summ_path in sorted(init_dir.glob("*/summary.json")):
            with open(summ_path, "r", encoding="utf-8") as f:
                summ = json.load(f)
            attack_name = summ.get("attack", summ_path.parent.name)
            objective = summ.get("objective")
            eps = summ.get("eps")
            for model_name, m in (summ.get("models") or {}).items():
                rec: Dict[str, float] = {
                    "init": init_name,
                    "attack": attack_name,
                    "objective": objective,
                    "model": model_name,
                    "n": m.get("num_examples"),
                    "eps": eps,
                }
                for key in _AGGREGATE_METRICS:
                    rec[f"{key}_mean"] = m.get(f"{key}_mean", float("nan"))
                    rec[f"{key}_median"] = m.get(f"{key}_median", float("nan"))
                records.append(rec)
    return records

def write_aggregate_summary(attacks_root) -> List[Dict[str, float]]:
    """Write aggregate_summary.csv and aggregate_summary.json under ``attacks_root``
    consolidating every attack/model run, and return the flat records.

    The CSV has one row per (init, attack, model) with the curated
    <metric>_mean/<metric>_median columns so different attacks can be scanned and
    compared side by side; the JSON nests the same records by init -> attack ->
    model for programmatic access. Returns [] (and writes nothing) when no
    per-attack summaries are found."""
    root = Path(attacks_root)
    records = aggregate_from_disk(root)
    if not records:
        return records

    meta_cols = ["init", "attack", "objective", "model", "n", "eps"]
    metric_cols = [f"{k}_{stat}" for k in _AGGREGATE_METRICS for stat in ("mean", "median")]
    fieldnames = meta_cols + metric_cols
    csv_path = root / "aggregate_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Stable ordering so the same run always produces byte-identical output.
        for rec in sorted(records, key=lambda r: (str(r["init"]), str(r["attack"]), str(r["model"]))):
            writer.writerow(rec)

    nested: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for rec in records:
        nested.setdefault(str(rec["init"]), {}).setdefault(str(rec["attack"]), {})[str(rec["model"])] = rec
    with open(root / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2)

    # Console digest: the headline mean/median adversarial rel-L2 per row, so a
    # suite run ends with a consistent at-a-glance comparison of the attacks.
    print(f"[suite] aggregate over {len(records)} (init,attack,model) runs -> {csv_path}")
    print(f"[suite] {'init':<6} {'attack':<24} {'model':<10} "
          f"{'adv_rel_l2(mean)':>16} {'(median)':>10} {'rel_l2_ratio(mean)':>19}")
    for rec in sorted(records, key=lambda r: (str(r["init"]), str(r["attack"]), str(r["model"]))):
        print(f"[suite] {str(rec['init']):<6} {str(rec['attack']):<24} {str(rec['model']):<10} "
              f"{rec.get('adv_rel_l2_mean', float('nan')):>16.4f} "
              f"{rec.get('adv_rel_l2_median', float('nan')):>10.4f} "
              f"{rec.get('rel_l2_ratio_mean', float('nan')):>19.4f}")
    return records

def detect_epoch_checkpoints(model_dir: Optional[str],
                            model_name: str) -> List[Tuple[int, Path]]:
    """Per-epoch checkpoints ``{model}_epoch{NNN}.pt`` written by train.py with
    --checkpoint-every N, returned as [(epoch, path), ...] sorted by epoch."""
    base = Path(model_dir) if model_dir else Path(".")
    d = base / f"init_{INIT_NAME}" / "checkpoints"
    out: List[Tuple[int, Path]] = []
    if d.is_dir():
        for pth in d.glob(f"{model_name}_epoch*.pt"):
            try:
                epoch = int(pth.stem.split("_epoch")[1])
            except (IndexError, ValueError):
                continue
            out.append((epoch, pth))
    return sorted(out)

def load_epoch_history(model_dir: Optional[str],
                       model_name: str) -> Tuple[Dict[int, Tuple[float, float]], Optional[int]]:
    """Read {model}_history.json into {epoch: (train_loss, val_loss)} plus the
    best epoch, so the epoch-attack study can overlay attackability on the loss
    curves. Returns ({}, None) when no history was written."""
    base = Path(model_dir) if model_dir else Path(".")
    hp = base / f"init_{INIT_NAME}" / "checkpoints" / f"{model_name}_history.json"
    if not hp.exists():
        return {}, None
    blob = json.loads(hp.read_text(encoding="utf-8"))
    hist = {int(h["epoch"]): (float(h.get("train", float("nan"))),
                              float(h.get("val", float("nan"))))
            for h in blob.get("history", [])}
    return hist, blob.get("best_epoch")

# Objectives the epoch study can attack with: those that need no per-batch
# target image.
_EPOCH_OBJECTIVES = {"mse", "null", "range", "zero"}


def resolve_epoch_eps(epoch_eps: Optional[float], noise_rel: float) -> float:
    """The budget of the epoch study: ``epoch_eps`` when given, otherwise the
    training noise level, the budget of the attack suite. With the suite's
    budget the last snapshot of each curve is attacked as the suite attacks the
    best checkpoint, at every noise level alike."""
    eps = epoch_eps if epoch_eps is not None else noise_rel
    if not eps or eps <= 0:
        raise ValueError("the epoch study needs noise_sigma_rel in summary.json, "
                         "or pass --epoch-eps explicitly.")
    return float(eps)


def epoch_study_csv_name(model_name: str, objective: str = "mse") -> str:
    """epoch_study/{init}_{model}.csv for the total-error attack, the name every
    finished run already has; {init}_{model}_{objective}.csv for any other."""
    suffix = "" if objective == "mse" else f"_{objective}"
    return f"{INIT_NAME}_{model_name}{suffix}.csv"


def run_epoch_study(args) -> None:
    """Attack every saved epoch of each model individually and tabulate the
    adversarial error vs epoch alongside the train/val loss.

    This isolates *when* attackability arises during training and whether it
    tracks overfitting (validation loss diverging from training loss). For each
    model it loads every {model}_epoch{NNN}.pt, runs one PGD attack
    (``--epoch-objective``, by default the total error) on the shared sample
    cache, and writes epoch_study/{epoch_study_csv_name} (rendered by
    visualise.save_epoch_study_plots). Requires train.py to have been run with
    --checkpoint-every N."""
    setup = prepare_run(args)
    device, radon = setup.device, setup.radon
    out_root = setup.out_root

    eps_nominal = resolve_epoch_eps(getattr(args, "epoch_eps", None), setup.noise_rel)
    objective = getattr(args, "epoch_objective", "mse")
    if objective not in _EPOCH_OBJECTIVES:
        raise ValueError(f"--epoch-objective {objective!r}, expected one of "
                         f"{sorted(_EPOCH_OBJECTIVES)}")

    study_dir = out_root / "epoch_study"
    study_dir.mkdir(parents=True, exist_ok=True)

    # Optional model subset, so one array task can own one model. Each task writes
    # its own epoch_study/{init}_{model}.csv, so they never collide.
    model_filter = ([m.strip() for m in args.models.split(",") if m.strip()]
                    if getattr(args, "models", None) else None)

    wrote_any = False
    projector, input_cache = build_init_inputs(args, radon, args.max_samples, device)

    found = detect_suite_models(args.model_dir)
    if model_filter is not None:
        unknown = [m for m in model_filter if m not in found]
        if unknown:
            raise ValueError(
                f"--models requested {unknown} but only {found} have checkpoints.")
        found = [m for m in found if m in model_filter]
    for model_name in found:
        ckpts = detect_epoch_checkpoints(args.model_dir, model_name)
        if not ckpts:
            print(f"[epoch-study] model '{model_name}': no per-epoch checkpoints "
                  f"(train with --checkpoint-every N), skipping.")
            continue
        hist, best_epoch = load_epoch_history(args.model_dir, model_name)
        print(f"\n[epoch-study] model '{model_name}': {len(ckpts)} epochs, "
              f"objective={objective}, eps={eps_nominal:g}")
        rows_out: List[Dict[str, float]] = []
        for epoch, ckpt_path in ckpts:
            model = build_models([model_name], radon=radon)[model_name].to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
            model.eval()
            adapter = ModelAttackAdapter(model=model, radon=radon, projector=projector)
            # Every snapshot, of either model, starts from the same random points,
            # so the weights are the only thing that varies along a curve.
            set_seed(stage_seed("epoch", objective))
            rows: List[Dict[str, float]] = []
            processed = 0
            for x_gt, clean_init, y_clean in input_cache:
                with torch.no_grad():
                    clean_pred = model(clean_init)
                eps_batch = suite_eps_batch(y_clean, eps_nominal)
                result = pgd_attack(
                    adapter=adapter, x_gt=x_gt, y_clean=y_clean,
                    clean_pred=clean_pred, eps=eps_batch,
                    alpha=suite_step_size(eps_batch),
                    objective=objective)
                with torch.no_grad():
                    adv_pred, adv_init, y_adv = adapter.forward(result.y_adv)
                rows.extend(evaluate_batch(
                    x_gt=x_gt, clean_init=clean_init, clean_y=y_clean, clean_pred=clean_pred,
                    adv_init=adv_init, adv_y=y_adv, adv_pred=adv_pred, delta=result.delta,
                    success_mse_factor=SUCCESS_MSE_FACTOR, radon=radon))
                processed += x_gt.shape[0]
                if processed >= args.max_samples:
                    break
            m = summarize_metrics(rows)
            tr, va = hist.get(epoch, (float("nan"), float("nan")))
            rows_out.append({
                "epoch": epoch, "train_loss": tr, "val_loss": va,
                "is_best": int(best_epoch is not None and epoch == best_epoch),
                "eps": eps_nominal,
                "clean_rel_l2_median": m.get("clean_rel_l2_median", float("nan")),
                "adv_rel_l2_mean": m.get("adv_rel_l2_mean", float("nan")),
                "adv_rel_l2_median": m.get("adv_rel_l2_median", float("nan")),
                "rel_l2_ratio_median": m.get("rel_l2_ratio_median", float("nan")),
                "adv_e_nul_frac_median": m.get("adv_e_nul_frac_median", float("nan")),
                "adv_consistency_rel_median": m.get("adv_consistency_rel_median", float("nan")),
                "adv_consistency_vs_clean_rel_median": m.get(
                    "adv_consistency_vs_clean_rel_median", float("nan")),
                # the null-space channel, which the null objective attacks
                "clean_rel_l2_nul_median": m.get("clean_rel_l2_nul_median", float("nan")),
                "adv_rel_l2_nul_median": m.get("adv_rel_l2_nul_median", float("nan")),
            })
            print(f"  epoch {epoch:03d}  val={va:.5f}  adv_rel_l2(med)="
                  f"{rows_out[-1]['adv_rel_l2_median']:.4f}  ratio(med)="
                  f"{rows_out[-1]['rel_l2_ratio_median']:.3f}")
        csv_path = study_dir / epoch_study_csv_name(model_name, objective)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        wrote_any = True
        print(f"[epoch-study] wrote {csv_path}")

    if not wrote_any:
        raise FileNotFoundError(
            "No per-epoch checkpoints found. Train with train.py --checkpoint-every N first.")
    print(f"\n[epoch-study] done -> {study_dir}")
    print(f"[epoch-study] render curves with:  python visualise.py {out_root}")

def run_attack_suite(args) -> None:
    if not args.data_root:
        raise ValueError("requires --data-root (it holds summary.json and the data).")
    setup = prepare_run(args)
    summary, radon = setup.summary, setup.radon
    noise_rel = setup.noise_rel
    attacks_root = setup.out_root

    eps_nominal = args.suite_eps if args.suite_eps is not None else noise_rel
    if eps_nominal <= 0:
        raise ValueError(
            "requires noise_sigma_rel in summary.json, or pass --suite-eps explicitly.")

    # eps is a relative L2 fraction: the per-sample budget is eps*||y_i||_2,
    # and the step size is derived from it per sample too.
    print(f"[suite] dataset={summary.get('dataset')}  "
          f"eps={eps_nominal:g}*||y||  alpha=2.5*eps/{SUITE_STEPS}")

    print(f"[suite] attacks ({len(_SUITE_ATTACKS)}): {', '.join(_SUITE_ATTACKS)}")
    if not run_suite(args, radon, summary, noise_rel, eps_nominal,
                     attacks_root):
        raise FileNotFoundError(
            f"No checkpoints found under model-dir '{args.model_dir}'."
        )
    
    write_aggregate_summary(attacks_root)

    print(f"\n[suite] done -> {attacks_root}")
    print(f"[suite] render figures with:  python visualise.py {attacks_root}")


def parse():
    parser = argparse.ArgumentParser(
        description="Adversarial attack suite for limited-angle Radon reconstruction models. "
                    "Runs five PGD attacks (total error, null-space, range, and two "
                    "targeted attacks) over every model checkpoint detected for each "
                    "init method. Writes .npz/.csv/.json artifacts only; render "
                    "figures afterwards with visualise.py.")

    # ---- data / model location ----
    parser.add_argument("--data-root", default=None,
                        help="Path to the {example}_out data directory (holds summary.json and the "
                             "per-init reconstruction folders). Required.")
    parser.add_argument("--model-dir", default=None,
                        help="Base dir containing init_pinv/checkpoints/{model}_best.pt (default: .).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: attacks_n<noise>).")
    parser.add_argument("--suite-eps", type=float, default=None,
                        help="Nominal L2 budget (fraction of ||y||) for the attack suite. "
                             "Default: noise_sigma_rel from summary.json — the training "
                             "noise level, the principled budget.")

    # ---- run size ----
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Test samples to attack. The one genuine dial: the suite and the "
                            "epoch study run on different sample counts.")
    
    # ---- optional analysis ----
    parser.add_argument("--epoch-study", action="store_true",
                        help="Instead of the attack suite, attack every saved training "
                             "epoch individually (needs train.py --checkpoint-every N) "
                             "and tabulate adversarial error vs epoch + train/val loss.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model subset for --epoch-study (e.g. "
                             "'nsn'). Default: every model with checkpoints. Each "
                             "model writes its own epoch_study/pinv_<model>.csv, so one "
                             "model per Slurm array task parallelises the study cleanly.")
    parser.add_argument("--lipschitz", action="store_true",
                        help="(Always on.) Estimate the local Lipschitz constant of each "
                             "model's learned correction (attack-independent robustness "
                             "measure).")
    parser.add_argument("--lipschitz-samples", type=int, default=32,
                        help="Clean reconstructions each gain is averaged over.")
    parser.add_argument("--lipschitz-iters", type=int, default=16,
                        help="Power iterations per sample for each gain.")
    parser.add_argument("--lipschitz-restrictions",
                        default=",".join(LIPSCHITZ_RESTRICTIONS),
                        help="Comma-separated subspaces to restrict the gain to, any of "
                             f"{', '.join(LIPSCHITZ_RESTRICTIONS)}. 'null' is the "
                             "architecture-comparable number, 'range' and 'full' say how "
                             "much of the comparison is the architecture rather than the "
                             "learned map, and 'cross' (input in N(A)^perp, output in "
                             "N(A)) is the gain a null-space attack exploits. Each one "
                             "costs a full pass of power iterations.")
    parser.add_argument("--lipschitz-only", action="store_true",
                        help="Skip the attacks and only estimate the Lipschitz gains of "
                             "the best checkpoints, merging them into an existing "
                             "lipschitz.json. Takes minutes rather than hours.")
    parser.add_argument("--epoch-eps", type=float, default=None,
                        help="Budget of --epoch-study (fraction of ||y||). Default: "
                             "noise_sigma_rel from summary.json, the budget of the "
                             "attack suite.")
    parser.add_argument("--epoch-objective", default="mse",
                        choices=sorted(_EPOCH_OBJECTIVES),
                        help="PGD objective of --epoch-study. 'mse' (the total error) "
                             "writes epoch_study/pinv_<model>.csv; any other writes "
                             "epoch_study/pinv_<model>_<objective>.csv next to it. 'null' "
                             "is the one informative about the NSN's learned correction.")
    args = parser.parse_args()
    # Validated here rather than at the Lipschitz stage: that stage runs after
    # hours of attacking, and a typo there would throw the run away.
    try:
        parse_lipschitz_restrictions(args.lipschitz_restrictions)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse()
    set_seed(SEED)
    if args.epoch_study:
        run_epoch_study(args)
    else:
        run_attack_suite(args)
