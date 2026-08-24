#!/usr/bin/env python3
"""Adversarial attack suite for limited-angle Radon reconstruction models.

This is the *attack/compute* half of the attack/visualisation split. It owns:

  * the attack primitives (norm projections, gradient normalisation),
  * the attack objectives and the PGD attack that maximises them,
  * the init reconstructor + model adapter that turn a sinogram into a prediction,
  * per-sample metric evaluation and aggregation,
  * the suite orchestration that attacks every model for every init method.

Each run writes its numeric artifacts to disk (per_sample_metrics.csv, attack_output.npz
with the adversarial sinogram + perturbation, examples.npz/.json,
transfer.npz/.json, summary.json, lipschitz_nullspace.json)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.ellipse_dataloader import get_ellipse_dataloader
from src.radon import AstraRadonAdapter, MatrixRadonAdapter
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

EPOCH_EPS = 0.01

NUM_WORKERS = 4
BATCH_SIZE = 32
MAX_SAMPLES = 128

N_TRAIN = 4000
N_TEST = 1000
SPLIT = "test"

LIPSCHITZ_SAMPLES = 8
LIPSCHITZ_ITERS = 8

SUCCESS_MSE_FACTOR = 2.0
# --------------------------------------------------------------------------- #
# Small tensor helpers.
# --------------------------------------------------------------------------- #
def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()

def l2_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)

def linf_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Largest absolute entry per sample. A diagnostic (the ``delta_linf``
    metric column), not a threat model: the attack itself is L2 only."""
    return x.reshape(x.shape[0], -1).abs().max(dim=1).values

Budget = Union[float, torch.Tensor]

def as_eps_batch(eps: Budget, x: torch.Tensor) -> torch.Tensor:
    """The one representation of a budget: a broadcastable [B,1,1,1] tensor.

    Every production attack is run with a per-sample budget (``suite_eps_batch``:
    eps_i = eps_rel * ||y_i||), so a bright and a faint sinogram are attacked at
    the same relative strength. A plain float is still accepted, meaning "this
    budget for every sample" — convenient for a one-off call or a test.

    Converting once, here, is what lets the projections and the attack steps
    below be written for the tensor case only. They used to carry a scalar branch
    each, which was dead in the suite and drifted: ``project_delta`` was still
    annotated ``eps: float`` while every real caller passed a tensor.

    A negative budget clamps to 0, which the projections turn into a zero
    perturbation — the same thing the old scalar early-returns did.
    """
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
    """Projection onto the feasible set S = range(P_ran) ∩ {||δ||_2 ≤ ε}."""
    return projector(proj_l2_ball(projector(delta), eps))

def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    """Unit-magnitude ascent direction:  ĝ = g / ||g||_2  (steepest ascent under
    the L2 metric)."""
    return grad / l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)

def suite_eps_batch(y_clean: torch.Tensor, eps_nominal: float) -> torch.Tensor:
    """Per-sample budget eps_i for one batch.

    eps_i = eps_nominal * ||y_i||_2 — per sample, so a bright and a faint
    sinogram are attacked at the same *relative* strength."""
    return eps_nominal * l2_norm_batch(y_clean)

def suite_step_size(eps_nominal: float, mean_sino_norm: float, steps: int) -> float:
    """Default PGD step alpha for a suite run: the classic 2.5*eps/steps, in the
    same units as the budget."""
    return 2.5 * eps_nominal * max(mean_sino_norm, 1.0) / max(steps, 1)

def random_start_like(y: torch.Tensor, eps: Budget,
                      projector: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    # A zero budget needs no special case: the draw projects onto a ball of
    # radius 0, giving the zero perturbation.
    return projector(proj_l2_ball(torch.randn_like(y), eps))

def reduce_loss(loss_map: torch.Tensor) -> torch.Tensor:
    if loss_map.ndim <= 1:
        return loss_map.mean()
    return loss_map.reshape(loss_map.shape[0], -1).mean(dim=1).mean()

def per_example_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).reshape(x.shape[0], -1).mean(dim=1)

def batch_mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.abs().reshape(x.shape[0], -1).mean(dim=1)

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
# Init reconstructor + model adapter.
# --------------------------------------------------------------------------- #
class InitReconstructor:
    """The operator that turns a sinogram into the image the network sees.

    Two inits are supported, and they are the two the data generator writes to
    disk: ``fbp`` (filtered backprojection over the measured angles) and
    ``pinv`` (the limited-angle pseudoinverse A_la^+). The iterative inits that
    used to live here -- Landweber and TV Chambolle-Pock -- are gone along with
    their solvers: nothing was ever trained on them, and being non-differentiable
    they forced every attack through the straight-through FBP surrogate."""

    def __init__(self, init_method: str, radon):
        if init_method not in ("fbp", "pinv"):
            raise ValueError(f"Unsupported init method '{init_method}' (fbp or pinv).")
        self.init_method = init_method
        self.radon = radon

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if self.init_method == "pinv":
            return self.radon.backward_la(y)
        return self.radon.fbp_la(y)

class ModelAttackAdapter:
    def __init__(
        self,
        model: nn.Module,
        init_reconstructor: InitReconstructor,
        projector: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.model = model
        self.init_reconstructor = init_reconstructor
        self.projector = projector

    def forward(self, y_adv: torch.Tensor, project: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sinogram -> (prediction, init reconstruction, projected sinogram).

        ``project`` is off inside the PGD loop, where the caller has already
        projected onto the measured rows and needs the graph to start at the
        perturbed sinogram itself."""       
        
        if project:
            y_adv = self.projector(y_adv)
        x_init = self.init_reconstructor(y_adv)
        return self.model(x_init, y_adv), x_init, y_adv

# --------------------------------------------------------------------------- #
# Attack objective + algorithms.
# --------------------------------------------------------------------------- #
def attack_objective(
    pred: torch.Tensor,
    x_gt: torch.Tensor,
    objective: str,
    radon=None,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attack loss to be *maximised*.

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
    if objective == "mse":
        return reduce_loss((pred - x_gt) ** 2)

    if objective == "range":
        # Null-space *complement*: reward only the range (measured) error component
        if radon is None:
            raise ValueError("Objective 'range' requires a radon operator.")
        err = pred - x_gt
        err_range = err - radon.proj_null_image(err)
        return reduce_loss(err_range ** 2)

    if objective == "null":
        # Null-space: reward only the null-space error component
        if radon is None:
            raise ValueError(f"Objective 'null' requires a radon operator.")
        err = pred - x_gt
        return reduce_loss(radon.proj_null_image(err)**2)

    if objective == "zero":
        # Targeted attack: drive the reconstruction toward the zero image
        return -reduce_loss(pred ** 2)

    if objective == "target":
        # General targeted attack: drive the reconstruction toward an arbitrary supplied image
        if target is None:
            raise ValueError("Objective 'target' requires a target image tensor.")
        return -reduce_loss((pred - target.detach()) ** 2)

    raise ValueError(f"Unknown objective '{objective}'")

def pgd_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: Budget,
    alpha: float,
    objective: str,
    target: Optional[torch.Tensor] = None,
) -> AttackResult:
    """Projected gradient ascent on the sinogram perturbation -- the one attack
    the suite runs.

    The feasible set is S = range(P_ran) intersect {||delta||_2 <= eps}: the
    perturbation must live on the measured angles (anything else is not a
    measurement an attacker could make) and stay inside the L2 budget. Each step is

        delta <- Pi_S( delta + alpha * normalize(grad_delta loss) )

    started from a random point of the ball, and the best of ``restarts`` runs
    (highest objective) is returned.

    ``objective`` is maximised; see attack_objective. ``target`` supplies the
    reference image for the targeted 'target' objective and is ignored by the
    others.
    """
    start = time.perf_counter()
    radon = adapter.init_reconstructor.radon
    best_y_adv = y_clean.detach().clone()
    best_delta = torch.zeros_like(y_clean)
    best_score = -float("inf")

    def loss_of(pred):
        return attack_objective(pred, x_gt, objective, radon=radon, target=target)

    for _ in range(SUITE_RESTARTS):
        delta = random_start_like(y_clean, eps, adapter.projector)
        delta = project_delta(delta, eps, adapter.projector)

        for _ in range(SUITE_STEPS):
            with torch.no_grad():
                y_proj = adapter.projector(y_clean + delta)
            y_adv = (y_clean + delta).detach().requires_grad_(True)
            pred, _, _ = adapter.forward(y_adv, project=False)
            grad = torch.autograd.grad(loss_of(pred), y_adv)[0]
            with torch.no_grad():
                delta = (y_proj + alpha * normalize_grad(grad)) - y_clean
                delta = project_delta(delta, eps, adapter.projector)

        with torch.no_grad():
            y_adv = adapter.projector(y_clean + delta)
            pred, _, _ = adapter.forward(y_adv, project=False)
            score = float(loss_of(pred).item())
            if score > best_score:
                best_score = score
                best_y_adv = y_adv.detach().clone()
                best_delta = (best_y_adv - y_clean).detach().clone()

    return AttackResult(y_adv=best_y_adv, delta=best_delta,
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
    angles = np.asarray(summary["angles"], dtype=np.float64)
    phi = tuple(summary["phi"])  # already in radians
    matrix_mode = int(summary.get("matrix_mode", 0))
    if matrix_mode == 1:
        return MatrixRadonAdapter(
            resolution=int(summary["img_size"]),
            angles=angles,
            det_count=int(summary["det_count"]),
            dx=float(summary["dx"]),
            estimate_norm=False,
            device=device,
            dtype=dtype,
            dense=dense,
            phi=phi,
            svd_threshold=float(summary.get("svd_threshold") or 0.0),
            cache_dir="radon_cache",
        )
    return AstraRadonAdapter(
        resolution=int(summary["img_size"]),
        angles=angles,
        det_count=int(summary["det_count"]),
        clip_to_circle=False,
        dx=float(summary["dx"]),
        estimate_norm=False,
        device=device,
        dtype=dtype,
        phi=phi,
    )

def load_model_checkpoint(
    init_method: str,
    model_name: str,
    radon,
    device: torch.device,
    model_dir: Optional[str] = None,
) -> nn.Module:
    base = Path(model_dir) if model_dir else None
    candidates = [
        base / f"init_{init_method}" / "checkpoints" / f"{model_name}_best.pt"
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_name}' and init '{init_method}'. "
            f"Searched:\n  {searched}"
        )

    model = build_models([model_name], radon=radon)[model_name].to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# --------------------------------------------------------------------------- #
# Per-sample metrics.
# --------------------------------------------------------------------------- #
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
) -> List[Dict[str, float]]:
    """Per-sample metrics for one batch: clean and adversarial reconstruction
    quality, the size of the perturbation, and the range/null decomposition of
    both error fields (when a radon operator is supplied)."""
    rows: List[Dict[str, float]] = []
    batch_size = x_gt.shape[0]

    gt_adv = x_gt
    clean_mse_batch = per_example_mse(clean_pred, x_gt)
    adv_mse_batch = per_example_mse(adv_pred, gt_adv)
    delta_l2_batch = l2_norm_batch(delta)
    delta_linf_batch = linf_norm_batch(delta)
    sino_shift_batch = batch_mean_abs(delta)

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

def estimate_lipschitz(
    model: nn.Module,
    clean_cache: List[Tuple],
    radon = None,
) -> Dict[str, float]:
    """Operator-norm (local Lipschitz) estimate of the *learned correction*
    restricted to the null space of A_la.

    Linearise the correction  g(x) = f(x) - x  (= P_null(UNet(x)) for the NSN,
    UNet(x) for the ResNet) around the clean init x0, restrict both input and
    output to null(A_la) with the same projector P = radon.proj_null_image, and
    estimate the largest singular value of  M = P . J_g . P  by power iteration:

        d <- P d / ||.|| ;   repeat:  u = M d ,  d = M^T u / ||.|| ;   sigma ~ ||M d||.

    Attack-independent and architecture-comparable: it measures how strongly a
    null-space input perturbation can be amplified into null-space output error,
    which is what governs worst-case robustness of the learned channel.

    ``clean_cache`` entries only need to supply (x_gt, x_init, y_clean) as their
    first three elements.
    """
    if radon is not None:
        proj = radon.proj_null_image
    samples: List[float] = []

    for entry in clean_cache:
        x_init, y_clean = entry[1], entry[2]
        for b in range(x_init.shape[0]):
            if len(samples) >= LIPSCHITZ_SAMPLES:
                break
            x0 = x_init[b: b + 1].detach()
            y0 = y_clean[b: b + 1].detach()

            def G(x: torch.Tensor) -> torch.Tensor:
                # learned correction, output restricted to the null space
                return proj(model(x, y0) - x) if radon is not None else model(x, y0) - x
            d = proj(torch.randn_like(x0)) if radon is not None else torch.randn_like(x0)
            d = d / (torch.linalg.norm(d.reshape(-1)) + 1e-12)
            for _ in range(LIPSCHITZ_ITERS):
                _, u = torch.autograd.functional.jvp(G, x0, d, strict=False)
                _, w = torch.autograd.functional.vjp(G, x0, proj(u) if radon is not None else u, strict=False)
                w = proj(w) if radon is not None else w
                nw = torch.linalg.norm(w.reshape(-1))
                if nw < 1e-12:
                    break
                d = w / nw
            _, u = torch.autograd.functional.jvp(G, x0, d, strict=False)
            samples.append(float(torch.linalg.norm(proj(u).reshape(-1) if radon is not None else u.reshape(-1)).item()))
        if len(samples) >= LIPSCHITZ_SAMPLES:
            break

    if not samples:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "std": float(np.std(samples)),
        "n": len(samples),
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
    arange = torch.arange(b, device=x_gt.device)
    # Draw a random derangement (a permutation with no fixed point) by rejection
    # sampling so every target is guaranteed to be a *different* sample. A random
    # permutation has no fixed point with probability ~1/e ≈ 0.37, so 20 attempts
    # miss only with negligible probability; the cyclic-shift fallback is a
    # guaranteed derangement for that rare case.
    perm = None
    for _ in range(20):
        cand = torch.randperm(b, generator=generator, device=x_gt.device)
        if not bool((cand == arange).any()):
            perm = cand
            break
    if perm is None:
        shift = int(torch.randint(1, b, (1,), generator=generator, device=x_gt.device).item())
        perm = (arange + shift) % b
    return x_gt[perm]

def detect_suite_models(model_dir: Optional[str], init_method: str) -> List[str]:
    """Return the known model names whose checkpoints exist for this init."""
    base = Path(model_dir) if model_dir else Path(".")
    known = ["resnet", "nsn", "dpnsn", "dpnsn_res"]
    found = []
    for m in known:
        candidates = [
            base / f"init_{init_method}" / "checkpoints" / f"{m}_best.pt"
        ]
        if any(p.exists() for p in candidates):
            found.append(m)
    return found

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
    mean_sino_norm: float
    inits: List[str]
    out_root: Path

def prepare_run(args) -> RunSetup:
    """Resolve the data set, operator and output root shared by all run modes."""
    if not args.data_root:
        raise ValueError("requires --data-root (used to infer dataset type and init methods).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = load_summary(args.data_root)
    dataset_shape = str(summary.get("dataset"))
    noise_rel = float(summary.get("noise_sigma_rel") or 0.0)
    inits = [args.init.lower()] if args.init else detect_data_inits(args.data_root)
    if not inits:
        raise FileNotFoundError(
            f"No init-reconstruction folders (fbp, pinv) found in {args.data_root}.")
    return RunSetup(
        device=device,
        summary=summary,
        radon=build_radon(summary, device=device, dtype=torch.float64 if F64 else torch.float32, dense=not SPARSE),
        noise_rel=noise_rel,
        mean_sino_norm=float(summary.get("mean_norm_y") or 0.0),
        inits=inits,
        out_root=Path(args.out_dir or f"attacks_n{noise_rel}"),
    )

def build_init_inputs(args, radon, init_method: str, max_samples: int, device):
    """Loader, init reconstructor, range projector and the shared input cache for
    one init method — identical in both run modes, so it lives once.

    Returns (init_reconstructor, projector, input_cache). The cache is what makes
    every model see byte-identical inputs, which is the basis for comparing them."""
    loader = get_ellipse_dataloader(
        init_recon=init_method, batch_size=BATCH_SIZE,
        split=SPLIT, n_train=N_TRAIN, n_test=N_TEST,
        shuffle=False, num_workers=NUM_WORKERS, data_root=args.data_root,
    )
    init_recon = InitReconstructor(init_method=init_method, radon=radon)
    proj = lambda y: radon.proj_ran(y)
    return init_recon, proj, build_input_cache(proj, loader, max_samples, device)

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
                      init_reconstructor: Optional["InitReconstructor"] = None) -> Dict:
    """Assemble one example-image row (GT, inits, preds, sinos and range/null
    error decompositions) for the saved examples bundle (rendered later by
    visualise.save_examples)."""
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
    if init_reconstructor is not None:
        # Reference for the NSN range-shift identity: Delta e_ran should equal
        # proj_ran(R_init(delta)), where R_init is the operator that produced
        # the network input (exact init mode). Both inits are linear, so the
        # identity holds for either.
        e_ran_init_d, _ = decompose_error(
            init_reconstructor(delta[i:i + 1]), radon)
        row["proj_ran_init_delta"] = e_ran_init_d.squeeze().numpy()
    return row

def detect_data_inits(data_root) -> List[str]:
    """Init-reconstruction folders present in a data directory (each holding .npy)."""
    root = Path(data_root)
    known = ["fbp", "pinv"]
    return [m for m in known if (root / m).is_dir() and any((root / m).glob("*.npy"))]

def _stack_chunks(chunks: List[torch.Tensor]) -> np.ndarray:
    """Concatenate per-batch [B,1,H,W] tensor chunks and drop the channel axis,
    giving a single [N,H,W] numpy array for the .npz attack_output archive."""
    return torch.cat(chunks, dim=0)[:, 0].numpy()

def run_suite_for_init(args, init_method: str, radon, summary: Dict,
                       noise_rel: float, eps_nominal: float,
                       attacks_root: Path, alpha: float) -> bool:
    """Run the six-attack suite for one init method and write every artifact to
    disk. Returns False (and skips) when no model checkpoints exist for it.

    This function only *computes and saves*; it never plots. The figures are
    produced afterwards by ``visualise.py`` from the artifacts written here."""
    device = radon.device
    model_names = detect_suite_models(args.model_dir, init_method)
    if not model_names:
        print(f"[suite] init '{init_method}': no checkpoints found under "
              f"'{args.model_dir}', skipping.")
        return False
    print(f"\n[suite] ===== init '{init_method}'  models={model_names} =====")

    init_reconstructor, projector, input_cache = build_init_inputs(
        args, radon, init_method, MAX_SAMPLES, device)

    # Load every model + adapter once (reused for both attacking and transfer).
    models: Dict[str, nn.Module] = {}
    adapters: Dict[str, ModelAttackAdapter] = {}
    for name in model_names:
        m = load_model_checkpoint(init_method=init_method, model_name=name,
                                  radon=radon, device=device,
                                  model_dir=args.model_dir)
        models[name] = m
        adapters[name] = ModelAttackAdapter(model=m, init_reconstructor=init_reconstructor,
                                            projector=projector)

    out_root = attacks_root / f"init_{init_method}"
    out_root.mkdir(parents=True, exist_ok=True)

    for attack_name in _SUITE_ATTACKS:
        attack_dir = out_root / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        objective = _SUITE_OBJECTIVE[attack_name]
        print(f"\n[suite] === {attack_name}  (objective={objective}) ===")

        summary_by_model: Dict[str, Dict] = {}
        transfer_pert: Dict[str, torch.Tensor] = {}  # first-batch perturbation per source

        for model_name in model_names:
            adapter = adapters[model_name]
            model = models[model_name]
            rows: List[Dict[str, float]] = []
            example_rows: List[Dict] = []
            worst: List[Tuple[float, Dict]] = []
            delta_chunks: List[torch.Tensor] = []
            yadv_chunks: List[torch.Tensor] = []
            processed = 0

            for bi, (x_gt, clean_init, y_clean) in enumerate(input_cache):
                with torch.no_grad():
                    clean_pred = model(clean_init, y_clean)
                eps_batch = suite_eps_batch(y_clean, eps_nominal)

                # Targeted attacks steer the recon toward a fixed reference:
                # 'zero' targets the zero image internally (target=None), while
                # 'target' needs a per-batch reference — a random *other*
                # sample's ground truth.
                target = (make_other_sample_target(x_gt)
                          if attack_name in _SUITE_TARGETED_ATTACKS else None)
                result = pgd_attack(
                    adapter=adapter, x_gt=x_gt, y_clean=y_clean,
                    clean_pred=clean_pred, eps=eps_batch, alpha=alpha,
                    objective=objective,
                    target=target)
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
                ))

                # One example-image dict for sample j, shared by the first-K
                # saved examples and the worst-case capture so both use
                # identical panel content.
                def make_example_row(j):
                    return build_example_row(
                        radon, x_gt, clean_init, adv_init, clean_pred, adv_pred,
                        y_clean, y_adv, delta, j, init_reconstructor=init_reconstructor)

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
                if processed >= MAX_SAMPLES:
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
                clean_preds0[target] = models[target](clean_init0, y_clean0)
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

    
    lip_res: Dict[str, Dict[str, float]] = {}
    for name in model_names:
        lip_res[name] = estimate_lipschitz(model=models[name], clean_cache=input_cache, radon=None)#radon,)
        r = lip_res[name]
        print(f"[suite][lipschitz] {name} mean={r['mean']:.4g} "
              f"max={r['max']:.4g} (n={r['n']})")
    if lip_res:
        # Plotted later by visualise.render_tree from this json.
        with open(out_root / "lipschitz.json", "w", encoding="utf-8") as f:
            json.dump(lip_res, f, indent=2)

    print(f"[suite] init '{init_method}' done -> {out_root}")
    return True


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

def detect_epoch_checkpoints(model_dir: Optional[str], init_method: str,
                            model_name: str) -> List[Tuple[int, Path]]:
    """Per-epoch checkpoints ``{model}_epoch{NNN}.pt`` written by train.py with
    --checkpoint-every-epoch, returned as [(epoch, path), ...] sorted by epoch."""
    base = Path(model_dir) if model_dir else Path(".")
    d = base / f"init_{init_method}" / "checkpoints"
    out: List[Tuple[int, Path]] = []
    if d.is_dir():
        for pth in d.glob(f"{model_name}_epoch*.pt"):
            try:
                epoch = int(pth.stem.split("_epoch")[1])
            except (IndexError, ValueError):
                continue
            out.append((epoch, pth))
    return sorted(out)

def load_epoch_history(model_dir: Optional[str], init_method: str,
                       model_name: str) -> Tuple[Dict[int, Tuple[float, float]], Optional[int]]:
    """Read {model}_history.json into {epoch: (train_loss, val_loss)} plus the
    best epoch, so the epoch-attack study can overlay attackability on the loss
    curves. Returns ({}, None) when no history was written."""
    base = Path(model_dir) if model_dir else Path(".")
    hp = base / f"init_{init_method}" / "checkpoints" / f"{model_name}_history.json"
    if not hp.exists():
        return {}, None
    blob = json.loads(hp.read_text(encoding="utf-8"))
    hist = {int(h["epoch"]): (float(h.get("train", float("nan"))),
                              float(h.get("val", float("nan"))))
            for h in blob.get("history", [])}
    return hist, blob.get("best_epoch")

def run_epoch_study(args) -> None:
    """Attack every saved epoch of each model individually and tabulate the
    adversarial error vs epoch alongside the train/val loss.

    This isolates *when* attackability arises during training and whether it
    tracks overfitting (validation loss diverging from training loss). For each
    init and model it loads every {model}_epoch{NNN}.pt, runs one PGD attack
    (total-error objective) on the shared sample cache, and writes
    epoch_study/{init}_{model}.csv (rendered by visualise.save_epoch_study_plots).
    Requires train.py to have been run with --checkpoint-every-epoch."""
    setup = prepare_run(args)
    device, summary, radon = setup.device, setup.summary, setup.radon
    inits, out_root = setup.inits, setup.out_root

    eps_nominal = EPOCH_EPS

    study_dir = out_root / "epoch_study"
    study_dir.mkdir(parents=True, exist_ok=True)

    suite_alpha = suite_step_size(eps_nominal, setup.mean_sino_norm, SUITE_STEPS)
    # Optional model subset, so one array task can own one model. Each task writes
    # its own epoch_study/{init}_{model}.csv, so they never collide.
    model_filter = ([m.strip() for m in args.models.split(",") if m.strip()]
                    if getattr(args, "models", None) else None)

    wrote_any = False
    for init_method in inits:
        init_reconstructor, projector, input_cache = build_init_inputs(
            args, radon, init_method, MAX_SAMPLES, device)

        found = detect_suite_models(args.model_dir, init_method)
        if model_filter is not None:
            unknown = [m for m in model_filter if m not in found]
            if unknown:
                raise ValueError(
                    f"--models requested {unknown} but init '{init_method}' only has "
                    f"checkpoints for {found}.")
            found = [m for m in found if m in model_filter]
        for model_name in found:
            ckpts = detect_epoch_checkpoints(args.model_dir, init_method, model_name)
            if not ckpts:
                print(f"[epoch-study] init '{init_method}' model '{model_name}': "
                      f"no per-epoch checkpoints (train with --checkpoint-every-epoch), skipping.")
                continue
            hist, best_epoch = load_epoch_history(args.model_dir, init_method, model_name)
            print(f"\n[epoch-study] init '{init_method}' model '{model_name}': "
                  f"{len(ckpts)} epochs")
            rows_out: List[Dict[str, float]] = []
            for epoch, ckpt_path in ckpts:
                model = build_models([model_name], radon=radon)[model_name].to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
                model.eval()
                adapter = ModelAttackAdapter(model=model, init_reconstructor=init_reconstructor,
                                             projector=projector)
                rows: List[Dict[str, float]] = []
                processed = 0
                for x_gt, clean_init, y_clean in input_cache:
                    with torch.no_grad():
                        clean_pred = model(clean_init, y_clean)
                    eps_batch = suite_eps_batch(y_clean, eps_nominal)
                    result = pgd_attack(
                        adapter=adapter, x_gt=x_gt, y_clean=y_clean,
                        clean_pred=clean_pred, eps=eps_batch, alpha=suite_alpha,
                        objective="mse")
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
                    "clean_rel_l2_median": m.get("clean_rel_l2_median", float("nan")),
                    "adv_rel_l2_mean": m.get("adv_rel_l2_mean", float("nan")),
                    "adv_rel_l2_median": m.get("adv_rel_l2_median", float("nan")),
                    "rel_l2_ratio_median": m.get("rel_l2_ratio_median", float("nan")),
                    "adv_e_nul_frac_median": m.get("adv_e_nul_frac_median", float("nan")),
                    "adv_consistency_rel_median": m.get("adv_consistency_rel_median", float("nan")),
                    "adv_consistency_vs_clean_rel_median": m.get(
                        "adv_consistency_vs_clean_rel_median", float("nan")),
                })
                print(f"  epoch {epoch:03d}  val={va:.5f}  adv_rel_l2(med)="
                      f"{rows_out[-1]['adv_rel_l2_median']:.4f}  ratio(med)="
                      f"{rows_out[-1]['rel_l2_ratio_median']:.3f}")
            csv_path = study_dir / f"{init_method}_{model_name}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                writer.writeheader()
                writer.writerows(rows_out)
            wrote_any = True
            print(f"[epoch-study] wrote {csv_path}")

    if not wrote_any:
        raise FileNotFoundError(
            "No per-epoch checkpoints found. Train with train.py --checkpoint-every-epoch first.")
    print(f"\n[epoch-study] done -> {study_dir}")
    print(f"[epoch-study] render curves with:  python visualise.py {out_root}")

def run_attack_suite(args) -> None:
    if not args.data_root:
        raise ValueError("requires --data-root (used to infer dataset type and init methods).")
    setup = prepare_run(args)
    device, summary, radon = setup.device, setup.summary, setup.radon
    noise_rel = setup.noise_rel
    inits, attacks_root = setup.inits, setup.out_root

    eps_nominal = args.suite_eps if args.suite_eps is not None else noise_rel
    if eps_nominal <= 0:
        raise ValueError(
            "requires noise_sigma_rel in summary.json, or pass --suite-eps explicitly.")

    suite_alpha = suite_step_size(eps_nominal, setup.mean_sino_norm, SUITE_STEPS)
    # eps is a relative L2 fraction: the per-sample budget is eps*||y_i||_2.
    print(f"[suite] dataset={summary.get('dataset')}"
          f"eps={eps_nominal:g}*||y||  alpha={suite_alpha:.4g}  inits={inits}")
    
    print(f"[suite] attacks ({len(_SUITE_ATTACKS)}): {', '.join(_SUITE_ATTACKS)}")
    ran_any = False
    for init_method in inits:
        ran_any |= run_suite_for_init(args, init_method, radon, summary,
                                      noise_rel, eps_nominal, attacks_root, suite_alpha)
    if not ran_any:
        raise FileNotFoundError(
            f"No checkpoints found under model-dir '{args.model_dir}' for any detected init {inits}."
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
                        help="Base dir containing init_{init}/checkpoints/{model}_best.pt (default: .).")
    parser.add_argument("--init", default=None, choices=["fbp", "pinv"],
                        help="Restrict the suite to a single init method. Default: every init "
                             "reconstruction folder detected under --data-root.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: attacks_n<noise>).")
    parser.add_argument("--suite-eps", type=float, default=None,
                        help="Nominal L2 budget (fraction of ||y||) for the attack suite. "
                             "Default: noise_sigma_rel from summary.json — the training "
                             "noise level, the principled budget.")

    # ---- run size ----
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Test samples to attack. The one genuine dial: the suite and the "
                            "epoch study run at different budgets.")
    
    # ---- optional analysis ----
    parser.add_argument("--epoch-study", action="store_true",
                        help="Instead of the attack suite, attack every saved training "
                             "epoch individually (needs train.py --checkpoint-every-epoch) "
                             "and tabulate adversarial error vs epoch + train/val loss.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model subset for --epoch-study (e.g. "
                             "'nsn,dpnsn'). Default: every model with checkpoints. Each "
                             "model writes its own epoch_study/<init>_<model>.csv, so one "
                             "model per Slurm array task parallelises the study cleanly.")
    parser.add_argument("--lipschitz", action="store_true",
                        help="Also estimate the null-restricted local Lipschitz constant of each "
                             "model's learned correction (attack-independent robustness measure).")
    return parser.parse_args()


def main() -> None:
    args = parse()
    set_seed(SEED)
    if args.epoch_study:
        run_epoch_study(args)
    else:
        run_attack_suite(args)
