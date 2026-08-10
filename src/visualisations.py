#!/usr/bin/env python3
"""Plotting / figure generation for the adversarial attack suite.

This is the *visualisation* half of the attack/visualisation split. It owns:

  * every matplotlib figure function (save_examples, the decomposition bars, the
    scatter clouds, the cross-model transfer grids, the Lipschitz bar, ...),
  * ``render_tree`` / ``render_init``, which rebuild every figure of a saved
    attack run purely from the artifacts src/attack.py wrote.

The on-disk artifact format lives in ``src/artifacts.py`` and is re-exported
here, so ``from src.visualisations import read_rows_bundle`` still works. It was
moved out because src/attack.py needs the write_* half: importing them from here
dragged matplotlib into every attack run.

It needs no models and no radon operator: given a saved ``attacks_n<noise>``
tree it can re-render the whole run without re-attacking. (It does still reach
into src/utils for psnr/rel_l2_np/ssim, which imports torch — so "no torch" is
not yet literally true.)

This module must never import ``src/attack.py`` back (keeps the dependency
one-way).

The thin top-level ``visualise.py`` is the CLI entry point and simply calls
``render_tree``.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import (  # re-exported: callers may import either
    read_epoch_study_csv,
    read_metric_rows,
    read_rows_bundle,
    read_transfer_bundle,
    write_rows_bundle,
    write_transfer_bundle,
)
from src.utils import psnr, rel_l2_np, ssim


# Init-method context stamped onto every saved figure. Set once per init run
# (render_init) so figures from different init folders are self-identifying;
# empty string = no tag (e.g. unit tests).
_PLOT_INIT_LABEL = ""


def set_plot_init_label(init_method: str) -> None:
    global _PLOT_INIT_LABEL
    _PLOT_INIT_LABEL = init_method


def _init_tag(prefix: str = "  |  init: ") -> str:
    return f"{prefix}{_PLOT_INIT_LABEL}" if _PLOT_INIT_LABEL else ""


def _metric_caption(m: Optional[Dict[str, float]]) -> str:
    """One-line 'rel-L2 / PSNR / SSIM' caption for a single sample's panel."""
    if not m:
        return ""
    return ("\nrel-L2=%.3f  PSNR=%.1f  SSIM=%.3f"
            % (m["rel_l2"], m["psnr"], m["ssim"]))


def _median_of(rows: List[Dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return float(np.median(vals)) if vals else float("nan")


def _decomp_metric_caption(m: Optional[dict]) -> str:

    """Append 'SSIM / PSNR / rel-L2' for a single panel, or '' when not given.



    For the range/null panels the metric is scored on the *component-only*

    reconstruction (gt + e_component vs gt), i.e. how much that error subspace

    alone degrades the image — the SSIM/PSNR are non-additive so they cannot be

    read off the L2 fractions."""

    if not m:

        return ""

    return "\nSSIM=%.3f  PSNR=%.1f  rel-L2=%.3f" % (m["ssim"], m["psnr"], m["rel_l2"])

def visualise_decomposition(

    gt: np.ndarray,

    recon: np.ndarray,

    e_ran: np.ndarray,

    e_nul: np.ndarray,

    out_path: Path,

    title: str = "",

    recon_metric: Optional[dict] = None,

    ran_metric: Optional[dict] = None,

    nul_metric: Optional[dict] = None,

) -> None:

    """

    Save a 1×5 figure: GT | Recon | Error | Range error | Null-space error.



    Parameters

    ----------

    gt, recon : (H, W) arrays

    e_ran, e_nul : (H, W) range and null-space components of (recon - gt)

    out_path : where to save the PNG

    title : overall figure title

    recon_metric, ran_metric, nul_metric : optional {"ssim","psnr","rel_l2"} dicts

        for this sample. When given, the value is stamped on the corresponding

        panel — ran_metric/nul_metric score the component-only reconstruction so

        you can read the SSIM/PSNR of the range- and null-space error directly.

    """

    error = recon - gt

    e_abs = np.abs(error).max()

    e_norm = np.linalg.norm(error.ravel())

    ran_frac = np.linalg.norm(e_ran.ravel()) / max(e_norm, 1e-12)

    nul_frac = np.linalg.norm(e_nul.ravel()) / max(e_norm, 1e-12)



    fig, axes = plt.subplots(1, 5, figsize=(20, 4))



    im0 = axes[0].imshow(gt, cmap="gray")

    axes[0].set_title("Ground Truth")

    axes[0].axis("off")

    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)



    im1 = axes[1].imshow(recon, cmap="gray")

    axes[1].set_title("Reconstruction" + _decomp_metric_caption(recon_metric))

    axes[1].axis("off")

    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)



    im2 = axes[2].imshow(error, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

    axes[2].set_title("Error")

    axes[2].axis("off")

    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)



    im3 = axes[3].imshow(e_ran, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

    axes[3].set_title(f"Range error\n‖e_ran‖/‖e‖={ran_frac:.3f}"

                      + _decomp_metric_caption(ran_metric))

    axes[3].axis("off")

    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)



    im4 = axes[4].imshow(e_nul, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

    axes[4].set_title(f"Null-space error\n‖e_nul‖/‖e‖={nul_frac:.3f}"

                      + _decomp_metric_caption(nul_metric))

    axes[4].axis("off")

    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)



    if title:

        fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-example qualitative figures.
# --------------------------------------------------------------------------- #
def save_examples(
    out_dir: Path,
    example_rows: List[Dict],
) -> None:
    if not example_rows:
        return

    for idx, row in enumerate(example_rows):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        images = [
            (row["x_gt"], "Ground Truth", "gray"),
            (row["clean_init"], "Clean Init" + _metric_caption(row.get("m_clean_init")), "gray"),
            (row["adv_init"], "Adv Init" + _metric_caption(row.get("m_adv_init")), "gray"),
            (row["delta"], "Sinogram Delta", "viridis"),
            (row["clean_pred"], "Clean Pred" + _metric_caption(row.get("m_clean_pred")), "gray"),
            (row["adv_pred"], "Adv Pred" + _metric_caption(row.get("m_adv_pred")), "gray"),
            (row["clean_y"], "Clean Sino", "gray"),
            (row["adv_y"], "Adv Sino", "gray"),
        ]
        for ax, (img, title, cmap) in zip(axes.reshape(-1), images):
            im = ax.imshow(img, cmap=cmap, aspect="auto" if img.ndim == 2 and img.shape[0] != img.shape[1] else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Attack example {idx}{_init_tag()}", fontsize=11)
        plt.savefig(out_dir / f"example_{idx:03d}.png", dpi=160)
        plt.close(fig)

        if "e_ran_init_clean" in row:
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["clean_init"],
                e_ran=row["e_ran_init_clean"],
                e_nul=row["e_nul_init_clean"],
                out_path=out_dir / f"decomp_init_clean_{idx:03d}.png",
                title=f"Clean — init error decomposition, before network (example {idx}){_init_tag()}",
                recon_metric=row.get("m_clean_init"),
                ran_metric=row.get("m_ran_init_clean"),
                nul_metric=row.get("m_nul_init_clean"),
            )
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["adv_init"],
                e_ran=row["e_ran_init_adv"],
                e_nul=row["e_nul_init_adv"],
                out_path=out_dir / f"decomp_init_adv_{idx:03d}.png",
                title=f"Adversarial — init error decomposition, before network (example {idx}){_init_tag()}",
                recon_metric=row.get("m_adv_init"),
                ran_metric=row.get("m_ran_init_adv"),
                nul_metric=row.get("m_nul_init_adv"),
            )

        if "e_ran_clean" in row:
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["clean_pred"],
                e_ran=row["e_ran_clean"],
                e_nul=row["e_nul_clean"],
                out_path=out_dir / f"decomp_clean_{idx:03d}.png",
                title=f"Clean — error decomposition (example {idx}){_init_tag()}",
                recon_metric=row.get("m_clean_pred"),
                ran_metric=row.get("m_ran_clean"),
                nul_metric=row.get("m_nul_clean"),
            )
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["adv_pred"],
                e_ran=row["e_ran_adv"],
                e_nul=row["e_nul_adv"],
                out_path=out_dir / f"decomp_adv_{idx:03d}.png",
                title=f"Adversarial — error decomposition (example {idx}){_init_tag()}",
                recon_metric=row.get("m_adv_pred"),
                ran_metric=row.get("m_ran_adv"),
                nul_metric=row.get("m_nul_adv"),
            )
            e_ran_c = row["e_ran_clean"]
            e_ran_a = row["e_ran_adv"]
            e_ran_diff = e_ran_a - e_ran_c
            e_abs = max(np.abs(e_ran_c).max(), np.abs(e_ran_a).max(), 1e-12)
            diff_abs = max(np.abs(e_ran_diff).max(), 1e-12)
            panels = [
                (e_ran_c, f"e_ran clean  ‖·‖={np.linalg.norm(e_ran_c.ravel()):.3f}", e_abs),
                (e_ran_a, f"e_ran adv    ‖·‖={np.linalg.norm(e_ran_a.ravel()):.3f}", e_abs),
                (e_ran_diff, f"Δe_ran (adv − clean)  ‖·‖={np.linalg.norm(e_ran_diff.ravel()):.3f}", diff_abs),
            ]
            if "proj_ran_init_delta" in row:
                p = row["proj_ran_init_delta"]
                panels.append((p, f"proj_ran(R_init(δ))  ‖·‖={np.linalg.norm(p.ravel()):.3f}", diff_abs))
            ncols = len(panels)
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            if ncols == 1:
                axes = [axes]
            for ax, (img, title, vabs) in zip(axes, panels):
                im = ax.imshow(img, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"Range-space error shift (example {idx}) — "
                f"should equal proj_ran(R_init(δ)) for NSN (linear init){_init_tag()}",
                fontsize=9,
            )
            plt.tight_layout()
            plt.savefig(out_dir / f"range_diff_{idx:03d}.png", dpi=150)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# Aggregate bar / scatter figures.
# --------------------------------------------------------------------------- #
def save_decomposition_bar(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    clean_key: str = "clean_e_nul_frac",
    adv_key: str = "adv_e_nul_frac",
    fname: str = "decomp_nul_frac.png",
    title: str = "Null-space fraction of error: clean vs adversarial (median)",
) -> None:
    """Grouped bar chart of the null-space fraction of the error (‖e_nul‖/‖e‖), clean
    vs adversarial, per model. Shows which channel the attack pushes each model's
    error into (direction depends on model and objective: e.g. an MSE attack drives
    the NSN's error into the passed-through range channel but the ResNet's into
    hallucinated null-space structure).

    Call with the ``*_init_e_nul_frac`` keys to draw the same chart for the
    initialisation reconstruction (the network input, before the NSN)."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return

    clean_nul = [_median_of(rows_by_model[m], clean_key) for m in models]
    adv_nul = [_median_of(rows_by_model[m], adv_key) for m in models]
    if all(math.isnan(v) for v in clean_nul + adv_nul):
        return

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, clean_nul, w, label="clean", color="#1D9E75")
    ax.bar(x + w / 2, adv_nul, w, label="adversarial", color="#D4537E")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("‖e_nul‖ / ‖e‖  (median)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title + _init_tag())
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def save_totals_decomposition_bar(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    eps: float,
    attack_name: str,
    fname: str = "error_totals_decomposition.png",
) -> None:
    """Grouped bar chart per model: the TOTAL rel-L2 error with its range and
    null-space components directly alongside, clean vs adversarial (median).

    total² = range² + null² (orthogonal channels), so this shows at a glance
    which channel the total error of each model lives in — e.g. the NSN's large
    total under a plain MSE attack is almost entirely the data-consistent range
    channel (passed-through noise), while the ResNet's is null-space
    (hallucinated structure). Log y-scale so clean and adversarial bars remain
    readable together."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    comps = [("rel_l2", "total", "#555555"),
             ("rel_l2_ran", "range", "#1f77b4"),
             ("rel_l2_nul", "null", "#d62728")]
    vals: Dict[Tuple[str, str], List[float]] = {}
    for cond in ("clean", "adv"):
        for key, _, _ in comps:
            vals[(cond, key)] = [_median_of(rows_by_model[m], f"{cond}_{key}") for m in models]
    if all(math.isnan(v) for vs in vals.values() for v in vs):
        return

    x = np.arange(len(models))
    w = 0.13
    fig, ax = plt.subplots(figsize=(2.6 * len(models) + 4, 5.2))
    for ci, (key, label, color) in enumerate(comps):
        ax.bar(x + (ci - 2.5) * w, vals[("clean", key)], w,
               color=color, alpha=0.4, label=f"clean {label}")
    for ci, (key, label, color) in enumerate(comps):
        ax.bar(x + (ci + 0.5) * w, vals[("adv", key)], w,
               color=color, alpha=1.0, label=f"adv {label}")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yscale("log")
    ax.set_ylabel("rel-L2 error  (median, log scale)")
    ax.set_title(f"Total error next to its range/null decomposition — clean vs adversarial\n"
                 f"({attack_name}, eps={eps:g};  total² = range² + null²{_init_tag(';  init: ')})", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def save_null_growth_headline(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    eps: float,
    attack_name: str,
) -> None:
    """headline: for each model, the null-space error magnitude ||e_nul|| clean vs
    adversarial, with the adversarial range floor ||e_ran|| shown alongside. The
    growth of the null channel is the fair robustness signal; the range floor is the
    shared inversion error every data-consistent method carries."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    clean_nul = [_median_of(rows_by_model[m], "clean_e_nul_l2") for m in models]
    adv_nul = [_median_of(rows_by_model[m], "adv_e_nul_l2") for m in models]
    adv_ran = [_median_of(rows_by_model[m], "adv_e_ran_l2") for m in models]
    if all(math.isnan(v) for v in clean_nul + adv_nul):
        return
    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 3, 5))
    ax.bar(x - w, clean_nul, w, label="||e_nul|| clean", color="#9ecae1")
    ax.bar(x, adv_nul, w, label="||e_nul|| adversarial", color="#d62728")
    ax.bar(x + w, adv_ran, w, label="||e_ran|| adversarial", color="#7f7f7f", alpha=0.7)
    for xi, c, a in zip(x, clean_nul, adv_nul):
        if not (math.isnan(c) or math.isnan(a)):
            ax.annotate("d_null=%+.3g" % (a - c), (xi - w / 2, max(c, a)),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||error component||  (median)")
    ax.set_title("Headline - null-space error growth under attack "
                 "(%s, eps=%g)%s\nfair signal = growth of ||e_nul||; for data-consistent "
                 "models ||e_ran|| equals the init's range error (inversion floor)"
                 % (attack_name, eps, _init_tag()), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "headline_null_growth.png", dpi=150)
    plt.close(fig)


def save_consistency_plot(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    eps: float,
) -> None:
    """measurement (data) consistency ||proj_ran(A x_hat) - y||/||y||, clean vs adv,
    per model. Low/flat under attack => the adversarial error hides in the null /
    small-singular-value subspace (the literature finding), which is exactly why a
    null-targeted attack/metric is the fair lens."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    clean_c = [_median_of(rows_by_model[m], "clean_consistency_rel") for m in models]
    adv_c = [_median_of(rows_by_model[m], "adv_consistency_rel") for m in models]
    if all(math.isnan(v) for v in clean_c + adv_c):
        return
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.bar(x - w / 2, clean_c, w, label="clean", color="#1D9E75")
    ax.bar(x + w / 2, adv_c, w, label="adversarial", color="#D4537E")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||proj_ran(A x_hat) - y|| / ||y||  (median)")
    ax.set_title("Measurement consistency: clean vs adversarial (eps=%g)%s\n"
                 "stays low => adversarial error lives in the null subspace"
                 % (eps, _init_tag()), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "measurement_consistency.png", dpi=150)
    plt.close(fig)


def save_lipschitz_plot(out_dir: Path, lip_res: Dict[str, Dict[str, float]]) -> None:
    """bar chart: null-restricted local Lipschitz constant per model (mean +/- std,
    max marked). Higher => the learned channel amplifies null-space perturbations more,
    i.e. is intrinsically less robust there - independent of any particular attack."""
    models = [m for m in lip_res if lip_res[m].get("n", 0) > 0]
    if not models:
        return
    means = [lip_res[m]["mean"] for m in models]
    stds = [lip_res[m]["std"] for m in models]
    maxes = [lip_res[m]["max"] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.bar(x, means, 0.5, yerr=stds, capsize=4, color="#4C72B0", label="mean +/- std")
    ax.scatter(x, maxes, color="#C44E52", zorder=3, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||P.J_g.P||  (null-restricted local Lipschitz)")
    ax.set_title("null-restricted Lipschitz of the learned correction%s\n"
                 "operator norm of P_null . J(f-x) . P_null (power iteration)"
                 % _init_tag(), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "lipschitz_nullspace.png", dpi=150)
    plt.close(fig)


def save_cross_model_examples(out_dir: Path, source_model: str, model_names: List[str],
                              gt_imgs: List[np.ndarray], clean_pred_source: List[np.ndarray],
                              preds: Dict[Tuple[str, str], List[np.ndarray]],
                              n_ex: int, attack_name: str) -> None:
    """Per-example cross-model figures, saved into the *source* model's folder:
    the perturbation crafted on ``source_model`` plugged into every model
    (e.g. the NSN's adversarial noise fed to the ResNet and vice versa).
    Columns: GT | source clean recon | each model's recon under the source's δ,
    annotated with the rel-L2 error vs GT.

    ``gt_imgs`` / ``clean_pred_source`` are lists of 2-D image arrays and
    ``preds`` maps (source, target) -> list of 2-D image arrays (all restored
    from the transfer artifact, so no tensors are needed here)."""
    for idx in range(n_ex):
        gt_np = gt_imgs[idx]
        panels = [(gt_np, "GT"),
                  (clean_pred_source[idx], f"{source_model} clean")]
        for m in model_names:
            img = preds[(source_model, m)][idx]
            tag = " (self)" if m == source_model else ""
            panels.append((img, f"{m} ← {source_model}'s δ{tag}"))
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.8), squeeze=False)
        for ax, (img, title) in zip(axes[0], panels):
            ax.imshow(img, cmap="gray")
            err = rel_l2_np(img, gt_np)
            ax.set_title(f"{title}\nrel-L2 vs GT = {err:.3f}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"'{source_model}' {attack_name} perturbation plugged into every model "
                     f"(example {idx}){_init_tag()}", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"transfer_example_{idx:03d}.png", dpi=150)
        plt.close(fig)


def save_transfer_figure(out_path: Path, source_model: str, target_models: List[str],
                         recon_by_model: Dict[str, List[np.ndarray]],
                         gt_imgs: List[np.ndarray], attack_name: str,
                         title: Optional[str] = None, gt_label: str = "GT") -> None:
    """Grid: rows = samples, cols = GT + one reconstruction per model, all using
    the perturbation crafted on ``source_model``. Shows what that adversarial
    noise does inside every other reconstruction method."""
    T = len(gt_imgs)
    if T == 0:
        return
    ncols = 1 + len(target_models)
    fig, axes = plt.subplots(T, ncols, figsize=(3 * ncols, 3 * T), squeeze=False)
    for r in range(T):
        ax = axes[r][0]
        ax.imshow(gt_imgs[r], cmap="gray")
        if r == 0:
            ax.set_title(gt_label, fontsize=9)
        ax.axis("off")
        for c, m in enumerate(target_models):
            ax = axes[r][c + 1]
            ax.imshow(recon_by_model[m][r], cmap="gray")
            if r == 0:
                ax.set_title(m + ("  (source)" if m == source_model else ""), fontsize=9)
            ax.axis("off")
    fig.suptitle((title or f"Transfer of '{source_model}' {attack_name} perturbation across models")
                 + _init_tag(), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _rows_xy(rows, x_key, y_key):
    xs = [r[x_key] for r in rows if x_key in r and y_key in r]
    ys = [r[y_key] for r in rows if x_key in r and y_key in r]
    return xs, ys


def _identity_line(ax) -> None:
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=1.0, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def save_suite_scatter(out_dir: Path, rows_by_model: Dict[str, List[Dict]],
                       attack_name: str, eps: float) -> None:
    """Per-sample cross-model scatter clouds for one attack.

    Left: adversarial error split into its range (x) and null-space (y) channels,
    one colour per model - shows which channel each model's error lives in under
    this attack (the suite's central comparison; a data-consistent NSN keeps its
    points low on the null axis, a ResNet does not).
    Right: clean (x) vs adversarial (y) relative-L2 error per sample with the y=x
    line; points above the line are the samples the attack actually degraded."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    colors = plt.cm.tab10.colors
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, m in enumerate(models):
        c = colors[i % len(colors)]
        xr, yn = _rows_xy(rows_by_model[m], "adv_e_ran_l2", "adv_e_nul_l2")
        if xr:
            ax0.scatter(xr, yn, s=20, alpha=0.6, color=c, label=m)
        xc, ya = _rows_xy(rows_by_model[m], "clean_rel_l2", "adv_rel_l2")
        if xc:
            ax1.scatter(xc, ya, s=20, alpha=0.6, color=c, label=m)
    ax0.set_xlabel("range error  ||e_ran||  (data-consistent channel)")
    ax0.set_ylabel("null-space error  ||e_nul||  (structural channel)")
    ax0.set_title("adversarial error by channel")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)
    ax1.set_xlabel("clean error  (rel L2)")
    ax1.set_ylabel("adversarial error  (rel L2)")
    ax1.set_title("clean vs adversarial error")
    ax1.grid(True, alpha=0.3)
    _identity_line(ax1)
    ax1.legend(fontsize=8)
    fig.suptitle(f"{attack_name}: per-sample cross-model scatter (eps={eps:g}){_init_tag()}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_cross_model.png", dpi=150)
    plt.close(fig)


def save_attack_comparison_scatter(out_dir: Path,
                                   all_rows: Dict[str, Dict[str, List[Dict]]],
                                   eps: float) -> None:
    """Init-level summary: one panel per model, per-sample adversarial error split
    into range (x) vs null (y), coloured by attack. Shows how the four attacks
    redistribute a model's error between the data-consistent and structural
    channels (null-targeted attacks push points upward, range-targeted rightward)."""
    attacks = [a for a in all_rows if all_rows[a]]
    if not attacks:
        return
    models = sorted({m for a in attacks for m in all_rows[a] if all_rows[a][m]})
    if not models:
        return
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.2), squeeze=False)
    for col, m in enumerate(models):
        ax = axes[0][col]
        for i, a in enumerate(attacks):
            xr, yn = _rows_xy(all_rows[a].get(m, []), "adv_e_ran_l2", "adv_e_nul_l2")
            if xr:
                ax.scatter(xr, yn, s=18, alpha=0.6, color=colors[i % len(colors)], label=a)
        ax.set_xlabel("range error  ||e_ran||")
        if col == 0:
            ax.set_ylabel("null-space error  ||e_nul||")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Adversarial error by channel, per model across attacks (eps={eps:g}){_init_tag()}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_attacks_by_channel.png", dpi=150)
    plt.close(fig)



# --------------------------------------------------------------------------- #
# Data-consistency overview, null-space structure analysis, and attack overview.
# All read only the artifacts a run already wrote (the per-attack summary.json,
# the per_sample_metrics rows and the examples bundles), so they re-render
# without torch, a model or the radon operator like everything else here.
# --------------------------------------------------------------------------- #
def save_consistency_overview(init_dir: Path, all_rows: Dict[str, Dict[str, List[Dict]]],
                              eps: float) -> None:
    """Across every attack, the measurement-consistency residual
    ||proj_ran(A x_hat) - y|| / ||y|| of each model: adversarial (grouped bars,
    one per attack) against the clean value (marker, attack-independent).

    A data-consistent model (NSN / DPNSN) keeps the adversarial residual pinned
    at its clean marker — the adversarial error lives in the null / small-
    singular-value subspace it cannot alter — while an unconstrained ResNet's
    residual can rise. This is the cross-attack view of measurement_consistency."""
    # Per model: data-consistency residual  c(x̂) = ||P_ran(A_la x̂) - y|| / ||y||,
    # median over samples, clean vs adversarial. Data-consistent models keep
    # c_adv ≈ c_clean because their adversarial error lives in null(A_la).
    attacks = [a for a in all_rows if all_rows[a]]
    if not attacks:
        return
    models = sorted({m for a in attacks for m in all_rows[a] if all_rows[a][m]})
    if not models:
        return
    colors = plt.cm.tab10.colors
    x = np.arange(len(models))
    nb = max(len(attacks), 1)
    width = 0.8 / nb
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3, 5))
    any_bar = False
    for i, a in enumerate(attacks):
        adv = [_median_of(all_rows[a].get(m, []), "adv_consistency_rel") for m in models]
        any_bar = any_bar or any(not math.isnan(v) for v in adv)
        ax.bar(x + (i - (nb - 1) / 2.0) * width, adv, width,
               color=colors[i % len(colors)], label=f"{a} (adv)")
    if not any_bar:
        plt.close(fig)
        return
    clean = []
    for m in models:
        val = float("nan")
        for a in attacks:
            v = _median_of(all_rows[a].get(m, []), "clean_consistency_rel")
            if not math.isnan(v):
                val = v
                break
        clean.append(val)
    ax.scatter(x, clean, color="k", marker="_", s=400, zorder=5, label="clean (all attacks)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||proj_ran(A x_hat) - y|| / ||y||  (median)")
    ax.set_title("Data-consistency across attacks (eps=%g)%s\n"
                 "flat at the clean marker => adversarial error is invisible to the "
                 "measurements (data-consistent)" % (eps, _init_tag()), fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(init_dir) / "consistency_overview.png", dpi=150)
    plt.close(fig)


def save_ghost_structure_plot(attack_dir: Path, rows_by_model: Dict[str, List[Dict]],
                              eps: float, attack_name: str) -> None:
    """How ghost-like is each model's adversarial error? Box plots of the null-space
    energy fraction ||e_nul|| / ||e|| of the per-sample adversarial error.

    A value near 1 means the error lives (almost) entirely in null(A_la) — invisible
    to the measurements, i.e. a data-consistent 'ghost'. The clean null fraction
    (green marker) is the no-attack baseline; the attack pushing the box up toward 1
    is the signature of ghost-like structure being synthesised rather than plain
    measured-domain error."""
    # Ghost fraction per sample:  g = ||P_N e|| / ||e|| = ||e_nul|| / ||e||,
    # e = adv_pred - x_gt.  g -> 1  <=>  the adversarial error is entirely in
    # null(A_la), i.e. measurement-invisible (ghost-like).
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    adv_series: List[List[float]] = []
    clean_med: List[float] = []
    have = False
    for m in models:
        adv = [r["adv_e_nul_frac"] for r in rows_by_model[m] if "adv_e_nul_frac" in r]
        cln = [r["clean_e_nul_frac"] for r in rows_by_model[m] if "clean_e_nul_frac" in r]
        adv_series.append(adv or [float("nan")])
        clean_med.append(float(np.median(cln)) if cln else float("nan"))
        have = have or bool(adv)
    if not have:
        return
    x = np.arange(len(models)) + 1
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.boxplot(adv_series, positions=x, widths=0.5, showfliers=False)
    ax.scatter(x, clean_med, color="#1D9E75", zorder=5, label="clean null fraction")
    ax.axhline(1.0, color="0.6", ls="--", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||e_nul|| / ||e||  (adversarial error, per sample)")
    ax.set_title("%s: ghost-likeness of the adversarial error (eps=%g)%s\n"
                 "box near 1 => error hides in the null space (ghost-like); "
                 "green = clean baseline" % (attack_name, eps, _init_tag()), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(attack_dir) / "ghost_structure.png", dpi=150)
    plt.close(fig)


def collect_attack_overview(attacks_root) -> List[Dict]:
    """One record per (init, attack, model) with the headline adversarial metrics,
    read from the per-attack summary.json files. Pure — needs no matplotlib."""
    root = Path(attacks_root)
    init_dirs = sorted(p for p in root.glob("init_*") if p.is_dir())
    if not init_dirs:
        init_dirs = [root]
    records: List[Dict] = []
    for init_dir in init_dirs:
        init_name = (init_dir.name[len("init_"):]
                     if init_dir.name.startswith("init_") else init_dir.name)
        for summ_path in sorted(init_dir.glob("*/summary.json")):
            summ = json.loads(Path(summ_path).read_text(encoding="utf-8"))
            attack = summ.get("attack", summ_path.parent.name)
            eps = summ.get("eps")
            for model, m in (summ.get("models") or {}).items():
                records.append({
                    "init": init_name, "attack": attack, "model": model, "eps": eps,
                    "adv_rel_l2_median": m.get("adv_rel_l2_median", float("nan")),
                    "adv_ssim_median": m.get("adv_ssim_median", float("nan")),
                    "adv_e_nul_frac_median": m.get("adv_e_nul_frac_median", float("nan")),
                    "adv_consistency_rel_median": m.get("adv_consistency_rel_median", float("nan")),
                    "example_npz": str(summ_path.parent / model / "examples.npz"),
                    "worst_npz": str(summ_path.parent / model / "worst.npz"),
                    "worst_json": str(summ_path.parent / model / "worst.json"),
                })
    return records


def write_overview_markdown(attacks_root, records: List[Dict], out_path) -> None:
    """Write overview.md: every executed attack with its headline adversarial
    metrics per model, grouped by init. Pure text; testable without matplotlib."""
    lines = ["# Attack suite overview", ""]
    if not records:
        lines.append("_No attack summaries found._")
        Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    inits = sorted({r["init"] for r in records})
    attacks = sorted({r["attack"] for r in records})
    models = sorted({r["model"] for r in records})
    lines += [f"- inits: {', '.join(inits)}",
              f"- attacks: {', '.join(attacks)}",
              f"- models: {', '.join(models)}", ""]
    for init in inits:
        lines += [f"## init: {init}", "",
                  "| attack | model | adv rel-L2 | worst rel-L2 | adv SSIM | null-frac | data-consistency |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in sorted([r for r in records if r["init"] == init],
                        key=lambda r: (r["attack"], r["model"])):
            lines.append("| %s | %s | %.4f | %.4f | %.4f | %.3f | %.4f |" % (
                r["attack"], r["model"], r["adv_rel_l2_median"], _worst_rel_l2(r),
                r["adv_ssim_median"], r["adv_e_nul_frac_median"],
                r["adv_consistency_rel_median"]))
        lines += ["", f"Montage: `overview_{init}.png`", ""]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worst_rel_l2(record: Dict) -> float:
    """Worst sample's prediction rel-L2 from a record's worst.json bundle (the
    first / most-degraded row), or NaN when unavailable."""
    p = Path(record.get("worst_json", ""))
    if not p.exists():
        return float("nan")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if not meta:
        return float("nan")
    m = meta[0].get("m_adv_pred") or {}
    if "rel_l2" in m:
        return float(m["rel_l2"])
    return float(meta[0].get("worst_score", float("nan")))


def _load_example_image(npz_path, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    """First matching ``ex0__<key>`` image from an examples.npz, else None."""
    p = Path(npz_path)
    if not p.exists():
        return None
    with np.load(p) as z:
        for k in keys:
            if f"ex0__{k}" in z.files:
                return z[f"ex0__{k}"]
    return None


def _overview_montage(recs: List[Dict], out_path, title: str,
                      get_npz, get_caption) -> bool:
    """One montage: rows = attacks, cols = models, each cell an image pulled from
    ``get_npz(record)`` captioned by ``get_caption(record)``. Returns True if any
    image was drawn."""
    attacks = sorted({r["attack"] for r in recs})
    models = sorted({r["model"] for r in recs})
    by = {(r["attack"], r["model"]): r for r in recs}
    fig, axes = plt.subplots(len(attacks), len(models),
                             figsize=(3.0 * len(models) + 1, 3.0 * len(attacks) + 1),
                             squeeze=False)
    drew = False
    for i, a in enumerate(attacks):
        for j, mdl in enumerate(models):
            ax = axes[i][j]
            ax.axis("off")
            r = by.get((a, mdl))
            if r is None:
                continue
            img = _load_example_image(get_npz(r), ("adv_pred", "pred"))
            if img is not None:
                ax.imshow(img, cmap="gray")
                drew = True
            ax.set_title(get_caption(r), fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return drew


def save_attack_overview(attacks_root) -> None:
    """Collect every executed attack into per-init montages plus an overview.md
    index of the headline metrics. Produces a sample-0 montage and — when the
    worst-case bundles are present — a worst-case montage of the samples each
    attack degraded most. Reads only saved artifacts."""
    root = Path(attacks_root)
    records = collect_attack_overview(root)
    write_overview_markdown(root, records, root / "overview.md")
    if not records:
        return
    init_dirs = sorted(p for p in root.glob("init_*") if p.is_dir()) or [root]
    for init_dir in init_dirs:
        init_name = (init_dir.name[len("init_"):]
                     if init_dir.name.startswith("init_") else init_dir.name)
        recs = [r for r in records if r["init"] == init_name]
        if not recs:
            continue
        _overview_montage(
            recs, root / f"overview_{init_name}.png",
            "Attack overview - sample adversarial reconstructions (init: %s)" % init_name,
            get_npz=lambda r: r["example_npz"],
            get_caption=lambda r: "%s / %s\nrel-L2=%.3f"
            % (r["attack"], r["model"], r["adv_rel_l2_median"]))
        if any(Path(r.get("worst_npz", "")).exists() for r in recs):
            _overview_montage(
                recs, root / f"overview_worst_{init_name}.png",
                "Attack overview - WORST-case adversarial reconstructions (init: %s)" % init_name,
                get_npz=lambda r: r.get("worst_npz", ""),
                get_caption=lambda r: "%s / %s\nworst rel-L2=%.3f"
                % (r["attack"], r["model"], _worst_rel_l2(r)))




def save_epoch_attackability_plot(csv_path, out_path) -> None:
    """Adversarial error vs training epoch overlaid on the train/val loss curves.

    Left axis: clean and adversarial reconstruction rel-L2 (median) per epoch.
    Right axis: train and validation loss. The best-val epoch is marked. If
    attackability (adv rel-L2) keeps rising after the validation loss bottoms out
    and starts diverging from the training loss, the extra vulnerability is
    tracking overfitting rather than better fitting — the question this study
    exists to answer."""
    # Per epoch t: median adversarial rel-L2  r_adv(t) = median_i ||x̂_adv - x_gt||/||x_gt||
    # overlaid on the train/val loss L_train(t), L_val(t). If r_adv keeps growing
    # while L_val turns up away from L_train (overfitting onset), attackability is
    # tracking overfitting rather than better fit.
    rows = read_epoch_study_csv(csv_path)
    if not rows:
        return
    rows.sort(key=lambda r: r.get("epoch", 0.0))
    ep = [r.get("epoch", float("nan")) for r in rows]
    adv = [r.get("adv_rel_l2_median", float("nan")) for r in rows]
    clean = [r.get("clean_rel_l2_median", float("nan")) for r in rows]
    train = [r.get("train_loss", float("nan")) for r in rows]
    val = [r.get("val_loss", float("nan")) for r in rows]
    best = [r.get("epoch") for r in rows if r.get("is_best", 0.0) >= 1.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    l1, = ax.plot(ep, adv, color="#d62728", marker="o", ms=3, label="adv rel-L2 (median)")
    l2, = ax.plot(ep, clean, color="#1f77b4", marker="o", ms=3, label="clean rel-L2 (median)")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("reconstruction rel-L2")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    l3, = ax2.plot(ep, train, color="#7f7f7f", ls="--", label="train loss")
    l4, = ax2.plot(ep, val, color="#2ca02c", ls="--", label="val loss")
    ax2.set_ylabel("training loss")

    handles = [l1, l2, l3, l4]
    if best:
        bl = ax.axvline(best[0], color="0.4", ls=":", lw=1.2)
        handles.append(bl)
        bl.set_label(f"best-val epoch ({int(best[0])})")
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title("Attackability vs epoch — %s%s\n"
                 "adv rel-L2 rising while val loss diverges from train => "
                 "attackability tracks overfitting" % (Path(csv_path).stem, _init_tag()),
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_epoch_study_plots(attacks_root) -> None:
    """Render every epoch_study/{init}_{model}.csv under a tree into an
    attackability-vs-epoch figure. No-op when the study was not run."""
    d = Path(attacks_root) / "epoch_study"
    if not d.is_dir():
        return
    for csvp in sorted(d.glob("*.csv")):
        save_epoch_attackability_plot(csvp, d / (csvp.stem + ".png"))


# --------------------------------------------------------------------------- #
# Artifact I/O — the on-disk contract between the attack driver and this module.
#
# src/attack.py writes the numeric/image artifacts of a run with the write_*
# helpers below; the read_* helpers and render_tree load them back and drive the
# save_* figure functions above. Nothing here touches torch, a model or the
# radon operator, so a whole attack tree can be re-plotted (e.g. after tweaking a
# figure) without re-running the attack.
# --------------------------------------------------------------------------- #










def _render_transfer(attack_dir: Path, meta: Dict, data: Dict[str, np.ndarray]) -> None:
    names = meta["model_names"]
    attack_name = meta["attack_name"]
    T = int(meta["T"])
    n_ex = int(meta["n_ex"])
    gt = data["gt"]
    gt_imgs_T = [gt[k] for k in range(T)]
    for source in names:
        recon_by_model = {m: [data[f"pred__{source}__{m}"][k] for k in range(T)]
                          for m in names}
        save_transfer_figure(attack_dir / f"transfer_from_{source}.png",
                             source, names, recon_by_model, gt_imgs_T, attack_name)
        preds_np = {(source, m): [data[f"pred__{source}__{m}"][k] for k in range(n_ex)]
                    for m in names}
        save_cross_model_examples(
            attack_dir / source, source, names,
            [gt[k] for k in range(n_ex)],
            [data[f"clean__{source}"][k] for k in range(n_ex)],
            preds_np, n_ex, attack_name)


def count_render_steps(attacks_root) -> int:
    """How many progress steps render_tree will emit for this tree.

    One per (init, attack) pair plus the two tree-level steps, so the progress
    lines can carry a denominator from the very first tick."""
    root = Path(attacks_root)
    inits = sorted(p for p in root.glob("init_*") if p.is_dir()) or [root]
    n = 0
    for init_dir in inits:
        if not init_dir.is_dir():
            continue
        n += sum(1 for p in init_dir.iterdir()
                 if p.is_dir() and (p / "summary.json").exists())
    return n + 2  # + attack overview + epoch study


def render_init(init_dir: Path, on_step=None) -> None:
    """Regenerate every figure for one init_<init> folder from its artifacts.

    ``on_step`` is called with a short label before each attack directory, so a
    caller can drive a progress display."""
    init_dir = Path(init_dir)
    name = init_dir.name
    set_plot_init_label(name[len("init_"):] if name.startswith("init_") else name)

    all_rows: Dict[str, Dict[str, List[Dict]]] = {}
    eps_seen: Optional[float] = None
    attack_dirs = sorted(p for p in init_dir.iterdir()
                         if p.is_dir() and (p / "summary.json").exists())
    for attack_dir in attack_dirs:
        if on_step is not None:
            on_step(f"{init_dir.name}/{attack_dir.name}")
        meta = json.loads((attack_dir / "summary.json").read_text(encoding="utf-8"))
        attack_name = meta.get("attack", attack_dir.name)
        eps = meta.get("eps")
        eps_seen = eps if eps is not None else eps_seen
        model_order = list(meta.get("models", {}).keys())

        rows_by_model: Dict[str, List[Dict]] = {}
        for model in model_order:
            model_dir = attack_dir / model
            csv_path = model_dir / "per_sample_metrics.csv"
            if csv_path.exists():
                rows_by_model[model] = read_metric_rows(csv_path)
            ex_json = model_dir / "examples.json"
            if ex_json.exists():
                ex_rows = read_rows_bundle(model_dir / "examples.npz", ex_json)
                save_examples(model_dir, ex_rows)
            worst_json = model_dir / "worst.json"
            if worst_json.exists():
                worst_rows = read_rows_bundle(model_dir / "worst.npz", worst_json)
                worst_out = model_dir / "worst"
                worst_out.mkdir(exist_ok=True)
                save_examples(worst_out, worst_rows)

        if rows_by_model:
            save_suite_scatter(attack_dir, rows_by_model, attack_name, eps)
            save_totals_decomposition_bar(attack_dir, rows_by_model, eps, attack_name)
            save_decomposition_bar(attack_dir, rows_by_model)
            save_null_growth_headline(attack_dir, rows_by_model, eps, attack_name)
            save_consistency_plot(attack_dir, rows_by_model, eps)
            save_ghost_structure_plot(attack_dir, rows_by_model, eps, attack_name)
            all_rows[attack_name] = rows_by_model

        transfer_json = attack_dir / "transfer.json"
        if transfer_json.exists():
            t_meta, t_data = read_transfer_bundle(attack_dir / "transfer.npz", transfer_json)
            _render_transfer(attack_dir, t_meta, t_data)

    if all_rows and eps_seen is not None:
        save_attack_comparison_scatter(init_dir, all_rows, eps_seen)
        save_consistency_overview(init_dir, all_rows, eps_seen)

    lip_json = init_dir / "lipschitz_nullspace.json"
    if lip_json.exists():
        save_lipschitz_plot(init_dir, json.loads(lip_json.read_text(encoding="utf-8")))

    print(f"[visualise] rendered figures for {init_dir}")


def render_tree(attacks_root) -> None:
    """Render every init_<init> folder under an attacks_n<noise> tree. Also
    accepts a path to a single init_<init> folder.

    Each step prints a progress line

        [visualise][progress] 3/10 init_pinv/adversarial_null

    Rendering one run directory takes 35+ minutes, so without these the Slurm
    .out log for a render task shows nothing at all until it finishes."""
    root = Path(attacks_root)
    inits = sorted(p for p in root.glob("init_*") if p.is_dir())
    if not inits:
        inits = [root]

    total = count_render_steps(root)
    state = {"i": 0}

    def step(label: str) -> None:
        state["i"] += 1
        print(f"[visualise][progress] {state['i']}/{total} {label}", flush=True)

    for init_dir in inits:
        render_init(init_dir, on_step=step)
    # Tree-level overview: montage of every executed attack + overview.md index.
    step("attack overview")
    save_attack_overview(root)
    # Epoch-attack study curves, when attack.py --epoch-study was run.
    step("epoch study")
    save_epoch_study_plots(root)
    print(f"[visualise] done -> {root}")
