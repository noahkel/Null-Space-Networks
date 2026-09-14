#!/usr/bin/env python3
"""Does drawing the simulated noise from an *untruncated* operator buy anything?

Data generation used to build two operators: the reconstruction operator
truncated at ``--svd_thresh`` (tau = 4e-3), and a second, essentially
untruncated one (tau = 1e-15) whose range projector drew the noise. The second
operator costs a full extra SVD of the 21840 x 16384 system matrix, so it is
worth knowing what it buys.

This script answers that empirically on the real geometry. It compares, on the
same phantoms and the same underlying Gaussian draws:

  scheme A (two operators)  eta = P_{1e-15} g,  rescaled to sigma_rel*||y||
  scheme B (one operator)   eta = P_{tau}   g,  rescaled to sigma_rel*||y||

with the initial reconstruction x_init = A^+_{tau} (y + eta) in both cases.

The quantity that matters is the *effective* noise the reconstruction actually
receives. A^+_{tau} annihilates everything outside range(U_k), so under scheme A
the part of eta that lands in the discarded directions never reaches x_init:
the nominal sigma_rel then overstates the perturbation the network is trained
against, while the adversarial budget eps = sigma_rel is spent entirely inside
range(U_k) and is therefore fully effective. Scheme B removes that mismatch.

Run it on the cluster (needs astra + a GPU for a comfortable SVD):

    python truncation_study.py --n_samples 64
    python truncation_study.py --img_size 64 --n_samples 16   # quick smoke test
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.radon import MatrixRadonAdapter
from src.create_phantom_data import single_ellipse_generator
from src.utils import rel_l2_np, set_seed, to_4d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--min_angle", type=float, default=0)
    p.add_argument("--max_angle", type=float, default=120)
    p.add_argument("--num_thetas", type=int, default=180)
    p.add_argument("--svd_thresh", type=float, default=4e-3)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--noise", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--cache_dir", type=str, default="radon_cache")
    p.add_argument("--out", type=str, default="truncation_study.json")
    return p.parse_args()


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(0)

    det_count = int(np.sqrt(2) * args.img_size) + 1
    angles = np.linspace(0, 180, args.num_thetas, endpoint=False) * np.pi / 180
    phi = (args.min_angle * np.pi / 180, args.max_angle * np.pi / 180)
    # Same dtype and layout as the pipeline, so the tau=4e-3 operator is read
    # straight from radon_cache and only the tau=1e-15 one has to be decomposed.
    common = dict(resolution=args.img_size, angles=angles, det_count=det_count,
                  dx=1.0, phi=phi, device=device, cache_dir=args.cache_dir,
                  dtype=torch.float32, dense=True, estimate_norm=False)

    print(f"geometry: {args.img_size}^2 image, {det_count} detectors, "
          f"{args.num_thetas} angles, window {args.min_angle}-{args.max_angle} deg")
    print(f"device: {device}\n")

    t0 = time.perf_counter()
    radon = MatrixRadonAdapter(svd_threshold=args.svd_thresh, **common)
    t_trunc = time.perf_counter() - t0

    t0 = time.perf_counter()
    radon_full = MatrixRadonAdapter(svd_threshold=1e-15, **common)
    t_full = time.perf_counter() - t0

    # ---------------------------------------------------------------- dims --
    n = args.img_size ** 2
    m_meas = radon.n_la * det_count                 # dim range(M), measured rows
    k_trunc = int(radon._s_k_la.numel())            # dim range(U_k), tau = 4e-3
    k_full = int(radon_full._s_k_la.numel())        # dim range(A), tau = 1e-15
    s = radon._s_k_la
    print("--- subspace dimensions -------------------------------------------")
    print(f"  image space                 n        = {n}")
    print(f"  measured rows   range(M)             = {m_meas}")
    print(f"  measurements    range(A)             = {k_full}   (tau=1e-15)")
    print(f"  retained range  range(U_k)           = {k_trunc}   (tau={args.svd_thresh:g})")
    print(f"  null space      dim N(A)  = n - k    = {n - k_trunc}")
    print(f"  sigma_max = {float(s[0]):.4e}   sigma_min(retained) = {float(s[-1]):.4e}"
          f"   ratio = {float(s[0] / s[-1]):.1f}  (bound 1/tau = {1/args.svd_thresh:.0f})")
    print(f"\n  SVD build time: tau={args.svd_thresh:g} -> {t_trunc:.1f}s, "
          f"tau=1e-15 -> {t_full:.1f}s"
          f"   (the second operator is the extra cost under scheme A)")

    # Fraction of an isotropic draw in range(A) that survives the truncation.
    # Expected value is k_trunc / k_full for the squared norm.
    print(f"\n  predicted surviving energy fraction k/k_full = "
          f"{k_trunc / k_full:.4f}  -> amplitude {math.sqrt(k_trunc / k_full):.4f}")

    # -------------------------------------------------------------- samples --
    from dival.datasets import EllipsesDataset
    gen = single_ellipse_generator(EllipsesDataset(image_size=args.img_size), "train")

    surviving = []          # ||P_k eta_A|| / ||eta_A||
    results = {str(sig): {"eff_A": [], "eff_B": [], "relA": [], "relB": [],
                          "AB_diff": []} for sig in args.noise}

    for i in range(args.n_samples):
        x_gt = torch.from_numpy(next(gen).data).to(device)
        x4 = to_4d(x_gt)
        y = radon.forward_la(x4)
        x_clean = radon.backward_la(y)             # noise-free initial reconstruction

        g = torch.randn_like(y)                    # one draw, shared by both schemes
        eta_A = radon_full.proj_ran(g)             # range(A)
        eta_B = radon.proj_ran(g)                  # range(U_k)

        nA = torch.linalg.norm(eta_A.reshape(-1))
        surviving.append(float(torch.linalg.norm(radon.proj_ran(eta_A).reshape(-1)) / nA))

        y_norm = torch.linalg.norm(y.reshape(-1))
        gt_np = x_gt.detach().cpu().numpy()
        clean_norm = float(torch.linalg.norm(x_clean.reshape(-1)))

        for sig in args.noise:
            r = results[str(sig)]
            recon = {}
            for tag, eta in (("A", eta_A), ("B", eta_B)):
                scaled = sig * (y_norm / torch.linalg.norm(eta.reshape(-1))) * eta
                x_init = radon.backward_la(y + scaled)
                recon[tag] = x_init
                # effective relative perturbation of the initial reconstruction
                eff = float(torch.linalg.norm((x_init - x_clean).reshape(-1))) / clean_norm
                r[f"eff_{tag}"].append(eff)
                r[f"rel{tag}"].append(rel_l2_np(x_init.squeeze().detach().cpu().numpy(), gt_np))
            r["AB_diff"].append(
                float(torch.linalg.norm((recon["A"] - recon["B"]).reshape(-1))
                      / max(clean_norm, 1e-12)))

        if (i + 1) % 16 == 0:
            print(f"  ...{i + 1}/{args.n_samples} samples")

    # --------------------------------------------------------------- report --
    surv = _stats(surviving)
    print("\n--- how much of the scheme-A noise reaches the reconstruction ------")
    print(f"  ||P_k eta|| / ||eta|| = {surv['mean']:.4f} +/- {surv['std']:.4f}"
          f"   (predicted {math.sqrt(k_trunc / k_full):.4f})")
    print(f"  => under scheme A the reconstruction receives {100*surv['mean']:.1f}% of the")
    print(f"     nominal perturbation; the remaining {100*(1-surv['mean']):.1f}% is annihilated")
    print("     by A^+ and is also stripped from y_delta by the attack's own projector.")

    print("\n--- effective noise in the initial reconstruction -------------------")
    print(f"  {'sigma_rel':>10} {'eff A':>12} {'eff B':>12} {'B/A':>8}"
          f" {'relL2 A':>10} {'relL2 B':>10}")
    out = {"dims": {"n": n, "range_M": m_meas, "range_A": k_full,
                    "range_Uk": k_trunc, "null": n - k_trunc},
           "svd_seconds": {"truncated": t_trunc, "untruncated": t_full},
           "surviving_fraction": surv, "per_noise": {}}
    for sig in args.noise:
        r = results[str(sig)]
        a, b = _stats(r["eff_A"]), _stats(r["eff_B"])
        ra, rb = _stats(r["relA"]), _stats(r["relB"])
        print(f"  {sig:>10g} {a['mean']:>12.5f} {b['mean']:>12.5f}"
              f" {b['mean']/max(a['mean'],1e-12):>8.3f}"
              f" {ra['mean']:>10.5f} {rb['mean']:>10.5f}")
        out["per_noise"][str(sig)] = {"eff_A": a, "eff_B": b, "rel_l2_A": ra,
                                      "rel_l2_B": rb, "AB_diff": _stats(r["AB_diff"])}

    print("\n--- verdict ---------------------------------------------------------")
    print("  The two schemes differ only in which subspace the noise occupies.")
    print(f"  Scheme A costs one extra SVD ({t_full:.0f}s here, and a second set of")
    print(f"  cached factors) and delivers {100*surv['mean']:.0f}% of its nominal noise to the")
    print("  reconstruction; scheme B delivers 100% and needs one operator.")
    print("  Since the attack budget eps = sigma_rel is spent entirely inside")
    print(f"  range(U_k), scheme A makes the attacker ~{1/surv['mean']:.2f}x stronger than the")
    print("  noise it is supposed to be matched against.")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
