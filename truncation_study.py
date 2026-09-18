#!/usr/bin/env python3
"""Two one-off studies about the truncation threshold tau. Neither produces a
result of the thesis; both justify a choice made in producing them.

``--mode noise``  Does drawing the simulated noise from an *untruncated*
operator buy anything?

    Data generation used to build two operators: the reconstruction operator
    truncated at ``--svd_thresh`` (tau = 4e-3), and a second, essentially
    untruncated one (tau = 1e-15) whose range projector drew the noise. The
    second operator costs a full extra SVD of the 21840 x 16384 system matrix,
    so it is worth knowing what it buys.

    This compares, on the same phantoms and the same underlying Gaussian draws:

      scheme A (two operators)  eta = P_{1e-15} g,  rescaled to sigma_rel*||y||
      scheme B (one operator)   eta = P_{tau}   g,  rescaled to sigma_rel*||y||

    with the initial reconstruction x_init = A^+_{tau} (y + eta) in both cases.

    The quantity that matters is the *effective* noise the reconstruction
    actually receives. A^+_{tau} annihilates everything outside range(U_k), so
    under scheme A the part of eta that lands in the discarded directions never
    reaches x_init: the nominal sigma_rel then overstates the perturbation the
    network is trained against, while the adversarial budget eps = sigma_rel is
    spent entirely inside range(U_k) and is therefore fully effective. Scheme B
    removes that mismatch.

``--mode tau``  Does the null space the whole thesis is stated on depend on
where tau was put?

    Everything reported per channel -- the range floor, the null-space error,
    the null-restricted attack -- is stated relative to N(A), and N(A) is the
    *numerical* null space at tau. Move tau and the boundary between the two
    channels moves with it. This sweep measures how much.

    The truncations are nested. A^+_tau keeps the singular directions with
    s_i >= tau*s_max, so for tau' > tau the retained set is a prefix of the one
    at tau and hence

        N(tau') = N(tau) (+) span{v_i : tau*s_max <= s_i < tau'*s_max},

    an orthogonal direct sum. The directions between two thresholds are called
    the *shell* below. This is why the sweep slices one decomposition instead of
    building one operator per tau: the shells are then exact rather than
    approximate, and the comparison costs a single SVD.

    Three things are reported for each tau:

      geometry   k, dim N(A), the smallest retained singular value and the
                 realised condition number against the bound 1/tau,
      floor      what the pseudoinverse reconstruction costs there, per noise
                 level, split into its range and null components,
      stability  how much of the null-space error at the reference tau a
                 different tau would reclassify as measured, and -- the number
                 that settles the question -- how much of it lies outside
                 range(A) altogether, where no choice of tau can reach it.

Run it on the cluster (needs astra + a GPU for a comfortable SVD):

    python truncation_study.py --n_samples 64
    python truncation_study.py --mode tau --n_samples 64
    python truncation_study.py --img_size 64 --n_samples 16   # quick smoke test
"""
import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.radon import MatrixRadonAdapter
from src.create_phantom_data import single_ellipse_generator
from src.utils import rel_l2_np, set_seed, to_4d

# Factor-of-two steps around the operating point, a decade either way. Fine
# enough that consecutive shells are thin, coarse enough that eight of them fit
# in one job. The operating value has to be in the grid, and is checked for.
DEFAULT_TAUS = [1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["noise", "tau", "both"], default="both",
                   help="which study to run: the noise-scheme comparison, the "
                        "tau sweep, or both (default).")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--min_angle", type=float, default=0)
    p.add_argument("--max_angle", type=float, default=120)
    p.add_argument("--num_thetas", type=int, default=180)
    p.add_argument("--svd_thresh", type=float, default=4e-3,
                   help="the operating truncation; the reference tau of the sweep.")
    p.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS,
                   help="thresholds to sweep. --svd_thresh is added if missing.")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--noise", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--amp_probes", type=int, default=8,
                   help="random sinogram draws per tau for the measured noise "
                        "amplification of A^+.")
    p.add_argument("--verify_rebuild", type=float, default=None,
                   help="optional tau at which to build a real adapter and check "
                        "that its null-space projector agrees with the sliced "
                        "one. Costs a second SVD; off by default.")
    p.add_argument("--cache_dir", type=str, default="radon_cache")
    p.add_argument("--out", type=str, default="truncation_study.json")
    return p.parse_args()


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max())}


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.reshape(-1)))


# --------------------------------------------------------------------------- #
# Slicing one decomposition down to a smaller k.
# --------------------------------------------------------------------------- #
def k_for_tau(s_full: torch.Tensor, tau: float) -> int:
    """Number of singular values a real adapter at ``tau`` would retain.

    Mirrors MatrixRadonAdapter._truncated_svd exactly, including the float64
    comparison, so that k(tau) here equals the k that building the operator at
    tau would give.
    """
    s = np.asarray(s_full.detach().cpu().numpy(), dtype=np.float64)
    return int((s >= tau * s[0]).sum())


@contextmanager
def truncated_to(radon: MatrixRadonAdapter, k: int, tau: float = None):
    """Temporarily present ``radon`` as the operator truncated to rank ``k``.

    The factors are replaced by prefix slices, so every method of the adapter --
    proj_ran, backward_la, proj_null_image, decompose_error -- runs the same code
    the pipeline runs, at a different truncation. The slices are views; nothing
    is copied. Restored on the way out, including on an exception.
    """
    saved = (radon._U_k_la, radon._s_k_la, radon._Vt_k_la, radon.svd_threshold)
    if not 0 < k <= saved[1].numel():
        raise ValueError(f"k={k} outside 1..{saved[1].numel()}")
    try:
        radon._U_k_la = saved[0][:, :k]
        radon._s_k_la = saved[1][:k]
        radon._Vt_k_la = saved[2][:k, :]
        if tau is not None:
            radon.svd_threshold = float(tau)
        yield radon
    finally:
        (radon._U_k_la, radon._s_k_la,
         radon._Vt_k_la, radon.svd_threshold) = saved


# --------------------------------------------------------------------------- #
# Study 1: which operator draws the noise.
# --------------------------------------------------------------------------- #
def run_noise_scheme_study(args, radon, radon_full, samples, k_trunc, k_full,
                           t_trunc, t_full):
    surviving = []          # ||P_k eta_A|| / ||eta_A||
    results = {str(sig): {"eff_A": [], "eff_B": [], "relA": [], "relB": [],
                          "AB_diff": []} for sig in args.noise}

    for i, smp in enumerate(samples):
        x_gt, y, g = smp["x_gt"], smp["y"], smp["g"]
        x_clean = radon.backward_la(y)             # noise-free initial reconstruction

        eta_A = radon_full.proj_ran(g)             # range(A)
        eta_B = radon.proj_ran(g)                  # range(U_k)

        nA = _norm(eta_A)
        surviving.append(_norm(radon.proj_ran(eta_A)) / nA)

        y_norm = _norm(y)
        gt_np = x_gt.detach().cpu().numpy()
        clean_norm = _norm(x_clean)

        for sig in args.noise:
            r = results[str(sig)]
            recon = {}
            for tag, eta in (("A", eta_A), ("B", eta_B)):
                scaled = sig * (y_norm / _norm(eta)) * eta
                x_init = radon.backward_la(y + scaled)
                recon[tag] = x_init
                # effective relative perturbation of the initial reconstruction
                r[f"eff_{tag}"].append(_norm(x_init - x_clean) / clean_norm)
                r[f"rel{tag}"].append(
                    rel_l2_np(x_init.squeeze().detach().cpu().numpy(), gt_np))
            r["AB_diff"].append(_norm(recon["A"] - recon["B"]) / max(clean_norm, 1e-12))

        if (i + 1) % 16 == 0:
            print(f"  ...{i + 1}/{len(samples)} samples")

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
    out = {"svd_seconds": {"truncated": t_trunc, "untruncated": t_full},
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
    return out


# --------------------------------------------------------------------------- #
# Study 2: where the channel boundary is put.
# --------------------------------------------------------------------------- #
def run_tau_sweep(args, radon_full, samples, taus, tau_ref, n_pixels):
    s_full = radon_full._s_k_la
    Vt_full = radon_full._Vt_k_la                    # (k_full, n)
    k_full = int(s_full.numel())
    s_np = np.asarray(s_full.detach().cpu().numpy(), dtype=np.float64)
    s_max = float(s_np[0])

    ks = {tau: k_for_tau(s_full, tau) for tau in taus}
    k_ref = ks[tau_ref]

    # ---------------------------------------------------------- geometry ----
    print("\n=== tau sweep ======================================================")
    print("--- geometry -------------------------------------------------------")
    print(f"  image space n = {n_pixels}, rank(A_la) = {k_full} "
          f"(at tau=1e-15), sigma_max = {s_max:.4e}")
    print(f"  {'tau':>9} {'k':>7} {'dim N(A)':>9} {'dimN/ref':>9} {'s_min':>11}"
          f" {'s_max/s_min':>12} {'1/tau':>8} {'shell vs ref':>13}")
    geometry = {}
    for tau in taus:
        k = ks[tau]
        s_min = float(s_np[k - 1])
        shell = k - k_ref                    # >0: extra measured directions
        geometry[f"{tau:g}"] = {
            "tau": tau, "k": k, "dim_null": n_pixels - k,
            "dim_null_over_ref": (n_pixels - k) / (n_pixels - k_ref),
            "s_min_retained": s_min, "cond": s_max / s_min, "cond_bound": 1.0 / tau,
            "shell_dim_vs_ref": shell,
        }
        mark = "  <- operating" if tau == tau_ref else ""
        print(f"  {tau:>9g} {k:>7} {n_pixels - k:>9} "
              f"{(n_pixels - k)/(n_pixels - k_ref):>9.3f} {s_min:>11.4e}"
              f" {s_max/s_min:>12.1f} {1/tau:>8.0f} {shell:>+13d}{mark}")

    # Amplification of A^+ on a random measurement direction, per tau. The
    # theoretical bound is 1/s_min; the realised value is what the noise sees.
    print("\n--- noise amplification of A^+ -------------------------------------")
    print(f"  {'tau':>9} {'||A+ P g||/||P g||':>19} {'1/s_min':>11} {'bound 1/(tau*s_max)':>21}")
    probes = [smp["g"] for smp in samples[:max(1, args.amp_probes)]]
    for tau in taus:
        with truncated_to(radon_full, ks[tau], tau) as r:
            vals = []
            for g in probes:
                eta = r.proj_ran(g)
                vals.append(_norm(r.backward_la(eta)) / max(_norm(eta), 1e-12))
        a = _stats(vals)
        s_min = geometry[f"{tau:g}"]["s_min_retained"]
        geometry[f"{tau:g}"]["amplification"] = a
        print(f"  {tau:>9g} {a['mean']:>19.4f} {1/s_min:>11.4f} "
              f"{1/(tau*s_max):>21.4f}")

    # ------------------------------------------------- floor, per tau/noise --
    # For each tau the noise is drawn the way create_phantom_data would draw it
    # at that tau (scheme B: through that tau's own range projector), so each
    # row is the pseudoinverse floor a pipeline run at that tau would start from.
    print("\n--- pseudoinverse floor --------------------------------------------")
    print(f"  {'tau':>9} {'sigma':>7} {'rel L2':>9} {'PSNR':>7} {'||e_N||':>9}"
          f" {'||e_R||':>9} {'null frac':>10}")
    floor = {}
    for tau in taus:
        k = ks[tau]
        floor[f"{tau:g}"] = {}
        with truncated_to(radon_full, k, tau) as r:
            for sig in args.noise:
                rel, psnr, en, er = [], [], [], []
                for smp in samples:
                    y, g, gt_np = smp["y"], smp["g"], smp["gt_np"]
                    gt_norm = smp["gt_norm"]
                    eta = r.proj_ran(g)
                    eta = sig * (smp["y_norm"] / max(_norm(eta), 1e-12)) * eta
                    x_init = r.backward_la(y + eta)
                    e = x_init - smp["x4"]
                    e_ran, e_nul = r.decompose_error(e)
                    rel.append(rel_l2_np(
                        x_init.squeeze().detach().cpu().numpy(), gt_np))
                    mse = float(torch.mean(e ** 2))
                    rng = float(gt_np.max() - gt_np.min())
                    psnr.append(20 * math.log10(max(rng, 1e-12))
                                - 10 * math.log10(max(mse, 1e-20)))
                    en.append(_norm(e_nul) / gt_norm)
                    er.append(_norm(e_ran) / gt_norm)
                nf = [a / max(math.hypot(a, b), 1e-12) for a, b in zip(en, er)]
                rec = {"rel_l2": _stats(rel), "psnr": _stats(psnr),
                       "e_null": _stats(en), "e_range": _stats(er),
                       "null_frac": _stats(nf)}
                floor[f"{tau:g}"][str(sig)] = rec
                print(f"  {tau:>9g} {sig:>7g} {rec['rel_l2']['mean']:>9.4f}"
                      f" {rec['psnr']['mean']:>7.2f} {rec['e_null']['mean']:>9.4f}"
                      f" {rec['e_range']['mean']:>9.4f} {rec['null_frac']['mean']:>10.4f}")

    # --------------------------------------------- reclassification vs ref --
    # How much of the null-space error at the reference tau a different tau
    # would move across the boundary. Nested truncations make this exact: the
    # coefficients of e along the right singular vectors are computed once, and
    # every tau is a different place to cut that coefficient vector.
    print("\n--- what moving tau does to the reference null-space error ----------")
    print(f"  reference tau = {tau_ref:g}, k_ref = {k_ref}, "
          f"dim N = {n_pixels - k_ref}")
    print(f"  {'tau':>9} {'sigma':>7} {'||e_N(tau)||/||e_N(ref)||':>26}"
          f" {'reclassified':>13} {'beyond range(A)':>16}")
    stability = {}
    for sig in args.noise:
        # The error is the one a pipeline run at the reference tau starts from,
        # so it is computed once per sample and then cut at every tau.
        cums, totals = [], []
        for smp in samples:
            with truncated_to(radon_full, k_ref, tau_ref) as r:
                eta = r.proj_ran(smp["g"])
                eta = sig * (smp["y_norm"] / max(_norm(eta), 1e-12)) * eta
                e = r.backward_la(smp["y"] + eta) - smp["x4"]
            e_flat = e.reshape(-1).to(dtype=Vt_full.dtype, device=Vt_full.device)
            c2 = ((Vt_full @ e_flat) ** 2).double()        # (k_full,) coefficients
            # cumulative energy: cum[j] = sum of the first j squared coefficients
            cums.append(torch.cat([torch.zeros(1, dtype=torch.float64,
                                               device=c2.device),
                                   torch.cumsum(c2, dim=0)]).cpu().numpy())
            totals.append(float((e_flat.double() ** 2).sum()))

        for tau in taus:
            k = ks[tau]
            ratios, moved, deep = [], [], []
            for cum, total in zip(cums, totals):
                null_ref = max(total - float(cum[k_ref]), 1e-30)   # ||e_N(ref)||^2
                null_tau = max(total - float(cum[k]), 0.0)
                # energy the move pushes across the boundary, either direction
                lo, hi = (k_ref, k) if k > k_ref else (k, k_ref)
                shell = float(cum[hi] - cum[lo])
                ratios.append(math.sqrt(null_tau / null_ref))
                moved.append(shell / null_ref)
                deep.append(max(total - float(cum[-1]), 0.0) / null_ref)
            rec = {"null_ratio": _stats(ratios), "moved_frac": _stats(moved),
                   "beyond_range_frac": _stats(deep)}
            stability.setdefault(str(sig), {})[f"{tau:g}"] = rec
            mark = "  <- operating" if tau == tau_ref else ""
            print(f"  {tau:>9g} {sig:>7g} {rec['null_ratio']['mean']:>26.4f}"
                  f" {rec['moved_frac']['mean']:>13.4f}"
                  f" {rec['beyond_range_frac']['mean']:>16.4f}{mark}")
        print()

    # ----------------------------------------------------- candidate taus ---
    # A second tau is only informative if it moves the boundary enough to matter.
    # These are the two that come closest to halving and to doubling dim N(A).
    print("--- candidates for a second run ------------------------------------")
    below = [t for t in taus if ks[t] > k_ref]          # smaller tau -> smaller N
    above = [t for t in taus if ks[t] < k_ref]
    picks = {}
    for name, pool, target in (("smaller_null", below, 0.5), ("larger_null", above, 2.0)):
        if not pool:
            continue
        best = min(pool, key=lambda t: abs(
            (n_pixels - ks[t]) / (n_pixels - k_ref) - target))
        picks[name] = best
        print(f"  {name:>13}: tau = {best:g}  ->  dim N(A) = {n_pixels - ks[best]}"
              f"  ({(n_pixels - ks[best])/(n_pixels - k_ref):.2f} x reference)"
              f",  cond <= {1/best:.0f}")
    if picks:
        print("\n  A second full run at one of these, at one noise level, is what")
        print("  tests whether the channel-restricted result depends on tau:")
        for name, t in picks.items():
            print(f"    sbatch --export=ALL,NOISE=0.01,SVD_THRESH={t:g} slurm_full_run.sh")

    return {"tau_ref": tau_ref, "taus": list(taus), "k_full": k_full,
            "n_pixels": n_pixels, "geometry": geometry, "floor": floor,
            "stability": stability, "candidates": picks}


def verify_rebuild(radon_full, tau, common, n_pixels):
    """Build a real adapter at ``tau`` and check the sliced projector agrees.

    The slices and a freshly decomposed operator need not have the same singular
    *vectors* when singular values are close, so what is compared is the
    projector, which is basis independent.
    """
    print(f"\n--- verifying the slice at tau = {tau:g} against a real adapter ----")
    real = MatrixRadonAdapter(svd_threshold=tau, **common)
    k = k_for_tau(radon_full._s_k_la, tau)
    k_real = int(real._s_k_la.numel())
    print(f"  k sliced = {k}, k rebuilt = {k_real}")
    side = math.isqrt(n_pixels)
    v = torch.randn(1, 1, side, side, device=real.device)
    with truncated_to(radon_full, k, tau) as r:
        a = r.proj_null_image(v)
    b = real.proj_null_image(v)
    err = _norm(a - b) / max(_norm(b), 1e-12)
    print(f"  relative difference of the null-space projections: {err:.3e}")
    return {"tau": tau, "k_sliced": k, "k_rebuilt": k_real, "proj_rel_diff": err}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(0)

    taus = sorted(set(list(args.taus) + [args.svd_thresh]))
    tau_ref = args.svd_thresh
    # tau is relative to sigma_max, so tau > 1 retains nothing and tau <= 0 is
    # not a truncation. Caught here rather than after the SVD.
    bad = [t for t in taus if not 0 < t <= 1]
    if bad:
        raise SystemExit(f"tau must lie in (0, 1], got {bad}")

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
    print(f"device: {device}\nmode: {args.mode}\n")

    t_trunc = float("nan")
    radon = None
    if args.mode in ("noise", "both"):
        t0 = time.perf_counter()
        radon = MatrixRadonAdapter(svd_threshold=tau_ref, **common)
        t_trunc = time.perf_counter() - t0

    # The sweep slices this one: every tau is a prefix of its factors.
    t0 = time.perf_counter()
    radon_full = MatrixRadonAdapter(svd_threshold=1e-15, **common)
    t_full = time.perf_counter() - t0

    # ---------------------------------------------------------------- dims --
    n = args.img_size ** 2
    m_meas = radon_full.n_la * det_count            # dim range(M), measured rows
    k_full = int(radon_full._s_k_la.numel())        # dim range(A), tau = 1e-15
    k_trunc = k_for_tau(radon_full._s_k_la, tau_ref)
    s = radon_full._s_k_la[:k_trunc]
    print("--- subspace dimensions -------------------------------------------")
    print(f"  image space                 n        = {n}")
    print(f"  measured rows   range(M)             = {m_meas}")
    print(f"  measurements    range(A)             = {k_full}   (tau=1e-15)")
    print(f"  retained range  range(U_k)           = {k_trunc}   (tau={tau_ref:g})")
    print(f"  null space      dim N(A)  = n - k    = {n - k_trunc}")
    print(f"  sigma_max = {float(s[0]):.4e}   sigma_min(retained) = {float(s[-1]):.4e}"
          f"   ratio = {float(s[0] / s[-1]):.1f}  (bound 1/tau = {1/tau_ref:.0f})")
    if args.mode in ("noise", "both"):
        print(f"\n  SVD build time: tau={tau_ref:g} -> {t_trunc:.1f}s, "
              f"tau=1e-15 -> {t_full:.1f}s"
              f"   (the second operator is the extra cost under scheme A)")
        # Fraction of an isotropic draw in range(A) that survives the truncation.
        # Expected value is k_trunc / k_full for the squared norm.
        print(f"\n  predicted surviving energy fraction k/k_full = "
              f"{k_trunc / k_full:.4f}  -> amplitude {math.sqrt(k_trunc / k_full):.4f}")

    # -------------------------------------------------------------- samples --
    # Drawn once and shared by both studies, so the two are on the same phantoms
    # and the same Gaussian draws.
    from dival.datasets import EllipsesDataset
    gen = single_ellipse_generator(EllipsesDataset(image_size=args.img_size), "train")
    samples = []
    for _ in range(args.n_samples):
        x_gt = torch.from_numpy(next(gen).data).to(device)
        x4 = to_4d(x_gt)
        y = radon_full.forward_la(x4)               # forward does not depend on tau
        samples.append({"x_gt": x_gt, "x4": x4, "y": y, "g": torch.randn_like(y),
                        "gt_np": x_gt.detach().cpu().numpy(),
                        "gt_norm": max(_norm(x4), 1e-12), "y_norm": _norm(y)})

    out = {"dims": {"n": n, "range_M": m_meas, "range_A": k_full,
                    "range_Uk": k_trunc, "null": n - k_trunc},
           "config": {"mode": args.mode, "tau_ref": tau_ref,
                      "n_samples": args.n_samples, "noise": list(args.noise)}}

    if args.mode in ("noise", "both"):
        out["noise_scheme"] = run_noise_scheme_study(
            args, radon, radon_full, samples, k_trunc, k_full, t_trunc, t_full)

    if args.mode in ("tau", "both"):
        out["tau_sweep"] = run_tau_sweep(args, radon_full, samples, taus, tau_ref, n)
        if args.verify_rebuild is not None:
            out["verify_rebuild"] = verify_rebuild(
                radon_full, args.verify_rebuild, common, n)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
