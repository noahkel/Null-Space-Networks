"""The truncation study of a run: which tau to use, and how much it matters.

Every per-channel number of a run -- the range floor, the null-space error, the
null-restricted attack -- is stated on the numerical null space at the run's
truncation tau: everything outside the span of the right singular vectors with
s_i >= tau*s_max. This study looks at that choice on the run's own geometry and
phantoms:

  which tau is optimal   per noise level, the tau at which the pseudoinverse
                         reconstruction A^+ y^delta has the smallest mean
                         relative error. This is the oracle choice for a
                         truncated SVD, with the noise drawn the way
                         create_phantom_data draws it at that tau: through that
                         tau's own range projector, rescaled to sigma*||y||.
  how much tau matters   how far dim N(A) moves with tau, and how much of the
                         reference null-space error another tau would count as
                         measured instead.

It names candidates for a second run: the optimum at the run's noise level, and
the taus closest to halving and to doubling dim N(A). It also restates the
noise-scheme number of the thesis, the share of a noise draw in range(A) that
survives into range(U_k).

Everything comes from one decomposition, the operator at tau = FULL_TAU. The
truncations are nested, since a larger tau keeps a prefix of the singular
directions. With the coefficients c_i = v_i^T x of a phantom and a_i = u_i^T g
of a noise draw, the pseudoinverse keeping k directions has, exactly and
orthogonally,

    ||e_N||^2 = sum_{i>k} c_i^2   (+ any part of x outside range(A))
    ||e_R||^2 = (sigma ||y||)^2 * sum_{i<=k} a_i^2 / s_i^2 / sum_{i<=k} a_i^2,

so cumulative sums over i give every tau at once. One sample is recomputed with
the pipeline's own projectors as a check on these formulas.

The phantoms are the first --n_samples of the run's training stream, so the
choice of tau never sees the test set.

Run as a stage of slurm_full_run.sh, or by hand from the repository root:

    python -m src.truncation --svd_thresh 4e-3 --run_noise 0.01 --out attacks_n0.01_l2/truncation
"""
import argparse
import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.radon import MatrixRadonAdapter

# The decomposition every tau is cut from: all of range(A).
FULL_TAU = 1e-15
# Rows of the printed table: factor-of-two steps around the operating point, a
# decade either way. The reference, the optimum and the candidates are added.
REPORT_TAUS = (1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
NOISE_LEVELS = (0.005, 0.01, 0.02, 0.05)
# A second run tests the dependence on tau only if its tau moves dim N(A) by at
# least this fraction.
MIN_DIM_CHANGE = 0.2


def nice_taus(lo: float = 1e-5, hi: float = 1e-1) -> List[float]:
    """Every two-significant-digit value from lo to hi: 1.0e-5, 1.1e-5, ...

    The optimum and the candidates are picked from these, so a chosen tau reads
    well in a table and as a directory name."""
    taus = {float(f"{m / 10:.1f}e{e}")
            for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
            for m in range(10, 100)}
    return sorted(t for t in taus if lo <= t <= hi)


def k_for_tau(s: Sequence[float], tau: float) -> int:
    """Directions an adapter at ``tau`` retains: the rule of
    MatrixRadonAdapter._truncated_svd, applied to the singular values as the
    adapter stores them."""
    s = np.asarray(s, dtype=np.float64)
    return int((s >= tau * s[0]).sum())


@contextmanager
def truncated_to(radon: MatrixRadonAdapter, k: int, tau: Optional[float] = None):
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


def _stats(values) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


# --------------------------------------------------------------------------- #
# The error formulas.
# --------------------------------------------------------------------------- #
def coefficients(radon: MatrixRadonAdapter, phantoms: torch.Tensor,
                 noise: torch.Tensor) -> Dict[str, np.ndarray]:
    """What the error formulas need, for phantoms (S, 1, H, W) and noise draws on
    the full sinogram (S, 1, angles, detectors), accumulated in float64:

      c2      (S, K)  c_i^2 = (v_i^T x)^2
      a2      (S, K)  a_i^2 = (u_i^T g)^2, g read on the measured rows as proj_ran reads it
      a2s2    (S, K)  a_i^2 / s_i^2
      x2, y2  (S,)    ||x||^2 and ||y||^2, with y = A x as forward_la computes it
    """
    n_s = phantoms.shape[0]
    x = phantoms.to(device=radon.device, dtype=radon.dtype)
    g = noise[..., radon._la_mask(), :].reshape(n_s, -1).to(device=radon.device,
                                                              dtype=radon.dtype)
    s = radon._s_k_la.to(torch.float64)
    c = radon._mm64(radon._Vt_k_la, x.reshape(n_s, -1).t().double()).t()
    a = radon._mm64_t(radon._U_k_la, g.t().double()).t()
    y = radon.forward_la(x).reshape(n_s, -1).double()

    def cpu(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    return {"c2": cpu(c ** 2), "a2": cpu(a ** 2), "a2s2": cpu((a / s) ** 2),
            "x2": cpu((x.reshape(n_s, -1).double() ** 2).sum(1)), "y2": cpu((y ** 2).sum(1))}


class Errors:
    """Per-sample squared errors of the pseudoinverse at every truncation k.

    Arrays are indexed by the number of retained directions, k = 0..K."""

    def __init__(self, coef: Dict[str, np.ndarray], n: int, dx: float = 1.0):
        c2 = coef["c2"]
        zero = np.zeros((c2.shape[0], 1))
        # x outside range(A): what no truncation reaches
        self.beyond = np.maximum(coef["x2"] - c2.sum(1), 0.0)
        # tail[:, k] = energy of x outside the first k directions, summed from
        # the small end so a short tail is not the difference of two large sums
        self.tail = (np.concatenate([np.cumsum(c2[:, ::-1], 1)[:, ::-1], zero], 1)
                     + self.beyond[:, None])
        self.C2 = np.concatenate([zero, np.cumsum(coef["a2"], 1)], 1)
        self.C3 = np.concatenate([zero, np.cumsum(coef["a2s2"], 1)], 1)
        self.c2 = c2
        self.y2 = coef["y2"] / dx ** 2              # backward_la divides by dx
        # the denominator of rel_l2_np, floored for near-empty phantoms
        self.x_norm = np.maximum(np.sqrt(coef["x2"]), 1e-3 * math.sqrt(n))

    def null2(self, ks: np.ndarray) -> np.ndarray:
        """||e_N||^2, (S, len(ks)): the phantom's energy the truncation discards."""
        return self.tail[:, ks]

    def range2(self, ks: np.ndarray, sigma: float) -> np.ndarray:
        """||e_R||^2, (S, len(ks)): the noise, rescaled to sigma*||y|| inside the
        retained range, after the pseudoinverse."""
        return ((sigma ** 2 * self.y2)[:, None] * self.C3[:, ks]
                / np.maximum(self.C2[:, ks], 1e-300))

    def at(self, k: int, sigma: float) -> Dict[str, Dict[str, float]]:
        """Relative errors of the pseudoinverse keeping k directions, over samples."""
        ks = np.array([k])
        e_n = np.sqrt(self.null2(ks)[:, 0]) / self.x_norm
        e_r = np.sqrt(self.range2(ks, sigma)[:, 0]) / self.x_norm
        total = np.hypot(e_n, e_r)
        return {"rel_l2": _stats(total), "e_null": _stats(e_n), "e_range": _stats(e_r),
                "null_frac": _stats(e_n / np.maximum(total, 1e-300))}

    def reclassified(self, k_ref: int, k: int, sigma: float) -> Dict[str, float]:
        """The reference null-space error, seen from truncation k (means).

        e_ref is the pseudoinverse error of a run at the reference, and its null
        part has energy null_ref. Cut at k instead, a larger tau (k < k_ref) adds
        the noise e_ref carries between the two cuts, and a smaller tau
        (k > k_ref) counts the phantom's coefficients there as measured.

          null_ratio  ||e_N(k)|| / ||e_N(ref)||
          moved       energy crossing the boundary / the larger null energy, in [0, 1]
          beyond      share of ||e_N(ref)||^2 outside range(A), what no tau reaches
        """
        null_ref = self.tail[:, k_ref]
        if k >= k_ref:
            shell = self.c2[:, k_ref:k].sum(1)
            null_k = np.maximum(null_ref - shell, 0.0)
        else:
            shell = (sigma ** 2 * self.y2 / np.maximum(self.C2[:, k_ref], 1e-300)
                     * (self.C3[:, k_ref] - self.C3[:, k]))
            null_k = null_ref + shell
        ref = np.maximum(null_ref, 1e-300)
        return {"null_ratio": float(np.sqrt(null_k / ref).mean()),
                "moved": float((shell / np.maximum(np.maximum(null_ref, null_k), 1e-300)).mean()),
                "beyond": float((self.beyond / ref).mean())}


def cross_check(radon: MatrixRadonAdapter, phantom: torch.Tensor, noise: torch.Tensor,
                k: int, sigma: float, null2: float, range2: float) -> float:
    """Recompute one sample's pseudoinverse error the way create_phantom_data and
    decompose_error do, at truncation k, and return its relative difference to
    the formulas' (null2, range2)."""
    with truncated_to(radon, k) as r:
        x = phantom.to(device=r.device, dtype=r.dtype)
        y = r.forward_la(x)
        eta = r.proj_ran(noise.to(device=r.device, dtype=r.dtype))
        eta = sigma * (torch.linalg.norm(y) / torch.linalg.norm(eta)) * eta
        e_ran, e_nul = r.decompose_error(r.backward_la(y + eta) - x)
    direct = np.array([float((e_nul.double() ** 2).sum()), float((e_ran.double() ** 2).sum())])
    formula = np.array([null2, range2])
    return float(np.abs(direct - formula).sum() / max(formula.sum(), 1e-300))


# --------------------------------------------------------------------------- #
# The study.
# --------------------------------------------------------------------------- #
def run_study(radon: MatrixRadonAdapter, phantoms: torch.Tensor, noises: Sequence[float],
              tau_ref: float, run_noise: Optional[float] = None,
              report_taus: Sequence[float] = REPORT_TAUS, seed: int = 0) -> Dict:
    """The study on one decomposition (``radon``, built at FULL_TAU) and a batch
    of phantoms (S, 1, H, W). Returns what write_outputs saves; the keys
    starting with an underscore hold the curves for the CSV files."""
    run_noise = float(run_noise if run_noise is not None else noises[0])
    noises = sorted({float(v) for v in noises} | {run_noise})
    s = radon._s_k_la.detach().double().cpu().numpy()
    n_k, n = s.size, radon.resolution ** 2
    k_ref = k_for_tau(s, tau_ref)
    if k_ref == 0:
        raise ValueError(f"tau_ref={tau_ref:g} retains no direction")
    dim_ref = n - k_ref

    gen = torch.Generator(device=radon.device).manual_seed(seed)
    noise = torch.randn(phantoms.shape[0], 1, len(radon.angles), radon.det_count,
                        generator=gen, device=radon.device, dtype=radon.dtype)
    err = Errors(coefficients(radon, phantoms, noise), n, radon.dx)

    grid = np.array(nice_taus())
    k_grid = np.array([k_for_tau(s, t) for t in grid])
    curve = {"tau": grid, "k": k_grid, "dim_null": n - k_grid,
             "e_null": (np.sqrt(err.null2(k_grid)) / err.x_norm[:, None]).mean(0)}
    optimal = {}
    for sig in noises:
        en2, er2 = err.null2(k_grid), err.range2(k_grid, sig)
        rel = (np.sqrt(en2 + er2) / err.x_norm[:, None]).mean(0)
        curve[f"rel_l2_{sig:g}"] = rel
        curve[f"e_range_{sig:g}"] = (np.sqrt(er2) / err.x_norm[:, None]).mean(0)
        j = int(np.argmin(rel))
        k = int(k_grid[j])
        optimal[f"{sig:g}"] = {"tau": float(grid[j]), "k": k, "dim_null": n - k,
                               **err.at(k, sig),
                               "rel_l2_at_ref": err.at(k_ref, sig)["rel_l2"]}

    def closest_dim(target: float) -> float:
        return float(grid[int(np.argmin(np.abs((n - k_grid) - target)))])

    candidates = {}
    for name, tau in (("optimal", optimal[f"{run_noise:g}"]["tau"]),
                      ("half_null", closest_dim(0.5 * dim_ref)),
                      ("double_null", closest_dim(2.0 * dim_ref))):
        k = k_for_tau(s, tau)
        candidates[name] = {"tau": tau, "k": k, "dim_null": n - k,
                            "dim_null_vs_ref": (n - k) / dim_ref, **err.at(k, run_noise)}

    best = candidates["optimal"]
    notes = []
    if best["tau"] in (grid[0], grid[-1]):
        notes.append(f"the optimum lies on the edge of the searched range "
                     f"[{grid[0]:g}, {grid[-1]:g}] and may lie beyond it")
    if abs(best["dim_null_vs_ref"] - 1.0) < MIN_DIM_CHANGE:
        notes.append(f"the optimum moves dim N(A) by only "
                     f"{100 * abs(best['dim_null_vs_ref'] - 1):.0f}%; to test how the "
                     f"result depends on tau, run half_null or double_null")
    note = "; ".join(notes)
    recommended = {"tau": best["tau"], "k": best["k"], "dim_null": best["dim_null"],
                   "reason": f"smallest mean rel-L2 of A^+ y^delta at sigma={run_noise:g}",
                   "note": note}

    table = []
    for tau in sorted(set(report_taus) | {tau_ref} | {c["tau"] for c in candidates.values()}):
        k = k_for_tau(s, tau)
        if k == 0:
            continue
        row = {"tau": tau, "k": k, "dim_null": n - k, "s_min_rel": float(s[k - 1] / s[0]),
               "cond": float(s[0] / s[k - 1]), "per_noise": {}}
        for sig in noises:
            row["per_noise"][f"{sig:g}"] = {**{m: v["mean"] for m, v in err.at(k, sig).items()},
                                            **err.reclassified(k_ref, k, sig)}
        table.append(row)

    surviving = np.sqrt(err.C2[:, k_ref] / np.maximum(err.C2[:, n_k], 1e-300))
    ks = np.array([k_ref])
    check = cross_check(radon, phantoms[:1], noise[:1], k_ref, noises[0],
                        float(err.null2(ks)[0, 0]), float(err.range2(ks, noises[0])[0, 0]))
    if not check < 1e-3:
        raise RuntimeError(f"the error formulas disagree with the pipeline's projectors "
                           f"by {check:.1e}; the study's numbers cannot be trusted")

    return {
        "tau_ref": float(tau_ref), "run_noise": run_noise, "noise": noises,
        "dims": {"n": n, "range_M": int(radon.n_la * radon.det_count), "range_A": n_k,
                 "k_ref": k_ref, "dim_null_ref": dim_ref},
        "sigma_max": float(s[0]),
        "noise_scheme": {"surviving_fraction": _stats(surviving),
                         "predicted": math.sqrt(k_ref / n_k)},
        "optimal": optimal,
        "candidates": candidates,
        "recommended": recommended,
        "table": table,
        "cross_check": {"k": k_ref, "sigma": noises[0], "rel_diff": check},
        "_curve": curve,
        "_spectrum": s / s[0],
    }


def print_report(res: Dict) -> None:
    d, keys = res["dims"], [f"{v:g}" for v in res["noise"]]
    ns = res["noise_scheme"]
    print("\n--- truncation study -------------------------------------------------")
    print(f"  n = {d['n']}, measured rows = {d['range_M']}, rank(A) = {d['range_A']}"
          f" (tau = {FULL_TAU:g}), sigma_max = {res['sigma_max']:.4e}")
    print(f"  reference tau = {res['tau_ref']:g}: k = {d['k_ref']}, dim N(A) = {d['dim_null_ref']}")
    print(f"  noise drawn in range(A) keeps {ns['surviving_fraction']['mean']:.3f}"
          f" +/- {ns['surviving_fraction']['std']:.3f} of its size in range(U_k)"
          f" (predicted {ns['predicted']:.3f})")
    print(f"  error formulas against the pipeline's projectors: "
          f"{res['cross_check']['rel_diff']:.1e} relative")

    print("\n--- pseudoinverse error by tau (means; e_N, e_R relative to ||x||) ----")
    print(f"  {'tau':>8} {'k':>6} {'dim N':>6} {'cond':>6}"
          + "".join(f" | {'sigma=' + key:>20}" for key in keys))
    print(f"  {'':>29}" + "".join(f" | {'rel-L2':>6} {'e_N':>6} {'e_R':>6}" for _ in keys))
    for row in res["table"]:
        cells = "".join(f" | {row['per_noise'][key]['rel_l2']:6.3f} {row['per_noise'][key]['e_null']:6.3f}"
                        f" {row['per_noise'][key]['e_range']:6.3f}" for key in keys)
        mark = "  <- reference" if math.isclose(row["tau"], res["tau_ref"]) else ""
        print(f"  {row['tau']:>8g} {row['k']:>6} {row['dim_null']:>6} {row['cond']:>6.0f}{cells}{mark}")

    run_key = f"{res['run_noise']:g}"
    print(f"\n--- the reference null-space error from another tau (sigma={run_key}) --")
    print(f"  {'tau':>8} {'||e_N||/ref':>12} {'moved':>7} {'beyond range(A)':>16}")
    for row in res["table"]:
        r = row["per_noise"][run_key]
        print(f"  {row['tau']:>8g} {r['null_ratio']:>12.3f} {r['moved']:>7.3f} {r['beyond']:>16.3f}")

    print("\n--- optimal tau: smallest mean rel-L2 of A^+ y^delta -------------------")
    for key, o in res["optimal"].items():
        print(f"  sigma={key:>6}: tau = {o['tau']:<8g} k = {o['k']:>6}, dim N = {o['dim_null']:>6},"
              f" rel-L2 {o['rel_l2']['mean']:.4f} (reference: {o['rel_l2_at_ref']['mean']:.4f})")

    print(f"\n--- candidates for a second run at sigma={run_key} ----------------------")
    for name, c in res["candidates"].items():
        print(f"  {name:>11}: tau = {c['tau']:<8g} dim N = {c['dim_null']:>6}"
              f" ({c['dim_null_vs_ref']:.2f} x reference), rel-L2 {c['rel_l2']['mean']:.4f}")
    rec = res["recommended"]
    if rec["note"]:
        print(f"  note: {rec['note']}")
    print(f"  sbatch --export=ALL,NOISE={run_key},SVD_THRESH=auto slurm_full_run.sh"
          f"   # runs at tau = {rec['tau']:g}")


def write_outputs(res: Dict, out_dir) -> None:
    """truncation.json (everything but the curves), spectrum.csv (s_i / s_max)
    and curve.csv (mean errors on the grid of nice_taus)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {key: value for key, value in res.items() if not key.startswith("_")}
    (out_dir / "truncation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(out_dir / "spectrum.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "sigma_rel"])
        w.writerows((i + 1, f"{v:.8e}") for i, v in enumerate(res["_spectrum"]))
    curve = res["_curve"]
    cols = list(curve)
    with open(out_dir / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for j in range(len(curve["tau"])):
            w.writerow([f"{curve[c][j]:.8g}" for c in cols])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--min_angle", type=float, default=0)
    p.add_argument("--max_angle", type=float, default=120)
    p.add_argument("--num_thetas", type=int, default=180)
    p.add_argument("--phantom", default="single",
                   help="phantom family, as for create_phantom_data --phantom.")
    p.add_argument("--svd_thresh", type=float, default=4e-3,
                   help="the reference truncation, normally the run's tau.")
    p.add_argument("--noise", type=float, nargs="+", default=list(NOISE_LEVELS))
    p.add_argument("--run_noise", type=float, default=None,
                   help="the run's noise level: the candidates and the recommended "
                        "tau are for this one. Defaults to the first --noise.")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="seed of the noise draws.")
    p.add_argument("--cache_dir", default="radon_cache")
    p.add_argument("--out", default="truncation", help="output directory.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0 < args.svd_thresh <= 1:
        raise SystemExit(f"--svd_thresh must lie in (0, 1], got {args.svd_thresh:g}")
    # dival and odl, only needed to draw the phantoms
    from src.create_phantom_data import phantom_generator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_count = int(np.sqrt(2) * args.img_size) + 1
    angles = np.linspace(0, 180, args.num_thetas, endpoint=False) * np.pi / 180
    phi = (args.min_angle * np.pi / 180, args.max_angle * np.pi / 180)
    radon = MatrixRadonAdapter(
        resolution=args.img_size, angles=angles, det_count=det_count, dx=1.0, phi=phi,
        svd_threshold=FULL_TAU, device=device, dtype=torch.float32, dense=True,
        estimate_norm=False, cache_dir=args.cache_dir)

    gen, phantom_seed = phantom_generator(args.phantom, args.img_size)
    phantoms = torch.stack([torch.from_numpy(np.asarray(next(gen).data, dtype=np.float32))
                            for _ in range(args.n_samples)])[:, None]
    res = run_study(radon, phantoms, args.noise, args.svd_thresh, args.run_noise,
                    seed=args.seed)
    res["config"] = {"img_size": args.img_size, "min_angle": args.min_angle,
                     "max_angle": args.max_angle, "num_thetas": args.num_thetas,
                     "phantom": args.phantom, "phantom_seed": phantom_seed,
                     "n_samples": args.n_samples, "noise_seed": args.seed}
    print_report(res)
    write_outputs(res, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
