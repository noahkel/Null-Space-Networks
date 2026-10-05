#!/usr/bin/env python3
"""The tables and the numbers of the thesis' results section, from the run
directories alone.

Every run of slurm_full_run.sh / slurm_ellipses.sh leaves an attacks_* directory
with per-sample metrics (init_pinv/<attack>/<model>/per_sample_metrics.csv),
lipschitz.json, budget_sweep.csv and, at the reference and optimal tau, the
seed<s>/ suites. This script reads them and writes the LaTeX rows of every
results table in the layout of thesis.tex, plus numbers.txt with the values the
prose cites. No torch, model or operator is needed.

    python make_tables.py --runs 'attacks_*_v2' --out thesis_tables

Statistics, as Section 3.7 of the thesis states them: levels are medians (clean
errors in Tables 1, 8 and 9 are means); a median over samples comes with a 95%
bootstrap percentile interval (10 000 resamples), and a paired difference
counts as distinguishable from sampling error when that interval excludes zero.
"""
import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REF_TAU = 4e-3
INIT = "init_pinv"
ATTACKS = [("clean", None), ("total-error attack", "adversarial"),
           ("null-space attack", "adversarial_null"), ("range attack", "adversarial_range"),
           ("target: zero image", "adversarial_target_zero"),
           ("target: other sample", "adversarial_target_sample"),
           ("random direction", "random_baseline")]
RUN_NAME = re.compile(r"attacks_n(?P<noise>[0-9.]+)(?:_tau(?P<tau>[0-9.e-]+))?"
                      r"(?P<ellipses>_ellipses)?_l2")


# --------------------------------------------------------------------------- #
# Reading a run.
# --------------------------------------------------------------------------- #
class Run:
    def __init__(self, root: Path):
        self.root = Path(root)
        m = RUN_NAME.match(self.root.name)
        if not m:
            raise ValueError(f"not a run directory: {root}")
        self.noise = float(m["noise"])
        self.tau = float(m["tau"]) if m["tau"] else REF_TAU
        self.phantoms = "multi" if m["ellipses"] else "single"
        self._cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    def rows(self, attack: str, model: str, root: Optional[Path] = None) -> Dict[str, np.ndarray]:
        root = Path(root or self.root)
        key = (str(root), attack + "/" + model)
        if key not in self._cache:
            path = root / INIT / attack / model / "per_sample_metrics.csv"
            with open(path, newline="", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            self._cache[key] = {k: np.array([float(r[k]) for r in data]) for k in data[0]}
        return self._cache[key]

    def has(self, attack: str, model: str = "nsn", root: Optional[Path] = None) -> bool:
        return (Path(root or self.root) / INIT / attack / model / "per_sample_metrics.csv").exists()

    def lipschitz(self) -> Dict:
        path = self.root / INIT / "lipschitz.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def seeds(self) -> List[Path]:
        return sorted(p for p in self.root.glob("seed*") if (p / INIT).is_dir())

    def dim_null(self) -> Optional[int]:
        path = self.root / "truncation" / "truncation.json"
        if not path.exists():
            return None
        meta = json.loads(path.read_text(encoding="utf-8"))
        return int(meta["dims"]["dim_null_ref"])

    @property
    def order(self):
        return (self.phantoms != "single", self.noise, self.tau)


def find_runs(pattern: str) -> List[Run]:
    runs = []
    for d in sorted(glob.glob(pattern)):
        try:
            runs.append(Run(Path(d)))
        except ValueError:
            continue
    return sorted(runs, key=lambda r: r.order)


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def rel(x: np.ndarray, gt_norm: np.ndarray) -> np.ndarray:
    """An absolute norm relative to ||x_gt||, with the floor of rel_l2_np."""
    n_pix = 128 * 128
    return x / np.maximum(gt_norm, 1e-3 * math.sqrt(n_pix))


def median_interval(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> Tuple[float, float, float]:
    """Median over samples and its 95% bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(d, size=(n_boot, d.size), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi)


def resolved(lo: float, hi: float) -> bool:
    """Whether a paired difference is told apart from sampling error: its
    bootstrap interval excludes zero."""
    return lo > 0 or hi < 0


def null_growth(run: Run, model: str, root: Optional[Path] = None) -> np.ndarray:
    r = run.rows("adversarial_null", model, root)
    return r["adv_rel_l2_nul"] - r["clean_rel_l2_nul"]


# --------------------------------------------------------------------------- #
# LaTeX helpers.
# --------------------------------------------------------------------------- #
def sci(x: float) -> str:
    """4e-3 -> $4\\cdot10^{-3}$, as the thesis spells thresholds."""
    mant, exp = f"{x:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"${mant}\\cdot10^{{{int(exp)}}}$"


def small(x: float) -> str:
    """1.3e-07 -> $1.3\\cdot10^{-7}$; ordinary values with two decimals."""
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1e-3:
        return f"{x:.2f}"
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\cdot10^{{{int(exp)}}}$"


def signed(x: float) -> str:
    """+0.015; a fourth decimal where three would print a nonzero value as
    +-0.000, so that the side of zero an interval end lies on stays visible."""
    s = f"{x:+.3f}"
    return f"{x:+.4f}" if x != 0 and float(s) == 0 else s


def num(x: float, digits: int = 4) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def block_rows(runs: List[Run], cells, phantoms: bool = True) -> List[str]:
    """One table row per run, the first row of every (phantoms, noise) block
    labelled and the blocks separated by midrules. Without ``phantoms`` the
    phantom column is left out (Table 8)."""
    out, prev = [], None
    for run in runs:
        key = (run.phantoms, run.noise)
        if prev is not None and key != prev:
            out.append(r"\midrule")
        first = key != prev
        label = [run.phantoms if first else "", f"${run.noise:g}$" if first else ""]
        if not phantoms:
            label = label[1:]
        out.append(" & ".join(label + [sci(run.tau)] + list(cells(run))) + r" \\")
        prev = key
    return out


# --------------------------------------------------------------------------- #
# The tables.
# --------------------------------------------------------------------------- #
def table_clean(runs: List[Run]) -> List[str]:
    """Tables 1 and 9: means of the clean reconstruction per method."""
    out = []
    for run in runs:
        r_res, r_nsn = run.rows("adversarial_null", "resnet"), run.rows("adversarial_null", "nsn")
        g = r_nsn["gt_norm"]
        pinv = [r_nsn["clean_init_rel_l2"], r_nsn["clean_init_psnr"], r_nsn["clean_init_ssim"],
                rel(r_nsn["clean_init_e_nul_l2"], g), rel(r_nsn["clean_init_e_ran_l2"], g)]
        out.append(f"% sigma={run.noise:g} tau={run.tau:g} {run.phantoms}")
        for name, vals in (("$\\mathbf{A}^{+}$", pinv),
                           ("Residual U-Net", None), ("Nullspace Network", None)):
            if vals is None:
                r = r_res if name.startswith("Residual") else r_nsn
                vals = [r["clean_rel_l2"], r["clean_psnr"], r["clean_ssim"],
                        r["clean_rel_l2_nul"], r["clean_rel_l2_ran"]]
            means = [float(np.mean(v)) for v in vals]
            out.append(f" & {name} & {means[0]:.4f} & {means[1]:.2f} & {means[2]:.3f} & "
                       f"{means[3]:.4f} & {means[4]:.4f} \\\\")
    return out


def table_attacks(run: Run) -> List[str]:
    """Table 2: medians of total, range and null-space error per attack."""
    out = []
    for label, attack in ATTACKS:
        cells = []
        for model in ("resnet", "nsn"):
            if attack is None:
                r = run.rows("adversarial_null", model)
                keys = ("clean_rel_l2", "clean_rel_l2_ran", "clean_rel_l2_nul")
            elif run.has(attack, model):
                r = run.rows(attack, model)
                keys = ("adv_rel_l2", "adv_rel_l2_ran", "adv_rel_l2_nul")
            else:
                cells += ["--"] * 3
                continue
            cells += [num(float(np.median(r[k]))) for k in keys]
        out.append(f"{label:22s} & " + " & ".join(cells) + r" \\")
    return out


def table_prop(runs: List[Run]) -> List[str]:
    """Table 3: total-error attack on the NSN against the exact worst case."""
    def cells(run):
        r = run.rows("adversarial", "nsn")
        g = r["gt_norm"]
        growth = rel(r["adv_e_ran_l2"] - r["clean_e_ran_l2"], g)
        gstar = rel(r["cert_e_ran_max"] - r["cert_e_ran_clean"], g)
        share = (r["adv_e_ran_l2"] - r["clean_e_ran_l2"]) / np.maximum(
            r["cert_e_ran_max"] - r["cert_e_ran_clean"], 1e-30)
        return [num(np.median(growth), 3), num(np.median(gstar), 3),
                num(np.median(share), 3), num(np.median(r["adv_e_nul_frac"]), 3)]
    return block_rows(runs, cells)


def range_attack_share(run: Run) -> float:
    r = run.rows("adversarial_range", "nsn")
    return float(np.median((r["adv_e_ran_l2"] - r["clean_e_ran_l2"]) / np.maximum(
        r["cert_e_ran_max"] - r["cert_e_ran_clean"], 1e-30)))


def growth_difference(run: Run, root: Optional[Path] = None) -> Tuple[float, float, float]:
    """Paired difference of the null-space growths, residual network minus NSN."""
    return median_interval(null_growth(run, "resnet", root) - null_growth(run, "nsn", root))


def table_robustness(runs: List[Run]) -> List[str]:
    """Table 4: the null-space attack, levels and the paired difference."""
    def cells(run):
        out = []
        for model in ("resnet", "nsn"):
            r = run.rows("adversarial_null", model)
            out += [num(np.median(r["clean_rel_l2_nul"])), num(np.median(r["adv_rel_l2_nul"]))]
        med, lo, hi = growth_difference(run)
        out.append(f"${signed(med)}\\;[{signed(lo)},{signed(hi)}]$")
        return out
    return block_rows(runs, cells)


def table_targeted(runs: List[Run]) -> List[str]:
    """Table 5: distance to the target and the number of successes."""
    out = []
    for attack, label in (("adversarial_target_zero", "zero image"),
                          ("adversarial_target_sample", "other sample")):
        for i, run in enumerate(runs):
            cells = []
            for model in ("resnet", "nsn"):
                r = run.rows(attack, model)
                cells += [num(np.median(r["tgt_dist_clean"]), 3), num(np.median(r["tgt_dist_adv"]), 3),
                          f"{int(r['tgt_success'].sum())}/{r['tgt_success'].size}"]
            out.append(f"{label if i == 0 else '':14s} & ${run.noise:g}$ & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    return out[:-1]


def gain(lip: Dict, model: str, restriction: str, stat: str = "median") -> float:
    entry = lip.get(model, {}).get(restriction, {})
    values = entry.get("values") or []
    if not values:
        return float("nan")
    return float(np.median(values) if stat == "median" else np.max(values))


def table_lipschitz(run: Run) -> List[str]:
    """Table 6: median and maximum of every gain."""
    lip = run.lipschitz()
    out = []
    for model, label in (("resnet", "residual"), ("nsn", "Nullspace")):
        cells = []
        for restriction in ("null", "range", "full", "cross", "attack"):
            for stat in ("median", "max"):
                cells.append(small(gain(lip, model, restriction, stat)))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


def table_gains(runs: List[Run]) -> List[str]:
    """Table 7: median gains next to the outcome of the null-space attack."""
    def cells(run):
        lip = run.lipschitz()
        out = [num(gain(lip, m, r), 2) for m in ("resnet", "nsn") for r in ("null", "cross", "attack")]
        d = null_growth(run, "resnet") - null_growth(run, "nsn")
        rel_diff = 100 * np.median(d) / max(np.median(null_growth(run, "resnet")), 1e-30)
        out.append(f"${rel_diff:+.0f}\\%$")
        return out
    return block_rows(runs, cells)


def table_clean_tau(runs: List[Run]) -> List[str]:
    """Table 8: clean errors of all single-ellipse runs."""
    def cells(run):
        r_res, r_nsn = run.rows("adversarial_null", "resnet"), run.rows("adversarial_null", "nsn")
        dim = run.dim_null()
        return ([f"{dim:,}".replace(",", "\\,") if dim else "--",
                 num(np.mean(r_nsn["clean_init_rel_l2"])), num(np.mean(r_res["clean_rel_l2"])),
                 num(np.mean(r_nsn["clean_rel_l2"])), num(np.mean(r_res["clean_rel_l2_nul"])),
                 num(np.mean(r_nsn["clean_rel_l2_nul"]))])
    return block_rows(runs, cells, phantoms=False)


# --------------------------------------------------------------------------- #
# Numbers for the prose.
# --------------------------------------------------------------------------- #
def numbers(runs: List[Run]) -> List[str]:
    out = []
    for run in runs:
        out.append(f"== {run.root.name} (sigma={run.noise:g}, tau={run.tau:g}, {run.phantoms})")
        r_nsn = run.rows("adversarial", "nsn")
        g = r_nsn["gt_norm"]
        growth = rel(r_nsn["adv_e_ran_l2"] - r_nsn["clean_e_ran_l2"], g)
        out.append(f"  total-error attack on NSN: median range growth / sigma = "
                   f"{np.median(growth) / run.noise:.1f}; share of G*: total-error "
                   f"{np.median((r_nsn['adv_e_ran_l2'] - r_nsn['clean_e_ran_l2']) / np.maximum(r_nsn['cert_e_ran_max'] - r_nsn['cert_e_ran_clean'], 1e-30)):.3f}, "
                   f"range attack {range_attack_share(run):.3f}")
        for model in ("resnet", "nsn"):
            r = run.rows("adversarial_null", model)
            out.append(f"  {model}: null-space error clean median {np.median(r['clean_rel_l2_nul']):.4f}"
                       f" -> attacked {np.median(r['adv_rel_l2_nul']):.4f} "
                       f"({np.median(r['adv_rel_l2_nul']) / max(np.median(r['clean_rel_l2_nul']), 1e-30):.0f}x); "
                       f"residual on discarded readings clean {np.median(r['clean_residual_disc_rel']):.4f}, "
                       f"attacked {np.median(r['adv_residual_disc_rel']):.4f}, "
                       f"ground truth {np.median(r['gt_residual_disc_rel']):.4f}")
            if run.has("random_baseline", model):
                rr = run.rows("random_baseline", model)
                out.append(f"    random direction: null-space error median {np.median(rr['adv_rel_l2_nul']):.4f}")
        levels = {m: median_interval(null_growth(run, m)) for m in ("resnet", "nsn")}
        out.append("  null-space growth median " + ", ".join(
            f"{m} {v:.3f} [{lo:.3f},{hi:.3f}]" for m, (v, lo, hi) in levels.items()))
        less = float(np.mean(null_growth(run, "nsn") < null_growth(run, "resnet")))
        d, lo, hi = growth_difference(run)
        out.append(f"  NSN null-space error grows less on {100 * less:.0f}% of the samples; "
                   f"paired difference {d:+.4f} [{lo:+.4f},{hi:+.4f}]"
                   f"{' (interval excludes zero)' if resolved(lo, hi) else ''}")
        lip = run.lipschitz()
        for model in ("resnet", "nsn"):
            for restriction in ("null", "cross", "attack"):
                entry = lip.get(model, {}).get(restriction, {})
                conv = entry.get("conv") or []
                if conv:
                    out.append(f"  gain {model}/{restriction}: last-step change <= 1e-3 at "
                               f"{sum(c <= 1e-3 for c in conv)}/{len(conv)} points, max {max(conv):.1e}")
        for seed_root in run.seeds():
            med = {m: float(np.median(run.rows("adversarial_null", m, seed_root)["adv_rel_l2_nul"]))
                   for m in ("resnet", "nsn")}
            d, lo, hi = growth_difference(run, seed_root)
            g_res = float(np.median(null_growth(run, "resnet", seed_root)))
            out.append(f"  {seed_root.name}: attacked null-space medians resnet {med['resnet']:.4f}, "
                       f"nsn {med['nsn']:.4f}; paired difference {d:+.4f} [{lo:+.4f},{hi:+.4f}]"
                       f"{' (interval excludes zero)' if resolved(lo, hi) else ''}, "
                       f"{100 * d / max(g_res, 1e-30):+.0f}% of the residual network's growth")
        sweep = run.root / "budget_sweep.csv"
        if sweep.exists():
            with open(sweep, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for kappa in sorted({float(r["budget_factor"]) for r in rows}):
                meds = {m: np.median([float(r["adv_rel_l2_nul"]) for r in rows
                                      if r["model"] == m and float(r["budget_factor"]) == kappa])
                        for m in ("resnet", "nsn")}
                out.append(f"  budget x{kappa:g}: attacked null-space medians "
                           f"resnet {meds['resnet']:.4f}, nsn {meds['nsn']:.4f}")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="attacks_*_v2", help="glob of the run directories")
    p.add_argument("--out", default="thesis_tables", help="output directory")
    args = p.parse_args(argv)
    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run directories match {args.runs!r}")
    single = [r for r in runs if r.phantoms == "single"]
    multi = [r for r in runs if r.phantoms == "multi"]
    ref_single = [r for r in single if math.isclose(r.tau, REF_TAU)]
    ref_001 = [r for r in ref_single if math.isclose(r.noise, 0.01)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "tab_clean.tex": table_clean(ref_single),
        "tab_attacks.tex": table_attacks(ref_001[0]) if ref_001 else [],
        "tab_prop.tex": table_prop(runs),
        "tab_robustness.tex": table_robustness(runs),
        "tab_targeted.tex": table_targeted(ref_single),
        "tab_lipschitz.tex": table_lipschitz(ref_001[0]) if ref_001 else [],
        "tab_gains.tex": table_gains(runs),
        "tab_clean_tau.tex": table_clean_tau(single),
        "tab_clean_ellipses.tex": table_clean(multi),
    }
    for name, rows in tables.items():
        (out / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "numbers.txt").write_text("\n".join(numbers(runs)) + "\n", encoding="utf-8")
    print(f"{len(runs)} runs -> {out}/ ({', '.join(tables)}, numbers.txt)")


if __name__ == "__main__":
    main()
