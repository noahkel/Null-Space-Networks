"""
Shared dataset-generation pipeline for phantom data (ellipses or rectangles).

For each sample, the pipeline:
  1. draws a random phantom (one ellipse or one rectangle),
  2. simulates the limited-angle sinogram y = A x_gt and adds relative noise,
  3. saves the ground truth, the pinv (truncated-SVD) initialisation and the
     sinogram as .npy files under <out_dir>/<noise>/{gt,pinv,sino}/,
  4. writes a summary.json with the geometry and noise statistics used by
     train.py and attack.py.

Entry points: create_data.py.
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.radon import AstraRadonAdapter, MatrixRadonAdapter
from dival.datasets import EllipsesDataset

from src.utils import ensure_dir, set_seed, to_4d
from odl.phantom import ellipsoid_phantom, cuboid


def single_rectangle_generator(dataset, part='train'):
    """Generator yielding images with exactly one random axis-aligned rectangle."""
    seed = dataset.fixed_seeds.get(part)
    r = np.random.RandomState(seed)

    lo, hi = np.asarray(dataset.space.min_pt), np.asarray(dataset.space.max_pt)
    center, half = (hi + lo) / 2, (hi - lo) / 2
    to_abs = lambda pt: (center + np.asarray(pt) * half).tolist()

    min_area = 0.1
    for i in range(dataset.get_len(part=part)):
        while True:
            a1 = 0.2 * r.exponential(1.0)
            a2 = 0.2 * r.exponential(1.0)
            if a1*a2>=min_area:
                break
        x   = r.uniform(-0.3, 0.3)   # tighter center range
        y   = r.uniform(-0.3, 0.3)
        image = cuboid(dataset.space, to_abs([y -a2/2, x-a1/2]), to_abs([y+a2/2, x+a1/2]))
        yield image


def single_ellipse_generator(dataset, part='train'):
    """Generator yielding images with exactly one random ellipse, centered and contained."""
    seed = dataset.fixed_seeds.get(part)
    r = np.random.RandomState(seed)
    n = dataset.get_len(part=part)
    from itertools import repeat
    it = repeat(None, n) if n is not None else repeat(None)
    for _ in it:
        min_area = 0.1

        while True:
            a1 = 0.2 * r.exponential(1.0)
            a2 = 0.2 * r.exponential(1.0)
            if np.pi * a1 * a2 < min_area:
                continue

            v   = r.uniform(0.3, 1.0)
            x   = r.uniform(-0.3, 0.3)   # tighter center range
            y   = r.uniform(-0.3, 0.3)
            rot = r.uniform(0., 2 * np.pi)

            # max extent of rotated ellipse along each axis
            dx = np.sqrt((a1 * np.cos(rot))**2 + (a2 * np.sin(rot))**2)
            dy = np.sqrt((a1 * np.sin(rot))**2 + (a2 * np.cos(rot))**2)

            if abs(x) + dx <= 1.0 and abs(y) + dy <= 1.0:
                break

        ellipsoids = np.array([[v, a1, a2, x, y, rot]])
        image = ellipsoid_phantom(dataset.space, ellipsoids)
        yield image


GENERATORS = {
    "ellipses": single_ellipse_generator,
    "rectangles": single_rectangle_generator,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--matrix_mode", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--min_angle", type=float, default=0)
    parser.add_argument("--max_angle", type=float, default=60)
    parser.add_argument("--num_thetas", type=int, default=180)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--svd_thresh", type=float, default=4e-3)
    # Small switch selecting which limited-angle system matrix produces the saved
    # pinv init (matrix_mode == 1 only):
    #   "thresholded"   — truncated-SVD operator A_la^+ (svd_threshold=svd_thresh),
    #                     drops small-singular-value directions -> noise-suppressed
    #                     but biased pinv (current default behaviour);
    #   "unthresholded" — full operator A_la^+ (svd_threshold≈0), keeps every
    #                     retained direction -> unbiased but noise-amplifying pinv.
    # Ignored for the astra backend (matrix_mode == 0), which has no SVD truncation.
    parser.add_argument("--pinv_mode", type=str, default="thresholded",
                        choices=["thresholded", "unthresholded"])
    return parser.parse_args()


def main(shape: str) -> None:
    '''Run the full data-generation pipeline for the given phantom shape
    ('ellipses' or 'rectangles').'''
    if shape not in GENERATORS:
        raise ValueError(f"Unknown shape '{shape}'. Use one of: {sorted(GENERATORS)}")
    '''Parse Args and setup variables'''
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    N_SAMPLES = args.n_samples
    IMG_SIZE = args.img_size
    NUM_ANGLES = args.num_thetas
    MIN_ANGLE = args.min_angle
    MAX_ANGLE = args.max_angle
    DET_COUNT = int(np.sqrt(2)*IMG_SIZE) + 1
    MATRIX_MODE = args.matrix_mode
    NOISE_sigma_REL = args.noise
    SVD_THRESH = float(args.svd_thresh)
    PINV_MODE = args.pinv_mode
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)
    print(f"Generating {shape} on device {DEVICE}")
    # output structure
    OUT_DIR = OUT_DIR / str(NOISE_sigma_REL)
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "gt")
    ensure_dir(OUT_DIR / "sino")
    # One init reconstruction per backend, named after what it actually is: the
    # matrix backend has a truncated-SVD pseudoinverse, astra does not and gets
    # the limited-angle FBP instead. train.py and attack.py detect whichever
    # folder is present.
    INIT_NAME = "pinv" if MATRIX_MODE == 1 else "fbp"
    ensure_dir(OUT_DIR / INIT_NAME)

    # dataset
    dataset = EllipsesDataset(image_size=IMG_SIZE)
    gen = GENERATORS[shape](dataset, 'train')

    # radon
    dx = 1.0
    angles = np.linspace(0, 180, NUM_ANGLES, endpoint=False) * np.pi / 180
    phi = (MIN_ANGLE * np.pi / 180, MAX_ANGLE * np.pi / 180)

    if MATRIX_MODE == 0:
        radon = AstraRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            clip_to_circle=False,
            dx=dx,
            phi=phi
        )
        # Astra has no SVD truncation, so the "full" operator is the same one.
        radon_full = radon
    else:
        radon = MatrixRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE,
            cache_dir="radon_cache",
            svd_threshold=SVD_THRESH
        )
        radon_full = MatrixRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE,
            cache_dir="radon_cache",
            svd_threshold=1e-15
        )
    print("Built Radon adapter...")

    # Select the limited-angle operator used for the saved pinv init. Both matrix
    # adapters expose the same backward_la; they differ only in svd_threshold, so
    # the flag simply picks the thresholded (radon) or unthresholded (radon_full)
    # pseudoinverse. Meaningless for astra, which has no SVD to truncate.
    pinv_radon = radon if PINV_MODE == "thresholded" else radon_full
    if MATRIX_MODE == 1:
        print(f"pinv_mode = {PINV_MODE} "
              f"(using {'radon' if pinv_radon is radon else 'radon_full'}.backward_la)")

    L = radon.norm_A2

    y_diff_norms: List[float] = []
    y_norms: List[float] = []

    print("Generating data...")
    print("x_gt from generator")
    print("y from radon_full.forward_la")
    print("y_delta = y with added noise")
    if MATRIX_MODE == 1:
        print(f"x_init from {'radon' if pinv_radon is radon else 'radon_full'}"
              f".backward_la ({PINV_MODE}) -> {INIT_NAME}/")
    else:
        print(f"x_init from radon.fbp_la -> {INIT_NAME}/")

    for i in range(N_SAMPLES):
        x_gt = torch.from_numpy(next(gen).data).to(DEVICE)

        y = radon_full.forward_la(to_4d(x_gt))
        noise = radon_full.proj_ran(torch.randn_like(y))
        add_noise = NOISE_sigma_REL * (torch.linalg.norm(y) / torch.linalg.norm(noise)) * noise
        y_delta = y + add_noise

        y_norms.append(float(torch.linalg.norm(y.reshape(-1))))
        y_diff_norms.append(float(torch.linalg.norm((add_noise).reshape(-1))))

        if MATRIX_MODE == 1:
            x_init = pinv_radon.backward_la(y_delta).squeeze()
        else:
            x_init = radon.fbp_la(y_delta).squeeze()

        np.save(OUT_DIR / "gt" / f"{i:05d}.npy", x_gt.detach().cpu().numpy())
        np.save(OUT_DIR / INIT_NAME / f"{i:05d}.npy", x_init.detach().cpu().numpy())
        np.save(OUT_DIR / "sino" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())

    y_diff_norms = np.array(y_diff_norms)
    np.save(OUT_DIR / "y_diff_norms.npy", y_diff_norms)

    summary = {
        "dataset": shape,
        "part": None,
        "n_samples": N_SAMPLES,
        "img_size": int(IMG_SIZE),
        "num_angles": int(NUM_ANGLES),
        "det_count": int(DET_COUNT),
        "angles": angles.tolist(),
        "dx": float(dx),
        "phi": list(phi),
        "phi_deg": [float(MIN_ANGLE), float(MAX_ANGLE)],
        "device": DEVICE,
        "add_noise": None,
        "noise_sigma_rel": float(NOISE_sigma_REL),
        "mean_norm_y": float(np.array(y_norms).mean()),
        "mean_norm_y_minus_y_delta": float(y_diff_norms.mean()),
        "operator_norm_A2": float(L),
        "matrix_mode": int(MATRIX_MODE),
        "svd_threshold": SVD_THRESH if MATRIX_MODE == 1 else None,
        "pinv_mode": PINV_MODE,
        "init_recon": INIT_NAME,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Data saved to:", OUT_DIR.resolve())
