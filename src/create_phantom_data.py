"""Dataset-generation pipeline for the single-ellipse phantoms.

For each sample, the pipeline:
  1. draws a random phantom (one ellipse),
  2. simulates the limited-angle sinogram y = A_la x_gt and adds relative noise,
  3. saves the ground truth, the pinv (truncated-SVD) initialisation and the
     sinogram as .npy files under <out_dir>/<noise>/{gt,pinv,sino}/,
  4. writes a summary.json with the geometry and noise statistics used by
     train.py and attack.py.

Run it as a module, so the ``src.`` imports resolve:

    python -m src.create_phantom_data --noise 0.01 --out_dir ./data
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.radon import MatrixRadonAdapter
from dival.datasets import EllipsesDataset

from src.utils import ensure_dir, set_seed, to_4d
from odl.phantom import ellipsoid_phantom


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--min_angle", type=float, default=0)
    parser.add_argument("--max_angle", type=float, default=120)
    parser.add_argument("--num_thetas", type=int, default=180)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--svd_thresh", type=float, default=4e-3)
    return parser.parse_args()


def main() -> None:
    '''Run the full data-generation pipeline for the single-ellipse phantoms.'''
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    N_SAMPLES = args.n_samples
    IMG_SIZE = args.img_size
    NUM_ANGLES = args.num_thetas
    MIN_ANGLE = args.min_angle
    MAX_ANGLE = args.max_angle
    DET_COUNT = int(np.sqrt(2)*IMG_SIZE) + 1
    NOISE_sigma_REL = args.noise
    SVD_THRESH = float(args.svd_thresh)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)
    print(f"Generating ellipses on device {DEVICE}")
    # output structure
    OUT_DIR = OUT_DIR / str(NOISE_sigma_REL)
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "gt")
    ensure_dir(OUT_DIR / "sino")
    ensure_dir(OUT_DIR / "pinv")

    # dataset
    dataset = EllipsesDataset(image_size=IMG_SIZE)
    gen = single_ellipse_generator(dataset, 'train')

    # radon
    dx = 1.0
    angles = np.linspace(0, 180, NUM_ANGLES, endpoint=False) * np.pi / 180
    phi = (MIN_ANGLE * np.pi / 180, MAX_ANGLE * np.pi / 180)

    # One operator for everything: forward projection, the range projector the
    # noise is drawn through, and the truncated pseudoinverse that reconstructs.
    # An earlier version drew the noise through a second, essentially
    # untruncated adapter (svd_threshold=1e-15). That cost a full extra SVD and
    # placed part of the noise in directions A^+ annihilates, so the nominal
    # noise level overstated what the reconstruction actually received -- while
    # the adversarial budget, which uses this operator's projector, was fully
    # effective. truncation_study.py quantifies the difference.
    radon = MatrixRadonAdapter(
        resolution=IMG_SIZE,
        angles=angles,
        det_count=DET_COUNT,
        dx=dx,
        phi=phi,
        device=DEVICE,
        dtype=torch.float32,
        dense=True,
        cache_dir="radon_cache",
        svd_threshold=SVD_THRESH
    )
    print("Built Radon adapter...")

    L = radon.norm_A2

    y_diff_norms: List[float] = []
    y_norms: List[float] = []

    print("Generating data...")
    print("x_gt from generator")
    print("y from radon.forward_la")
    print("y_delta = y with added noise, drawn from range(U_k)")
    print("x_init from radon.backward_la (truncated pinv) -> pinv/")

    for i in range(N_SAMPLES):
        x_gt = torch.from_numpy(next(gen).data).to(DEVICE)

        y = radon.forward_la(to_4d(x_gt))
        noise = radon.proj_ran(torch.randn_like(y))
        add_noise = NOISE_sigma_REL * (torch.linalg.norm(y) / torch.linalg.norm(noise)) * noise
        y_delta = y + add_noise

        y_norms.append(float(torch.linalg.norm(y.reshape(-1))))
        y_diff_norms.append(float(torch.linalg.norm((add_noise).reshape(-1))))

        x_init = radon.backward_la(y_delta).squeeze()

        np.save(OUT_DIR / "gt" / f"{i:05d}.npy", x_gt.detach().cpu().numpy())
        np.save(OUT_DIR / "pinv" / f"{i:05d}.npy", x_init.detach().cpu().numpy())
        np.save(OUT_DIR / "sino" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())

    y_diff_norms = np.array(y_diff_norms)
    np.save(OUT_DIR / "y_diff_norms.npy", y_diff_norms)

    summary = {
        "dataset": "ellipses",
        "n_samples": N_SAMPLES,
        "img_size": int(IMG_SIZE),
        "num_angles": int(NUM_ANGLES),
        "det_count": int(DET_COUNT),
        "angles": angles.tolist(),
        "dx": float(dx),
        "phi": list(phi),
        "phi_deg": [float(MIN_ANGLE), float(MAX_ANGLE)],
        "device": DEVICE,
        "noise_sigma_rel": float(NOISE_sigma_REL),
        "mean_norm_y": float(np.array(y_norms).mean()),
        "mean_norm_y_minus_y_delta": float(y_diff_norms.mean()),
        "operator_norm_A2": float(L),
        "svd_threshold": SVD_THRESH,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Data saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
