#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from src.radon import AstraRadonAdapter, MatrixRadonAdapter
import matplotlib.pyplot as plt
from src.utils import mse_loss, set_seed, to_4d, build_models
from typing import List

from src.ellipse_dataloader import get_ellipse_dataloader

# Saves an example reconstruction (GT / init / model output) after training.
def save_example_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    title: str,
):
    model.eval()
    x_gt, x_init, y_delta = next(iter(loader))
    x_gt = to_4d(x_gt).to(device)
    x_init = to_4d(x_init).to(device)
    y_delta = to_4d(y_delta).to(device)

    pred = model(x_init, y_delta)

    panels = [("GT", x_gt), ("Init", x_init), ("Model Output", pred)]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4))
    for ax, (name, tensor) in zip(axes, panels):
        im = ax.imshow(tensor[0, 0].detach().cpu().numpy(), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn=mse_loss,
) -> float:
    model.train()
    running, n = 0.0, 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = to_4d(y_delta).to(device)

        pred = model(x_init, y_delta)
        loss = loss_fn(pred, x_gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * x_gt.shape[0]
        n += x_gt.shape[0]
    return running / max(n, 1)

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn=mse_loss,
) -> float:
    model.eval()
    running, n = 0.0, 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = to_4d(y_delta).to(device)

        pred = model(x_init, y_delta)
        loss = loss_fn(pred, x_gt)

        running += float(loss.item()) * x_gt.shape[0]
        n += x_gt.shape[0]

    return running / max(n, 1)

def detect_init_methods(data_dir: Path) -> List[str]:
    """Auto-detect available init-reconstruction folders in a data directory
    produced by create_ellipse_data.py"""
    known = ["fbp", "pinv"]
    return [m for m in known if (data_dir / m).is_dir() and any((data_dir / m).glob("*.npy"))]

def main(example, out_dir, data_dir, models, checkpoint_every_epoch=False):

    #set_seed(42)

    DATA_ROOT = data_dir
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS_TO_TRAIN = models

    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_WORKERS = 2

    # -------------------------
    # Load summary produced by create_ellipse_data.py
    # -------------------------
    summary_path = DATA_ROOT / "summary.json"
    print(f"Loading summary from: {summary_path}")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    print("loaded summary :)")
    IMG_SIZE = int(summary["img_size"])
    NUM_ANGLES = int(summary["num_angles"])
    DET_COUNT = int(summary["det_count"])
    BETA = float(summary["mean_norm_y_minus_y_delta"])
    ANGLES = summary["angles"]
    PHI = summary["phi"]
    MATRIX_MODE = int(summary["matrix_mode"])
    SVD_THRESH = float(summary.get("svd_threshold", 1e-3))
    dx = summary["dx"]

    n_train = 4000
    n_test = 1000

    # -------------------------
    # Auto-detect init methods from the data directory
    # -------------------------
    INIT_METHODS = detect_init_methods(DATA_ROOT)
    if not INIT_METHODS:
        raise FileNotFoundError(
            f"No init-reconstruction folders (fbp, pinv) found in {DATA_ROOT}"
        )
    print(f"Detected init methods: {INIT_METHODS}")

    # -------------------------
    # Build radon geometry
    # -------------------------
    angles = np.asarray(ANGLES)
    phi = tuple(PHI)
    if MATRIX_MODE == 1:
        radon = MatrixRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE,
            svd_threshold=SVD_THRESH,
            cache_dir="radon_cache",
        )
    else:
        radon = AstraRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE
        )
    print(f"Loaded summary from {summary_path}")
    print(f"IMG_SIZE={IMG_SIZE}, NUM_ANGLES={NUM_ANGLES}, DET_COUNT={DET_COUNT}, PHI={PHI}")
    print(f"BETA (mean y_diff_norms) = {BETA:.6e}")

    for init in INIT_METHODS:
        run_dir = OUT_DIR / f"init_{init}"
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "examples").mkdir(parents=True, exist_ok=True)

        if example == 'ellipses':
            train_loader = get_ellipse_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="train",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=True,
                num_workers=NUM_WORKERS,
                device=None
            )

            val_loader = get_ellipse_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="test",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=False,
                num_workers=NUM_WORKERS,
                device=None
            )
        else:
            raise NotImplementedError("Lodopab not implemented yet")

        models = build_models(MODELS_TO_TRAIN, radon=radon)

        for name, model in models.items():
            model = model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            best_val = float("inf")
            best_epoch = 0
            ckpt_path = run_dir / "checkpoints" / f"{name}_best.pt"
            # Per-epoch train/val loss trace. Written next to the checkpoints so the
            # epoch-attack study can overlay attackability on the loss curves and
            # locate where overfitting (val diverging from train) sets in.
            history = []

            for epoch in range(1, EPOCHS + 1):
                tr = train_one_epoch(model, train_loader, optimizer, DEVICE)
                va = eval_one_epoch(model, val_loader, DEVICE)
                print(f"[init={init} | {name}] epoch {epoch:03d}/{EPOCHS} | train={tr:.6f} | val={va:.6f}")
                history.append({"epoch": epoch, "train": tr, "val": va})

                ckpt_blob = {
                    "init": init,
                    "model_name": name,
                    "state_dict": model.state_dict(),
                    "val_loss": va,
                    "epoch": epoch,
                }
                if va < best_val:
                    best_val = va
                    best_epoch = epoch
                    torch.save({**ckpt_blob, "val_loss": best_val}, ckpt_path)
                # Snapshot every epoch's weights so each epoch can be attacked
                # individually (opt-in: these are large). File names sort by epoch.
                if checkpoint_every_epoch:
                    torch.save(ckpt_blob, run_dir / "checkpoints" / f"{name}_epoch{epoch:03d}.pt")

            # Loss history + which epoch was best, consumed by the epoch-attack study.
            with open(run_dir / "checkpoints" / f"{name}_history.json", "w", encoding="utf-8") as f:
                json.dump({"init": init, "model_name": name, "epochs": EPOCHS,
                           "best_epoch": best_epoch, "history": history}, f, indent=2)

            print(f"[init={init} | {name}] best val={best_val:.6f} (epoch {best_epoch}) saved to {ckpt_path}")

            # save example recon output with best weights
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])

            ex_path = run_dir / "examples" / f"{name}_example.png"
            save_example_outputs(
                model=model,
                loader=val_loader,
                device=DEVICE,
                out_path=ex_path,
                title=f"init={init} | model={name} | best_val={best_val:.6f}",
            )
            print(f"[init={init} | {name}] example saved to {ex_path}")

def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

if __name__ == "__main__":
    #Initialization of parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--out_dir", type=str, default='./')
    parser.add_argument("--data_dir", type=str, default='./',
                        help="A single data directory produced by create_ellipse_data.py "
                             "(e.g. ./data/0.01), containing summary.json and the "
                             "gt/sino/init-recon folders.")
    parser.add_argument("--models", type=str, default="resnet,nsn")
    parser.add_argument("--checkpoint-every-epoch", action="store_true",
                        help="Save each epoch's weights ({model}_epoch{NNN}.pt) so the "
                             "epoch-attack study can attack every epoch individually "
                             "(loss history is always written regardless).")

    args = parser.parse_args()
    model_names = parse_list_arg(args.models)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    type = args.type
    print("Successfully parsed args")
    main(example=type, out_dir=out_dir, data_dir=data_dir, models=model_names,
         checkpoint_every_epoch=args.checkpoint_every_epoch)
    print("Finished.")
