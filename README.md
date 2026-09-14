# Null-Space-Networks

Adversarial robustness of a residual UNet and a Nullspace Network for
limited-angle CT (120° of a 180° parallel-beam scan, 128×128 images). Both
networks correct the truncated pseudoinverse reconstruction; the Nullspace
Network is constrained to correct only inside the null space of the forward
operator. `thesis.tex` is the write-up; this file is how to run it.

## Layout

| path | what it does |
| --- | --- |
| `src/radon.py` | `MatrixRadonAdapter`: the system matrix (via ASTRA), its truncated SVD, the pseudoinverse and both projectors, with an on-disk cache |
| `src/create_phantom_data.py` | single-ellipse phantoms, noisy sinograms, pseudoinverse initial reconstructions |
| `src/unet.py`, `src/wrappers.py` | the UNet and the two wrappers (`RESNET`, `NSN`) |
| `train.py` | trains both architectures on one noise level |
| `src/attack.py` (entry: `attack.py`) | PGD attack suite, epoch study, metrics, Lipschitz estimate |
| `src/visualisations.py` (entry: `visualise.py`) | every figure, rebuilt from saved artifacts only |
| `pipeline.sh` | the full Slurm pipeline over all noise levels |
| `truncation_study.py`, `slurm_truncation_study.sh` | the one-off study behind using a single truncation throughout |
| `tests/test_nsn.py` | the test suite; also the pre-submit gate |

## Running

Everything runs from the repository root, in the `data_prox2` environment.

The whole experiment, as chained Slurm arrays (one task per noise level):

```bash
bash pipeline.sh --dry-run        # show the plan
bash pipeline.sh                  # submit prep -> {attack, epoch} -> render
bash pipeline.sh --only render    # redraw figures from existing artifacts
CREATE_DATA=0 TRAIN=0 bash pipeline.sh   # reuse data and models
```

The test suite runs before anything is submitted and a failure aborts the
submission; `--skip-tests` overrides that.

The stages by hand, for one noise level:

```bash
python -m src.create_phantom_data --noise 0.01 --out_dir data
python train.py --data_dir data/0.01 --out_dir models/0.01 --checkpoint-every 1
python attack.py --data-root data/0.01 --model-dir models/0.01 --lipschitz
python attack.py --data-root data/0.01 --model-dir models/0.01 --epoch-study --max-samples 32
python visualise.py attacks_n0.01
```

`create_phantom_data` must be run with `-m`, since it imports from `src`.

## Conventions worth knowing

- **One operator, one truncation.** Noise, reconstruction, attack and the
  data-consistency residual all use the operator truncated at τ = 4·10⁻³.
  Drawing the noise from a second, untruncated operator delivered only 82 % of
  the nominal noise to the reconstruction while the attack budget was fully
  effective; `truncation_study.py` measures this.
- **Single precision, dense layout** in every stage. The geometry cache under
  `radon_cache/` is keyed on the geometry, τ and the dtype, and is shared by all
  stages. Entries are published atomically, so concurrent array tasks can build
  the same entry safely.
- **Splits are by index**: samples 0–3499 train, 3500–3999 select the
  checkpoint, 4000–4999 are the test set every reported number comes from.
  The test set is never used for training or for model selection.
- **Seeds are fixed** for data generation, training and the attacks. Training is
  reproducible closely rather than bit for bit, because cuDNN's convolution
  kernels are not deterministic.
- **Attack budget** is ε·‖y‖ per sample with ε = σ_rel by default, the step size
  2.5·ε_i/50, and of two random restarts the better one is kept per sample.

Changing the truncation, the precision or the splits changes the data or the
models, so data generation and training have to be rerun.

## Tests

```bash
python -m pytest -q
```

Tests that need the real operator are skipped where `astra` or `scipy` are
missing.

## Thesis

`thesis.tex` builds with `pdflatex` + `biber`. It needs `references.bib` and
`Parallel-beam-geometry.png`, which are not in the repository.
