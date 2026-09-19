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
| `slurm_full_run.sh` | the full experiment for one noise level as one Slurm job |
| `truncation_study.py`, `slurm_truncation_study.sh` | two one-off studies about τ: which operator draws the noise (`--mode noise`) and how much the channel split depends on where τ was put (`--mode tau`) |
| `tests/test_nsn.py` | the test suite |

## Running

Everything runs from the repository root, in the `data_prox2` environment.

The whole experiment for one noise level, as one Slurm job (tests, data,
training, attack suite, epoch study, figures; a failing stage aborts):

```bash
sbatch --export=ALL,NOISE=0.01 slurm_full_run.sh
sbatch --export=ALL,NOISE=0.01,CREATE_DATA=0,TRAIN=0 slurm_full_run.sh   # reuse data and models
sbatch --export=ALL,NOISE=0.01,SVD_THRESH=1e-3 slurm_full_run.sh         # a second truncation
```

A second truncation writes to its own data, model and output directories
(`..._tau1e-3`), so it never overwrites the main experiment. τ is baked into the
data, so `CREATE_DATA=0` cannot be combined with a new one.

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
  effective; `truncation_study.py --mode noise` measures this.
- **τ fixes the channel split.** Every per-channel number - the range floor, the
  null-space error, the null-restricted attack - is stated on the numerical null
  space at τ. `truncation_study.py --mode tau` sweeps thresholds on one
  decomposition (the truncations are nested, so each τ is a prefix of the
  factors) and reports how far the boundary moves, how much of the null-space
  error a different τ would reclassify as measured, and how much of it lies
  outside range(A) where no τ reaches it. It also names two candidate
  thresholds for a confirming run.
- **Single precision, dense layout** in every stage. The one exception is the SVD
  itself: it is computed in double precision and only stored in single. In
  single precision, `torch.linalg.svd` on the GPU returned factors that were
  orthonormal only to ~5·10⁻³. The null-space projector built from them leaked
  into the measurements, and the Nullspace Network learned to use the leak.
  Every build and every cache load checks the factors and refuses any defect
  above 10⁻⁴. The geometry cache under `radon_cache/` is keyed on the geometry,
  τ and the dtype, and is shared by all stages.
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
