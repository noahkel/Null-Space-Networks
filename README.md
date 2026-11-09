# Null-Space-Networks

Code for the experiments on the adversarial robustness of a residual U-Net and a
Nullspace Network in limited-angle CT: a 120° window of a 180° parallel-beam
scan, 128×128 images. Both networks correct the truncated-SVD pseudoinverse
reconstruction; the Nullspace Network may only add a correction that lies in the
(numerical) null space of the forward operator. The repository trains both,
attacks them with a PGD suite, estimates their local gains and renders every
figure. The thesis itself lives in a separate repository.

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Outputs](#outputs)
- [Conventions worth knowing](#conventions-worth-knowing)
- [Tests](#tests)

## Quick start

Requirements: Linux, Python 3.11 (other recent versions likely work), a CUDA
GPU for the full experiment. Everything runs from the repository root.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt     # for a CUDA build of torch, install torch from pytorch.org first
python -m pytest -q                 # ~1 min on a CPU, needs no data or GPU
```

To see the whole pipeline work end to end without a GPU, run it on a small
geometry (32×32 images, 45 angles, two epochs) for one noise level. It runs the
four truncations of that noise level one after another, about 25 minutes each
on four CPU cores; the first finished run already has
every stage's output (`attacks_n0.01_l2_smoke/`), so it can be stopped after
that:

```bash
CONDA_ENV= RUN_TAG=_smoke WORK_DIR=/tmp/nsn_smoke RUN_TESTS=0 NOISES=0.01 EXTRA_SEEDS= \
IMG_SIZE=32 NUM_THETAS=45 EPOCHS=2 CHECKPOINT_EVERY=1 \
MAX_SAMPLES=8 SWEEP_MAX=4 EPOCH_STUDY_MAX=4 LIPSCHITZ_SAMPLES=4 TRUNC_SAMPLES=8 \
bash slurm_full_run.sh
```

`CONDA_ENV=` (empty) tells the script to use the Python environment that is
already active instead of loading the cluster's modules and conda environment.
`RUN_TAG` and `WORK_DIR` keep the small data and models apart from those of the
real experiment. Results and figures go to
`attacks_*_smoke/` in the repository. The numbers of such a run mean nothing;
it only checks that every stage runs.

## Repository layout

| path | what it does |
| --- | --- |
| `train.py` | trains both architectures on one data directory |
| `attack.py` → `src/attack.py` | PGD attack suite, budget sweep, epoch study, metrics, Lipschitz estimate |
| `visualise.py` → `src/visualisations.py` | every figure, rebuilt from the saved artifacts alone (no models or operator needed) |
| `src/radon.py` | `MatrixRadonAdapter`: the system matrix (via ASTRA), its truncated SVD, the pseudoinverse and both projectors, with an on-disk cache in `radon_cache/` |
| `src/create_phantom_data.py` | ellipse phantoms (one per image, or DIVAL's multi-ellipse ones), noisy sinograms, pseudoinverse initial reconstructions |
| `src/truncation.py` | the truncation study: the τ per noise level at which the pseudoinverse is best, how much the channel split depends on τ |
| `src/unet.py`, `src/wrappers.py` | the U-Net and the two wrappers (`RESNET`, `NSN`) |
| `src/ellipse_dataloader.py` | the dataset and the fixed train/val/test split |
| `src/artifacts.py`, `src/utils.py` | on-disk result format; seeding, noise, metrics, model construction |
| `run_pipeline.sh` | the stages of one run, shared by both jobs (sourced, not run on its own) |
| `slurm_full_run.sh` | the whole single-ellipse experiment: every noise level at four truncations |
| `slurm_ellipses.sh` | the multi-ellipse phantoms at one noise level |
| `fetch_results.ps1` | (Windows, optional) copies the results of a cluster run to a local machine over ssh |
| `tests/test_nsn.py` | the test suite |

## Reproducing the experiments

### The full experiment

The experiment is two jobs, which can run at the same time. On a Slurm cluster:

```bash
sbatch slurm_full_run.sh     # single ellipses: 4 noise levels x 4 truncations, 16 runs (time limit 96 h)
sbatch slurm_ellipses.sh     # multi-ellipse phantoms at noise 0.01: 2 runs (time limit 24 h)
```

Without Slurm, run the same scripts with `bash` (the `#SBATCH` lines are then
plain comments), e.g. `CONDA_ENV= bash slurm_full_run.sh` inside your own
environment.

Before submitting on another cluster, check these site-specific settings:

| setting | where | default |
| --- | --- | --- |
| partition, time limit, mail | `#SBATCH` lines of both job scripts | `--partition=all`; mail to the submitting user |
| environment modules | `MODULES` | `"anaconda/anaconda3 cuda/12.5"` (skipped where `module` does not exist) |
| conda environment | `CONDA_ENV` | `data_prox2`; empty = use the active environment |
| data / model location | `WORK_DIR`, or `DATA_BASE` and `MODEL_BASE` | the directory containing the repository |

Every setting of `run_pipeline.sh` is an environment variable, so it can be
changed without editing a file, e.g.

```bash
sbatch --export=ALL,NOISES=0.02 slurm_full_run.sh      # one noise level only
sbatch --export=ALL,FORCE_RUN=1 slurm_full_run.sh      # redo finished runs
sbatch --export=ALL,FORCE_TRAIN=1 slurm_full_run.sh    # retrain on the existing data
sbatch --export=ALL,EXTRA_SEEDS= slurm_full_run.sh     # no extra seeds
sbatch --export=ALL,CHECKPOINT_EVERY=1 slurm_full_run.sh  # epoch study at every epoch
sbatch --export=ALL,RUN_TESTS=0 slurm_full_run.sh      # skip the test suite at the start
```

### What a job does

At every noise level `slurm_full_run.sh` runs the reference τ = 4·10⁻³, the τ at
which the truncation study finds the pseudoinverse best (the job runs the study
once before it builds the run matrix, and every run checks it again), and
τ = 4.4·10⁻⁴ and 6.2·10⁻², which halve and double dim N_τ. Each run goes
through the truncation study, data, training, the attack suite with the
Lipschitz estimate, the budget sweep, the epoch study (total and null-space
error) and the figures; a failing stage aborts the job. At the reference and
the optimal τ both models are trained again with the seeds `EXTRA_SEEDS`
(default `1 2`) and attacked by the suite, into `seed<s>/` of the run's model
and output directories.

A run reuses the data when its `summary.json` exists and the models when every
model has its `_history.json` (both are written last), after checking that the
data were made for this noise level, τ and phantom family. Everything after the
training is always recomputed, so every number of a run comes from one version
of the code, and a finished run leaves `<run>/.complete` with the commit. A job
that stopped is simply resubmitted: runs finished at the current commit are
skipped. Runs whose models exist go first, so a problem in the later stages
shows up before a day of training.

All directories of these runs carry the tag `RUN_TAG` (default `_v2`): the data,
models and results of the earlier runs, whose noise was confined to range(U_k),
stay where they are, and the stages refuse data without the new noise model.

A truncation or phantom family other than the reference writes to its own data,
model and output directories (`..._tau0.011`, `..._ellipses`), so no run
overwrites another. The operator is decomposed once per geometry and τ and
cached in `radon_cache/`; a job that needs an entry another job is still
building waits for it.

### The stages by hand

For one noise level at the reference τ (the paths are examples):

```bash
python -m src.truncation --svd_thresh 4e-3 --run_noise 0.01 --out attacks_n0.01/truncation
python -m src.create_phantom_data --noise 0.01 --out_dir data
python train.py --data_dir data/0.01 --out_dir models/0.01 --checkpoint-every 1
python attack.py --data-root data/0.01 --model-dir models/0.01 --lipschitz
python attack.py --data-root data/0.01 --model-dir models/0.01 --budget-sweep 0.25,0.5,1,2,4 --max-samples 64
python attack.py --data-root data/0.01 --model-dir models/0.01 --epoch-study --max-samples 32
python attack.py --data-root data/0.01 --model-dir models/0.01 --epoch-study --epoch-objective null --max-samples 32
python visualise.py attacks_n0.01
```

`src.truncation` and `src.create_phantom_data` must be run with `-m`, since they
import from `src`. Every script lists its options with `--help`. The data
generation needs at least 5000 samples (`--n_samples`, the default), since the
split below is by index.

## Outputs

| location | contents |
| --- | --- |
| `<DATA_BASE>.../<noise>/` | `gt/`, `sino/`, `pinv/` (one `.npy` per sample), `y_diff_norms.npy` and `summary.json` with the geometry and noise statistics |
| `<MODEL_BASE>.../<noise>/init_pinv/checkpoints/` | `{model}_best.pt` (lowest validation loss), `{model}_epochNNN.pt` snapshots, `{model}_history.json` loss curves |
| `attacks_n<noise>[_tau<τ>][_ellipses]_l2_v2/` | one run's results: per-attack metrics (`.csv`/`.npz`/`.json`), `truncation/`, `epoch_study/`, `budget_sweep.csv`, `seed<s>/` and every figure |
| `radon_cache/` | the decomposed operators, keyed on geometry, τ and dtype |
| `logs/` | the Slurm logs; each starts with the commit and the versions of the numerical packages used |

## Conventions worth knowing

- **Noise on every measured reading.** The noise is a standard normal draw on
  all m measured readings, rescaled to σ‖y‖ per sample, the same draw at every τ
  (and, rescaled, at every σ). The truncated pseudoinverse receives P_k η, about
  √(k/m) of it; the truncation study reports this share per τ. An earlier version
  drew the noise inside range(U_k): the physical noise level then depended on τ,
  and the readings the truncation discards were noise-free.
- **τ fixes the channel split.** Every per-channel number - the range floor, the
  null-space error, the null-restricted attack - is stated on the numerical null
  space at τ. `src/truncation.py` evaluates every τ on one decomposition (the
  truncations are nested, so each τ is a prefix of the factors, and the
  pseudoinverse error at every τ follows from cumulative sums of the
  coefficients). It reports the τ that minimises the pseudoinverse error per
  noise level - the pseudoinverse alone, since no network is trained per τ - how far the boundary moves, how much of the null-space error a
  different τ would reclassify as measured, and how much of it lies outside
  range(A) where no τ reaches it. It also names candidate thresholds for a
  second run.
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
- **Seeds are fixed** for data generation, training and the attacks. The
  phantoms come from DIVAL's fixed seeds (without them DIVAL draws a new seed
  every run), the noise from `set_seed(0)`. Every attack, every Lipschitz
  restriction and every epoch-study snapshot reseeds from its own name
  (`stage_seed`), so a stage's numbers do not depend on which stages ran before
  it, both models start from the same random points, and along an epoch-study
  curve only the weights change. Training and the attacks are reproducible
  closely rather than bit for bit, because cuDNN's convolution kernels, which
  the attacks backpropagate through as well, are not deterministic. `train.py`
  builds and trains the two models one after the other from one seed, so they
  start from different weights and see the batches in different orders.
- **Attack budget** is κ‖P_k η_i‖ per sample, the noise the reconstruction
  receives, with κ = 1 by default (`--budget-factor`), for the attack suite and
  the epoch study alike; the step size is 2.5·ε_i/50. Every PGD iterate is
  scored, and each sample keeps the best one over all steps and both restarts.
  For the Nullspace Network the largest range error any perturbation in the
  budget can cause is computed exactly (a trust-region subproblem), so the
  `cert_e_ran_max` column says how much the attacks leave on the table. A
  `random_baseline` "attack" spends the full budget in a random direction.
- **Residual on the discarded readings.** Besides the residual against the
  retained measurements, every row records `*_residual_disc_rel`, the residual
  on the readings the truncation discards, where a null-space error e_N appears
  as A e_N.
- **Local gains** are estimated by power iteration for five restrictions; `attack`
  is ‖P_N J_g A^+‖ on range(U_k) times ‖P_k y^δ‖/‖x‖, the gain a null-space
  attack sees to first order. Every estimate is a lower bound; `conv` records
  the relative change of its last iteration.
- **Targeted attacks are scored by the distance to their target** t, relative to
  ‖x_gt − t‖: 1 at the ground truth, 0 at the target (`tgt_*` columns), split
  into range and null-space part, plus the share of samples that end closer to
  the target than to the ground truth. The attack towards another sample draws
  its targets once, so both models are attacked towards the same ones.

Changing the truncation, the phantoms, the precision or the splits changes the
data or the models, so data generation and training have to be rerun.

## Tests

```bash
python -m pytest -q
```

Every test is deterministic and needs no data, checkpoint or GPU. Tests that
need the real operator or the phantoms are skipped where `astra`, `scipy`,
`odl` or `dival` are missing; `python -m pytest -rs` lists what was skipped.
Use `python -m pytest`, not a bare `pytest`, which may belong to another Python.

## Credits

This code builds on Simon Göppel's
[data_proximal_networks](https://github.com/sgoep/data_proximal_networks), the
starting point for the data generation, the U-Net and the Nullspace Network.
