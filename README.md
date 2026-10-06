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
| `src/create_phantom_data.py` | ellipse phantoms (one per image, or DIVAL's multi-ellipse ones), noisy sinograms, pseudoinverse initial reconstructions |
| `src/unet.py`, `src/wrappers.py` | the UNet and the two wrappers (`RESNET`, `NSN`) |
| `train.py` | trains both architectures on one noise level |
| `src/attack.py` (entry: `attack.py`) | PGD attack suite, epoch study, metrics, Lipschitz estimate |
| `src/visualisations.py` (entry: `visualise.py`) | every figure, rebuilt from saved artifacts only |
| `slurm_full_run.sh` | the whole single-ellipse experiment as one Slurm job: every noise level at four truncations |
| `slurm_ellipses.sh` | the multi-ellipse phantoms at one noise level, as a job of its own |
| `run_pipeline.sh` | the stages of one run, shared by both jobs (sourced, not submitted) |
| `src/truncation.py` | the truncation study of a run: the τ per noise level at which the pseudoinverse is best, how much the channel split depends on τ, candidates for a second truncation |
| `make_tables.py` | the rows of every results table of the thesis and the numbers its prose cites, from the run directories alone |
| `tests/test_nsn.py` | the test suite |

## Running

Everything runs from the repository root, in the `data_prox2` environment.

The whole experiment is two Slurm jobs, which can run at the same time:

```bash
sbatch slurm_full_run.sh     # single ellipses: 4 noise levels x 4 truncations, 16 runs
sbatch slurm_ellipses.sh     # multi-ellipse phantoms at noise 0.01: 2 runs
```

At every noise level `slurm_full_run.sh` runs the reference τ = 4·10⁻³, the τ at
which the truncation study finds the pseudoinverse best (the job runs the study
once before it builds the run matrix, and every run checks it again), and
τ = 4.4·10⁻⁴ and 6.2·10⁻², which halve and double dim N_τ. Each run goes
through the tests' environment, the truncation study, data, training, the attack
suite with the Lipschitz estimate, the budget sweep, the epoch study (total and
null-space error) and the figures; a failing stage aborts the job. At the
reference and the optimal τ both models are trained again with the seeds
`EXTRA_SEEDS` (default `1 2`) and attacked by the suite, into `seed<s>/` of the
run's model and output directories.

A run reuses the data when its `summary.json` exists and the models when every
model has its `_history.json` (both are written last), after checking that the
data were made for this noise level, τ and phantom family. Everything after the
training is always recomputed, so every number of a run comes from one version
of the code, and a finished run leaves `<run>/.complete` with the commit. A job
that stopped is simply resubmitted: runs finished at the current commit are
skipped. Runs whose models exist go first, so a problem in the later stages
shows up before a day of training.

```bash
sbatch --export=ALL,NOISES=0.02 slurm_full_run.sh      # one noise level only
sbatch --export=ALL,FORCE_RUN=1 slurm_full_run.sh      # redo finished runs
sbatch --export=ALL,FORCE_TRAIN=1 slurm_full_run.sh    # retrain on the existing data
sbatch --export=ALL,EXTRA_SEEDS= slurm_full_run.sh     # no extra seeds
sbatch --export=ALL,CHECKPOINT_EVERY=1 slurm_full_run.sh  # epoch study at every epoch
```

All directories of these runs carry the tag `RUN_TAG` (default `_v2`): the data,
models and results of the earlier runs, whose noise was confined to range(U_k),
stay where they are, and the stages refuse data without the new noise model.

A truncation or phantom family other than the reference writes to its own data,
model and output directories (`..._tau0.011`, `..._ellipses`), so no run
overwrites another. The operator is decomposed once per geometry and τ and
cached in `radon_cache/`; a job that needs an entry another job is still
building waits for it.

The stages by hand, for one noise level:

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

`truncation` and `create_phantom_data` must be run with `-m`, since they import
from `src`.

Once the runs are finished, the tables of the thesis come from

```bash
python make_tables.py --runs 'attacks_*_v2' --out thesis_tables
```

which writes the LaTeX rows of every results table and `numbers.txt` with the
values the text cites (medians, paired differences with bootstrap intervals,
the share of the exact worst case, seeds).

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

Tests that need the real operator are skipped where `astra` or `scipy` are
missing.

## Thesis

`thesis.tex` builds with `pdflatex` + `biber`. It needs `references.bib` and
`Parallel-beam-geometry.png`, which are not in the repository.

## Credits

This code builds on Simon Göppel's
[data_proximal_networks](https://github.com/sgoep/data_proximal_networks), the
starting point for the data generation, the U-Net and the Nullspace Network.
