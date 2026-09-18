#!/usr/bin/env bash
# The full experiment for one noise level, in one job:
#   tests -> data generation -> training -> attack suite (+ Lipschitz)
#         -> epoch study -> figures
#
#   sbatch slurm_full_run.sh                                # noise 0.01
#   sbatch --export=ALL,NOISE=0.02 slurm_full_run.sh
#   sbatch --export=ALL,NOISE=0.02,CREATE_DATA=0,TRAIN=0 slurm_full_run.sh
#
# A second truncation, to test whether a channel-restricted result depends on
# where tau was put (see truncation_study.py --mode tau for choosing the value):
#
#   sbatch --export=ALL,NOISE=0.01,SVD_THRESH=1e-3 slurm_full_run.sh
#
# Anything other than the default tau writes to its own data, model and output
# directories, so a second truncation never overwrites the main experiment.
#
# A failing stage aborts the job: every later stage consumes its output.
#
#SBATCH --job-name=nsn-full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c7021201@uibk.ac.at

set -eo pipefail

REPO_DIR=${REPO_DIR:-/scratch/noah/Null-Space-Networks}
NOISE=${NOISE:-0.01}

# Geometry / dataset
IMG_SIZE=${IMG_SIZE:-128}
MIN_ANGLE=${MIN_ANGLE:-0}
MAX_ANGLE=${MAX_ANGLE:-120}
NUM_THETAS=${NUM_THETAS:-180}
N_SAMPLES=${N_SAMPLES:-5000}
MODELS=${MODELS:-resnet,nsn}

# The truncation of the operator. It is baked into the data (the noise is drawn
# through this operator's range projector) and into summary.json, from where
# train.py and attack.py read it, so a new value means regenerating the data.
SVD_THRESH=${SVD_THRESH:-4e-3}
DEFAULT_SVD_THRESH=4e-3
# The default keeps the paths it has always had; any other tau gets its own
# tree, so the two never share data, checkpoints or results. The tag is the
# string as written, so spell a value the same way across runs: SVD_THRESH=1e-3
# and SVD_THRESH=0.001 are the same threshold but two directories.
if [ "$SVD_THRESH" = "$DEFAULT_SVD_THRESH" ]; then TAU_TAG=""; else TAU_TAG="_tau${SVD_THRESH}"; fi

DATA_BASE=${DATA_BASE:-/scratch/noah/data_matrices}
MODEL_BASE=${MODEL_BASE:-/scratch/noah/models_matrices}
DATA_ROOT=${DATA_BASE}${TAU_TAG}
DATA_DIR=$DATA_ROOT/$NOISE
MODEL_DIR=${MODEL_BASE}${TAU_TAG}/$NOISE
OUT_DIR=${OUT_DIR:-attacks_n${NOISE}${TAU_TAG}_l2}

# Attack budgets. eps is scaled per sample by ||y_i|| inside attack.py and
# defaults to the training noise level, so no eps is passed here.
MAX_SAMPLES=${MAX_SAMPLES:-128}
EPOCH_STUDY_MAX=${EPOCH_STUDY_MAX:-32}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1}
LIPSCHITZ_SAMPLES=${LIPSCHITZ_SAMPLES:-32}
LIPSCHITZ_ITERS=${LIPSCHITZ_ITERS:-16}
# Subspaces the local gain is estimated in: null, range (= null-complement) and
# unrestricted. Each costs a full pass of power iterations; set to "null" alone
# for the comparable number only.
LIPSCHITZ_RESTRICTIONS=${LIPSCHITZ_RESTRICTIONS:-null,range,full}

# Stage toggles
RUN_TESTS=${RUN_TESTS:-1}
CREATE_DATA=${CREATE_DATA:-1}
TRAIN=${TRAIN:-1}
RUN_EPOCH_STUDY=${RUN_EPOCH_STUDY:-1}

cd "$REPO_DIR" || exit 1
mkdir -p logs
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

module purge
module load anaconda/anaconda3
module load cuda/12.5
source ~/.bashrc
conda activate data_prox2

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-<interactive>}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):        ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Working dir:   $(pwd)"
echo "Commit:        $(git rev-parse --short HEAD 2>/dev/null || echo '?')$(
                      test -n "$(git status --porcelain 2>/dev/null)" && echo ' (DIRTY)')"
echo "Noise:         $NOISE"
echo "Truncation:    tau = $SVD_THRESH"
echo "Data:          $DATA_DIR"
echo "Models:        $MODEL_DIR"
echo "Output:        $OUT_DIR"
echo "Start time:    $(date)"
echo "============================================"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "============================================"

banner() { echo; echo "=== [$1] noise=$NOISE at $(date) ==="; }

# python -m pytest, not bare pytest: the PATH pytest can belong to a python
# without torch, where every test silently skips.
if [ "$RUN_TESTS" -eq 1 ]; then
    banner tests
    python -m pytest -q
fi

if [ "$CREATE_DATA" -eq 1 ]; then
    banner "data generation"
    python -u -m src.create_phantom_data --img_size "$IMG_SIZE" --noise "$NOISE" \
        --min_angle "$MIN_ANGLE" --max_angle "$MAX_ANGLE" --num_thetas "$NUM_THETAS" \
        --n_samples "$N_SAMPLES" --svd_thresh "$SVD_THRESH" --out_dir "$DATA_ROOT"
fi

if [ ! -f "$DATA_DIR/summary.json" ]; then
    echo "[abort] no data at $DATA_DIR (summary.json missing)" >&2
    exit 1
fi

if [ "$TRAIN" -eq 1 ]; then
    banner training
    python -u train.py --data_dir "$DATA_DIR" --out_dir "$MODEL_DIR" \
        --models "$MODELS" --checkpoint-every "$CHECKPOINT_EVERY"
fi

# Every model x every attack on one shared sample set, plus the attack-free
# Lipschitz estimate.
banner "attack suite"
python -u attack.py --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" \
    --max-samples "$MAX_SAMPLES" --lipschitz \
    --lipschitz-samples "$LIPSCHITZ_SAMPLES" --lipschitz-iters "$LIPSCHITZ_ITERS" \
    --lipschitz-restrictions "$LIPSCHITZ_RESTRICTIONS" \
    --out-dir "$OUT_DIR"

# Writes epoch_study/*.csv into the same run dir, so it runs before rendering.
if [ "$RUN_EPOCH_STUDY" -eq 1 ]; then
    banner "epoch study"
    python -u attack.py --epoch-study --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" \
        --max-samples "$EPOCH_STUDY_MAX" --out-dir "$OUT_DIR"
fi

# Compute nodes are headless.
banner render
MPLBACKEND=Agg python -u visualise.py "$OUT_DIR"

echo "============================================"
echo "End time:      $(date)"
echo "============================================"
