#!/usr/bin/env bash
# Two one-off diagnostics about the truncation threshold tau. See
# truncation_study.py; neither is part of producing a result.
#
#   mode=noise  does drawing the simulated noise through a second, essentially
#               untruncated operator buy anything over using the operating
#               truncation everywhere?
#   mode=tau    does the null space every per-channel number is stated on
#               depend on where tau was put?
#
#   sbatch slurm_truncation_study.sh                                  # both
#   sbatch --export=ALL,MODE=tau slurm_truncation_study.sh
#   sbatch --export=ALL,MODE=tau,N_SAMPLES=128 slurm_truncation_study.sh
#   sbatch --export=ALL,MODE=tau,TAUS="1e-3 4e-3 1.6e-2" slurm_truncation_study.sh
#
# The tau sweep slices one decomposition instead of building an operator per
# threshold, so adding values to TAUS costs matmuls, not SVDs.
#
#SBATCH --job-name=nsn-truncation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=c7021201@uibk.ac.at

set -o pipefail

REPO_DIR=${REPO_DIR:-/scratch/noah/Null-Space-Networks}
MODE=${MODE:-both}
N_SAMPLES=${N_SAMPLES:-64}
IMG_SIZE=${IMG_SIZE:-128}
SVD_THRESH=${SVD_THRESH:-4e-3}
EPS=${EPS:-"0.005 0.01 0.02 0.05"}

# Factor-of-two steps around the operating point, a decade either way. The
# operating value is added by the script if it is missing here.
TAUS=${TAUS:-"1e-4 3e-4 1e-3 2e-3 4e-3 8e-3 1.6e-2 3.2e-2"}
AMP_PROBES=${AMP_PROBES:-8}
# Set to a tau to build a real adapter there and check the sliced factors agree.
# Costs a second SVD, so it is off unless asked for.
VERIFY_REBUILD=${VERIFY_REBUILD:-}

OUT=${OUT:-truncation_study.json}

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
echo "Mode:          $MODE"
echo "Reference tau: $SVD_THRESH"
echo "Sweep taus:    $TAUS"
echo "Start time:    $(date)"
echo "============================================"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "============================================"

verify_args=()
if [ -n "$VERIFY_REBUILD" ]; then
    verify_args=(--verify_rebuild "$VERIFY_REBUILD")
fi

python -u truncation_study.py \
    --mode "$MODE" \
    --img_size "$IMG_SIZE" \
    --svd_thresh "$SVD_THRESH" \
    --taus $TAUS \
    --n_samples "$N_SAMPLES" \
    --noise $EPS \
    --amp_probes "$AMP_PROBES" \
    "${verify_args[@]}" \
    --cache_dir radon_cache \
    --out "$OUT"
status=$?

echo "============================================"
echo "End time:      $(date)   exit=$status"
echo "============================================"
exit $status
