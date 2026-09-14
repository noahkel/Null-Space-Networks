#!/usr/bin/env bash
# One-off diagnostic: does drawing the simulated noise through a second,
# essentially untruncated operator buy anything over using the operating
# truncation everywhere?  See truncation_study.py.
#
#   sbatch slurm_truncation_study.sh
#   sbatch --export=ALL,N_SAMPLES=128 slurm_truncation_study.sh
#
# It answers a design question once; it is not part of producing a result.
#
#SBATCH --job-name=nsn-truncation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=c7021201@uibk.ac.at

set -o pipefail

REPO_DIR=${REPO_DIR:-/scratch/noah/Null-Space-Networks}
N_SAMPLES=${N_SAMPLES:-64}
IMG_SIZE=${IMG_SIZE:-128}
SVD_THRESH=${SVD_THRESH:-4e-3}
EPS=${EPS:-"0.005 0.01 0.02 0.05"}

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
echo "Start time:    $(date)"
echo "============================================"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "============================================"

python -u truncation_study.py \
    --img_size "$IMG_SIZE" \
    --svd_thresh "$SVD_THRESH" \
    --n_samples "$N_SAMPLES" \
    --noise $EPS \
    --cache_dir radon_cache \
    --out truncation_study.json
status=$?

echo "============================================"
echo "End time:      $(date)   exit=$status"
echo "============================================"
exit $status
