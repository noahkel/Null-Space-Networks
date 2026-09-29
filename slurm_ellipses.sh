#!/usr/bin/env bash
# DIVAL's multi-ellipse phantoms in place of the single ellipse, at one noise
# level: whether the findings carry over to images with more structure. Two
# runs, at the reference tau and at the tau where the truncation study finds
# the pseudoinverse best for these phantoms.
#
#   sbatch slurm_ellipses.sh
#
# The stages of a run are those of slurm_full_run.sh (run_pipeline.sh),
# including the reuse of existing data and models and the skipping of runs
# finished at the current commit. The two jobs can run at the same time.
#
#SBATCH --job-name=nsn-ellipses
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c7021201@uibk.ac.at

set -eo pipefail

REPO_DIR=${REPO_DIR:-/scratch/noah/Null-Space-Networks}
cd "$REPO_DIR" || exit 1
mkdir -p logs
source "$REPO_DIR/run_pipeline.sh"
setup_env

NOISE=0.01
# Where the pseudoinverse is best for these phantoms at this noise level.
TAU_OPT=0.025

if [ "${RUN_TESTS:-1}" = 1 ]; then run_tests; fi

for TAU in "$DEFAULT_SVD_THRESH" "$TAU_OPT"; do
    run_one "$NOISE" "$TAU" ellipses "$TAU_OPT"
done

echo "============================================"
echo "All runs done: $(date)"
echo "============================================"
