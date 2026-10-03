#!/usr/bin/env bash
# The whole single-ellipse experiment in one job: every noise level at the four
# truncations the thesis compares, one run after another.
#
#   sbatch slurm_full_run.sh
#
# At every noise level the runs are at
#   tau = 4e-3      the reference,
#   tau = optimum   where the truncation study finds the pseudoinverse best
#                   (run once before the matrix is built; every run checks it),
#   tau = 4.4e-4    which halves dim N(A) against the reference,
#   tau = 6.2e-2    which doubles it.
# At the reference and the optimum, both models are also trained with the
# EXTRA_SEEDS and attacked again (the suite only), for the variation between
# trainings. EXTRA_SEEDS="" skips them.
# The stages of a run are in run_pipeline.sh, shared with slurm_ellipses.sh
# (the multi-ellipse phantoms, a job of its own). Data and trained models that
# exist are reused, and a job that stopped is simply resubmitted: runs finished
# at the current commit are skipped.
#
# The runs whose models exist already come first, so a problem in the attack,
# epoch or render stage shows up within the first hour rather than after a day
# of training.
#
#   sbatch --export=ALL,NOISES=0.02 slurm_full_run.sh     # one noise level only
#   sbatch --export=ALL,FORCE_RUN=1 slurm_full_run.sh     # redo finished runs
#
# The operator is decomposed in double precision the first time a geometry and
# tau are used, and cached in radon_cache/. A job that needs an entry another
# job is still building waits for it (src/radon.py), so this job and
# slurm_ellipses.sh can run at the same time.
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
cd "$REPO_DIR" || exit 1
mkdir -p logs
source "$REPO_DIR/run_pipeline.sh"
setup_env

NOISES=${NOISES:-"0.005 0.01 0.02 0.05"}
TAU_HALF=0.00044
TAU_DOUBLE=0.062
EXTRA_SEEDS=${EXTRA_SEEDS-"1 2"}

if [ "${RUN_TESTS:-1}" = 1 ]; then run_tests; fi

# Where the pseudoinverse is best, per noise level, from the truncation study.
declare -A TAU_OPT=()
OPT_FILE=$(mktemp)
pinv_optima single "$OPT_FILE"
while read -r n t; do TAU_OPT[$n]=$t; done < "$OPT_FILE"
rm -f "$OPT_FILE"

ROWS=()
for NOISE in $NOISES; do
    if [ -z "${TAU_OPT[$NOISE]}" ]; then
        echo "[abort] the truncation study has no optimum for noise $NOISE (TRUNC_NOISES)" >&2
        exit 1
    fi
    for TAU in "$DEFAULT_SVD_THRESH" "${TAU_OPT[$NOISE]}" "$TAU_HALF" "$TAU_DOUBLE"; do
        # the optimum may coincide with another threshold of the row
        dup=0
        for row in "${ROWS[@]}"; do
            read -r n t <<< "$row"
            if [ "$n" = "$NOISE" ] && same_tau "$t" "$TAU"; then dup=1; fi
        done
        [ "$dup" = 1 ] || ROWS+=("$NOISE $TAU")
    done
done

# Runs with trained models first, then those that still have to train.
for pass in trained untrained; do
    for row in "${ROWS[@]}"; do
        read -r NOISE TAU <<< "$row"
        if has_models "$NOISE" "$TAU" single; then kind=trained; else kind=untrained; fi
        if [ "$kind" = "$pass" ]; then
            seeds=""
            if same_tau "$TAU" "$DEFAULT_SVD_THRESH" || same_tau "$TAU" "${TAU_OPT[$NOISE]}"; then
                seeds=$EXTRA_SEEDS
            fi
            run_one "$NOISE" "$TAU" single "${TAU_OPT[$NOISE]}" "$seeds"
        fi
    done
done

echo "============================================"
echo "All runs done: $(date)"
echo "============================================"
