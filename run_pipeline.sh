#!/usr/bin/env bash
# The stages of one run -- one noise level, one truncation tau, one phantom
# family -- shared by slurm_full_run.sh and slurm_ellipses.sh, which source this
# file and call run_one for every run of their matrix. Not submitted on its own.
#
#   truncation study -> data -> training -> attack suite (+ Lipschitz)
#                    -> epoch study -> figures
#
# A run reuses what it already has: the data when summary.json exists, the
# models when every model has its _history.json (both are written last, so
# they only exist once their stage finished). Everything after the training is
# always recomputed, so every number of a run comes from one version of the
# code, and a finished run leaves a stamp with the commit in its output
# directory. A resubmitted job skips the runs stamped with the current commit.
#
#   FORCE_DATA=1    regenerate the data (and hence retrain)
#   FORCE_TRAIN=1   retrain on the existing data
#   FORCE_RUN=1     rerun runs that carry a stamp of the current commit
#
# Anything other than the reference tau and the single-ellipse phantoms writes to
# its own data, model and output directories, so no run overwrites another. The
# directory tag is tau as written, so spell a value the same way across runs.

# Geometry / dataset
IMG_SIZE=${IMG_SIZE:-128}
MIN_ANGLE=${MIN_ANGLE:-0}
MAX_ANGLE=${MAX_ANGLE:-120}
NUM_THETAS=${NUM_THETAS:-180}
N_SAMPLES=${N_SAMPLES:-5000}
MODELS=${MODELS:-resnet,nsn}

# The reference truncation; runs at it keep the untagged paths.
DEFAULT_SVD_THRESH=4e-3

DATA_BASE=${DATA_BASE:-/scratch/noah/data_matrices}
MODEL_BASE=${MODEL_BASE:-/scratch/noah/models_matrices}

# The attack budget is the noise level, scaled per sample by ||y_i|| inside
# attack.py, for the attack suite and the epoch study alike.
MAX_SAMPLES=${MAX_SAMPLES:-128}
EPOCH_STUDY_MAX=${EPOCH_STUDY_MAX:-32}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1}
LIPSCHITZ_SAMPLES=${LIPSCHITZ_SAMPLES:-32}
LIPSCHITZ_ITERS=${LIPSCHITZ_ITERS:-16}
# null, range (= null-complement), unrestricted, and cross (input in the
# null-complement, output in the null space: the gain a null-space attack exploits).
LIPSCHITZ_RESTRICTIONS=${LIPSCHITZ_RESTRICTIONS:-null,range,full,cross}
# The total error, and the null-space error, which is the one informative about
# the NSN's learned correction.
EPOCH_OBJECTIVES=${EPOCH_OBJECTIVES:-"mse null"}

# Truncation study: training phantoms it looks at, and the noise levels it
# evaluates (the run's own is always added).
TRUNC_SAMPLES=${TRUNC_SAMPLES:-64}
TRUNC_NOISES=${TRUNC_NOISES:-"0.005 0.01 0.02 0.05"}

FORCE_DATA=${FORCE_DATA:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_RUN=${FORCE_RUN:-0}

# Modules, conda environment and the commit every stamp records.
setup_env() {
    module purge
    module load anaconda/anaconda3
    module load cuda/12.5
    source ~/.bashrc
    conda activate data_prox2
    export PYTHONPATH=$REPO_DIR:$PYTHONPATH

    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then COMMIT="$COMMIT-dirty"; fi

    echo "============================================"
    echo "Job ID:        ${SLURM_JOB_ID:-<interactive>}"
    echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
    echo "GPU(s):        ${CUDA_VISIBLE_DEVICES:-<none>}"
    echo "Working dir:   $(pwd)"
    echo "Commit:        $COMMIT"
    echo "Start time:    $(date)"
    python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    echo "============================================"
}

# python -m pytest, not bare pytest: the PATH pytest can belong to a python
# without torch, where every test silently skips.
run_tests() {
    echo; echo "=== [tests] at $(date) ==="
    python -m pytest -q
}

banner() { echo; echo "=== [$1] noise=$NOISE tau=$SVD_THRESH phantoms=$PHANTOM at $(date) ==="; }

# run_paths NOISE TAU PHANTOM: the directories of one run.
run_paths() {
    NOISE=$1
    SVD_THRESH=$2
    PHANTOM=$3
    if [ "$SVD_THRESH" = "$DEFAULT_SVD_THRESH" ]; then TAU_TAG=""; else TAU_TAG="_tau${SVD_THRESH}"; fi
    if [ "$PHANTOM" = "single" ]; then PHANTOM_TAG=""; else PHANTOM_TAG="_${PHANTOM}"; fi
    DATA_ROOT=${DATA_BASE}${TAU_TAG}${PHANTOM_TAG}
    DATA_DIR=$DATA_ROOT/$NOISE
    MODEL_DIR=${MODEL_BASE}${TAU_TAG}${PHANTOM_TAG}/$NOISE
    OUT_DIR=attacks_n${NOISE}${TAU_TAG}${PHANTOM_TAG}_l2
}

# has_models NOISE TAU PHANTOM: whether that run's training has finished.
has_models() {
    run_paths "$@"
    local m
    for m in ${MODELS//,/ }; do
        [ -f "$MODEL_DIR/init_pinv/checkpoints/${m}_history.json" ] || return 1
    done
    return 0
}

# run_one NOISE TAU PHANTOM EXPECTED_OPT
#
# EXPECTED_OPT is the tau at which the truncation study should find the
# pseudoinverse best at this noise level. The run matrix was chosen from it, so
# the run aborts if the study now finds another.
run_one() {
    run_paths "$1" "$2" "$3"
    local expected_opt=$4
    local stamp="$OUT_DIR/.complete"

    echo
    echo "######## run: noise=$NOISE tau=$SVD_THRESH phantoms=$PHANTOM -> $OUT_DIR"
    if [ "$FORCE_RUN" != 1 ] && [[ "$COMMIT" != *-dirty ]] \
            && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$COMMIT" ]; then
        echo "[skip] finished at commit $COMMIT"
        return 0
    fi
    mkdir -p "$OUT_DIR"
    rm -f "$stamp"

    banner "truncation study"
    python -u -m src.truncation --img_size "$IMG_SIZE" --min_angle "$MIN_ANGLE" \
        --max_angle "$MAX_ANGLE" --num_thetas "$NUM_THETAS" --phantom "$PHANTOM" \
        --svd_thresh "$SVD_THRESH" --noise $TRUNC_NOISES "$NOISE" --run_noise "$NOISE" \
        --n_samples "$TRUNC_SAMPLES" --cache_dir radon_cache --out "$OUT_DIR/truncation"
    python - "$OUT_DIR/truncation/truncation.json" "$expected_opt" <<'EOF'
import json, math, sys
found = float(json.load(open(sys.argv[1]))["recommended"]["tau"])
if not math.isclose(found, float(sys.argv[2]), rel_tol=1e-6):
    sys.exit(f"[abort] the truncation study puts the optimum at tau={found:g}, "
             f"the run matrix at tau={sys.argv[2]}: update the matrix")
EOF

    local fresh_data=0
    if [ "$FORCE_DATA" = 1 ] || [ ! -f "$DATA_DIR/summary.json" ]; then
        banner "data generation"
        python -u -m src.create_phantom_data --img_size "$IMG_SIZE" --noise "$NOISE" \
            --min_angle "$MIN_ANGLE" --max_angle "$MAX_ANGLE" --num_thetas "$NUM_THETAS" \
            --n_samples "$N_SAMPLES" --svd_thresh "$SVD_THRESH" --phantom "$PHANTOM" \
            --out_dir "$DATA_ROOT"
        fresh_data=1
    else
        echo "[reuse] data at $DATA_DIR"
    fi
    # The directory names stand for these parameters; make sure the data agree.
    python - "$DATA_DIR/summary.json" "$NOISE" "$SVD_THRESH" "$PHANTOM" <<'EOF'
import json, math, sys
path, noise, tau, phantom = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
s = json.load(open(path))
bad = [f"{k}={v!r}, expected {w!r}" for k, v, w in (
    ("noise_sigma_rel", float(s["noise_sigma_rel"]), noise),
    ("svd_threshold", float(s["svd_threshold"]), tau)) if not math.isclose(v, w, rel_tol=1e-9)]
# data from before the multi-ellipse phantoms carry no "phantom" key
if s.get("phantom", "single") != phantom:
    bad.append(f"phantom={s.get('phantom')!r}, expected {phantom!r}")
if bad:
    sys.exit(f"[abort] {path}: " + "; ".join(bad))
EOF

    if [ "$FORCE_TRAIN" = 1 ] || [ "$fresh_data" = 1 ] || ! has_models "$NOISE" "$SVD_THRESH" "$PHANTOM"; then
        banner training
        python -u train.py --data_dir "$DATA_DIR" --out_dir "$MODEL_DIR" \
            --models "$MODELS" --checkpoint-every "$CHECKPOINT_EVERY"
    else
        echo "[reuse] models at $MODEL_DIR"
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
    local objective
    for objective in $EPOCH_OBJECTIVES; do
        banner "epoch study ($objective)"
        python -u attack.py --epoch-study --epoch-objective "$objective" \
            --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" \
            --max-samples "$EPOCH_STUDY_MAX" --out-dir "$OUT_DIR"
    done

    # Compute nodes are headless. Renders the truncation study with the rest.
    banner render
    MPLBACKEND=Agg python -u visualise.py "$OUT_DIR"

    echo "$COMMIT" > "$stamp"
    echo "[done] $OUT_DIR at $(date)"
}
