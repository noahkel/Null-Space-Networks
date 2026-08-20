#!/bin/bash
# The whole cluster pipeline
#
#   bash pipeline.sh                    submit the chained Slurm arrays (login node)
#   bash pipeline.sh --dry-run          print the plan, submit nothing
#   bash pipeline.sh --only render      submit a subset of the stages
#   bash pipeline.sh --serial           submit one long serial job instead
#
#   PINV_MODE=unthresholded bash pipeline.sh   # unthresholded pinv init
#   MATRIX_MODE=0 bash pipeline.sh             # astra backend instead of matrix
#   CREATE_DATA=0 TRAIN=0 bash pipeline.sh     # reuse existing data + models
#   EPS="0.005 0.01 0.02" bash pipeline.sh     # a different noise sweep
#
# Layout of the submitted arrays:
#
#   prep    array 0..N_noise-1            data generation + training
#     |  afterok
#     +-- attack  array 0..N_noise-1           attack suite per noise level
#     +-- epoch   array 0..N_noise*N_model-1   per-epoch study per (noise, model)
#            |  afterok (both)
#            +-- render  array 0..N_noise-1    figures per run dir
#
# attack and epoch depend only on prep, so they run concurrently; render waits
# for both, because the epoch study writes epoch_study/*.csv into the same run
# dir and those curves are part of the render. Critical path is
# prep + max(attack, epoch) + render ~= 24 h, against ~141 h fully serial.
#
# The directives below are the array worker's defaults; the submitter overrides
# --job-name / --array / --time (and --output for --serial) per stage.
#SBATCH --job-name=nsn
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=all
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=c7021201@uibk.ac.at
#SBATCH --signal=B:TERM@60

# Not `set -e`: a stage is allowed to fail without killing the rest of the job
# (see stage()/finish()), and the `[ "$TOGGLE" -eq 1 ] && stage ...` idiom below
# returns nonzero whenever a toggle is off. The submitter turns it on locally.
set -o pipefail

# A job with no STAGE means someone sbatch'ed the submitter. That would try to
# call sbatch from inside a job — which many clusters refuse outright — and
# would look like nothing happened. Caught here, before anything else runs.
if [ -z "${STAGE:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "pipeline.sh was sbatch'ed with no STAGE set." >&2
    echo "Run the submitter on the login node instead:" >&2
    echo "    bash pipeline.sh            # chained arrays" >&2
    echo "    bash pipeline.sh --serial   # one serial job" >&2
    exit 1
fi

# Inside a job REPO_DIR arrives via --export; a hand-run gets it from the script
# path. Slurm runs the batch script from a spool copy on the compute node, so
# the script path is only trustworthy outside a job.
if [ -z "${REPO_DIR:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        REPO_DIR=/scratch/noah/Null-Space-Networks
    else
        REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    fi
fi
cd "$REPO_DIR" || exit 1


# =========================================================================== #
# 1. Configuration — every knob, overridable from the submitting shell.
# =========================================================================== #

CREATE_SCRIPT=create_data.py

# ── Radon backend ────────────────────────────────────────────────────────────
# 1 = MatrixRadonAdapter: explicit A_la matrix with a truncated-SVD pseudoinverse.
#     The backend every current result uses, and the only one with a pinv init.
# 0 = AstraRadonAdapter. No SVD truncation, so PINV_MODE does not apply and the
#     data carries an fbp init instead; attack.py detects whichever of the two
#     exists.
MATRIX_MODE=${MATRIX_MODE:-1}
MATRIX_SUFFIX=""
[ "$MATRIX_MODE" -eq 1 ] && MATRIX_SUFFIX="_matrices"

# ── Geometry / dataset ───────────────────────────────────────────────────────
IMG_SIZE=${IMG_SIZE:-128}          # pixels
MIN_ANGLE=${MIN_ANGLE:-0}
MAX_ANGLE=${MAX_ANGLE:-120}        # limited angle
NUM_THETAS=${NUM_THETAS:-180}
N_SAMPLES=${N_SAMPLES:-5000}
TYPE=${TYPE:-ellipses}             # dataloader
MODELS=${MODELS:-resnet,nsn}

# Noise levels. One data set + one model set + one attack run per level.
EPS=${EPS:-"0.005 0.01 0.02 0.05"}

# ── pinv initialisation operator ─────────────────────────────────────────────
# "thresholded"   = truncated-SVD A_la^+ (the baseline every result so far uses)
# "unthresholded" = full A_la^+, keeps the tiny singular values
# The pinv init is baked into the *data*, so a switch means regenerating the
# dataset; the tag below keeps the two variants in separate trees. Meaningless
# under MATRIX_MODE=0 (astra has no SVD to truncate), so it earns no tag there.
PINV_MODE=${PINV_MODE:-thresholded}
PINV_TAG=""
if [ "$MATRIX_MODE" -eq 1 ] && [ "$PINV_MODE" != "thresholded" ]; then
    PINV_TAG="_${PINV_MODE}"
fi
# Training minimises L2/MSE and there is no configurable objective, so the pinv
# variant is the only thing left that namespaces an output tree.
VARIANT_TAG="${PINV_TAG}"

# Path scheme, chosen so the defaults reproduce the trees that already exist:
DATA_BASE=${DATA_BASE:-/scratch/noah/data${MATRIX_SUFFIX}${PINV_TAG}}
MODEL_BASE=${MODEL_BASE:-/scratch/noah/models${MATRIX_SUFFIX}${VARIANT_TAG}}

# ── Attack budgets ───────────────────────────────────────────────────────────
# eps is scaled per sample by ||y_i|| inside attack.py and --suite-eps defaults
# to the training noise level from summary.json — the principled budget — so no
# eps is passed explicitly.
MAX_SAMPLES=${MAX_SAMPLES:-128}        # test samples for the headline numbers
EPOCH_STUDY_MAX=${EPOCH_STUDY_MAX:-32} # smaller budget for the per-epoch study
# The study is scoped to one init. Astra data has no pinv folder, so the default
# follows the backend rather than being a constant that silently mismatches.
EPOCH_STUDY_INIT=${EPOCH_STUDY_INIT:-$([ "$MATRIX_MODE" -eq 1 ] && echo pinv || echo fbp)}

# Attack-free Lipschitz estimate, computed alongside the suite. Job 20585 ran it
# at the n=4 default, which is thin for a headline "NSN < 1 < ResNet" claim.
LIPSCHITZ_SAMPLES=${LIPSCHITZ_SAMPLES:-32}
LIPSCHITZ_ITERS=${LIPSCHITZ_ITERS:-16}

# ── Stage toggles ────────────────────────────────────────────────────────────
CREATE_DATA=${CREATE_DATA:-1}
TRAIN=${TRAIN:-1}
RUN_EPOCH_STUDY=${RUN_EPOCH_STUDY:-1}

# ── Per-stage wall clocks ────────────────────────────────────────────────────
# attack is the long pole at ~20 h/task; epoch is ~8 h per model. Generous but
# explicit — an unset --time is what let job 20585 run for six days without
# anyone choosing that.
PREP_TIME=${PREP_TIME:-12:00:00}
ATTACK_TIME=${ATTACK_TIME:-48:00:00}
EPOCH_TIME=${EPOCH_TIME:-24:00:00}
RENDER_TIME=${RENDER_TIME:-04:00:00}   # matplotlib only, ~35-40 min per run dir
SERIAL_TIME=${SERIAL_TIME:-168:00:00}  # --serial chains every stage in one job

# Cap concurrent tasks per array (%N) so one sweep does not take the whole
# partition. Unset MAX_CONCURRENT for no cap.
MAX_CONCURRENT=${MAX_CONCURRENT:-4}


# =========================================================================== #
# 2. Derived names and task tables.
#
# Each table prints one task per line; the array index selects the line, and the
# submitter uses the line count to size --array. Both roles read the same
# functions, so the plan and the work cannot disagree.
# =========================================================================== #

# The trailing _l2 is historical — the suite is L2-only now — but it is what the
# existing result trees are called, so a re-run lands in the same directory.
out_dir_for()   { echo "attacks_n${1}${VARIANT_TAG}_l2"; }
data_dir_for()  { echo "$DATA_BASE/$1"; }
model_dir_for() { echo "$MODEL_BASE/$1"; }

tasks_prep()   { for noise in $EPS; do echo "$noise"; done; }
tasks_attack() { for noise in $EPS; do echo "$noise"; done; }

tasks_epoch() {
    local model
    for noise in $EPS; do
        for model in ${MODELS//,/ }; do echo "$noise $model"; done
    done
}

# One render task per run directory. RENDER_DIRS overrides with an explicit
# list, which is how you render run directories that already exist on the
# cluster — including ones from an older naming scheme:
#
#   RENDER_DIRS="attacks_ellipses_fast_n0.005_l2" bash pipeline.sh --only render
tasks_render() {
    if [ -n "${RENDER_DIRS:-}" ]; then
        for d in $RENDER_DIRS; do echo "$d"; done
    else
        for noise in $EPS; do out_dir_for "$noise"; done
    fi
}

nth_task()    { "$1" | sed -n "$(( $2 + 1 ))p"; }   # <table-fn> <0-based index>
count_tasks() { "$1" | wc -l; }


# =========================================================================== #
# 3. Slurm plumbing — notifications, per-stage failure accounting, exit traps
#    and cluster environment setup. Used by the worker only; the submitter must
#    not install the traps or it would ping "job finished" for itself.
# =========================================================================== #
NTFY_URL="https://ntfy.sh/c7021201_slurmjobs"

# Hardened, best-effort ntfy POST. Direct egress from compute nodes comes and
# goes (curl 6 "Could not resolve host: ntfy.sh"), so when the direct POST fails
# the message is relayed over ssh through the submit (login) node, which does
# have internet access. Logs a single warning line per failure instead of
# spraying raw curl errors into the .err file. Never aborts the job.
notify() {
    local err
    err=$(curl -fsS --max-time 15 --retry 3 --retry-connrefused \
               -H "Title: $1" -d "$2" "$NTFY_URL" 2>&1 >/dev/null) && return
    if [ -n "$SLURM_SUBMIT_HOST" ] && \
       printf '%s' "$2" | \
       ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
           "$SLURM_SUBMIT_HOST" \
           "curl -fsS --max-time 15 --retry 3 -H $(printf '%q' "Title: $1") \
                 --data-binary @- $(printf '%q' "$NTFY_URL")" \
           >/dev/null 2>&1; then
        echo "[notify] INFO: no direct egress on ${SLURMD_NODENAME:-?} ($err) — relayed via $SLURM_SUBMIT_HOST" >&2
        return
    fi
    echo "[notify] WARN: ntfy POST failed (title: $1) — direct: $err; ssh relay via ${SLURM_SUBMIT_HOST:-<no SLURM_SUBMIT_HOST>} also failed" >&2
}

# Lightweight progress ping (same channel), used after each major stage so the
# phone timeline shows how far the job is without opening the .out log.
progress() { notify "SLURM progress: $SLURM_JOB_NAME" "$1 at $(date)"; }

# A stage that fails must not kill the job (later stages / noise levels should
# still run) but must not be silent either: each failure pings the phone right
# away, and finish() turns the count into a nonzero exit at the end.
FAILED_STEPS=0
step_failed() {
    FAILED_STEPS=$((FAILED_STEPS + 1))
    notify "SLURM step FAILED: $SLURM_JOB_NAME" "$1 at $(date)"
}

# stage "<label>" <command...> — run one pipeline stage, ping "<label> done" on
# success or record + ping "<label> (exit N)" on failure.
stage() {
    local label="$1"
    shift
    if "$@"; then
        progress "$label done"
    else
        step_failed "$label (exit $?)"
    fi
}

# Fire exactly one final notification however the job ends — normal exit, a
# python error, or a Slurm SIGTERM from TIMEOUT/scancel/OOM.
_notified=0
_finish() {
    local rc=$?
    [ "$_notified" -eq 1 ] && return
    _notified=1
    if [ "$rc" -eq 0 ]; then
        notify "SLURM finished: $SLURM_JOB_NAME" "Job $SLURM_JOB_ID done on ${SLURMD_NODENAME:-?} at $(date) (exit 0)"
    else
        notify "SLURM FAILED: $SLURM_JOB_NAME" "Job $SLURM_JOB_ID exited $rc on ${SLURMD_NODENAME:-?} at $(date)"
    fi
}

install_traps() {
    trap _finish EXIT
    # Convert Slurm's termination signals into a normal exit so the trap runs.
    trap 'exit 143' TERM
    trap 'exit 130' INT
}

# Cluster environment (modules + conda env) followed by a diagnostics banner.
setup_env() {
    module purge
    module load anaconda/anaconda3
    module load cuda/12.5
    source ~/.bashrc
    conda activate data_prox2

    echo "============================================"
    echo "Job ID:        $SLURM_JOB_ID"
    echo "Node:          $SLURMD_NODENAME"
    echo "GPU(s):        $CUDA_VISIBLE_DEVICES"
    echo "Working dir:   $(pwd)"
    echo "Start time:    $(date)"
    echo "============================================"
    python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    echo "============================================"
}

# Last line of the worker: exit nonzero when any stage failed, so the EXIT trap
# sends "FAILED" and Slurm sends the FAIL mail instead of a misleading
# "finished (exit 0)".
finish() {
    if [ "$FAILED_STEPS" -gt 0 ]; then
        echo "Job finished at $(date) with $FAILED_STEPS FAILED stage(s) — see the .err log"
        exit 1
    fi
    echo "Job finished at $(date)"
    exit 0
}


# =========================================================================== #
# 4. Pipeline stages. Each reads $NOISE / $DATA_DIR_NOISE / $MODEL_DIR_NOISE /
#    $OUT_DIR from the caller, so the array worker and the serial loop run
#    byte-identical commands.
# =========================================================================== #
create_data() {
    # --pinv_mode is accepted and ignored under --matrix_mode 0 (astra has no
    # SVD to truncate), so it can be passed unconditionally.
    python -u "$CREATE_SCRIPT" --img_size $IMG_SIZE --noise $NOISE \
        --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS \
        --n_samples $N_SAMPLES --matrix_mode $MATRIX_MODE --pinv_mode $PINV_MODE \
        --out_dir $DATA_BASE
}

train_models() {
    python -u train.py --type $TYPE --out_dir $MODEL_DIR_NOISE \
        --data_dir $DATA_DIR_NOISE --models $MODELS \
        --checkpoint-every-epoch
}

# One suite run = every model x every attack (total/null/range/targeted) x every
# detected init, on one shared sample set. Splitting those into separate jobs
# would be both slower and less fair — they must share identical samples to be
# compared. The attack-free Lipschitz estimate rides along with it.
attack_suite() {
    echo "=== [attack] noise=$NOISE data=$DATA_DIR_NOISE models=$MODEL_DIR_NOISE at $(date) ==="
    python -u attack.py --data-root $DATA_DIR_NOISE --model-dir $MODEL_DIR_NOISE \
        --max-samples $MAX_SAMPLES --lipschitz \
        --out-dir "$OUT_DIR"
}

# Scoped to one init and a smaller sample budget. Writes epoch_study/*.csv into
# the same run dir as the suite so visualise.py renders the curves alongside the
# suite figures. $MODEL restricts it to one architecture (one array task each);
# empty means every model, which is what the serial path does.
epoch_study() {
    echo "=== [epoch-study] noise=$NOISE init=$EPOCH_STUDY_INIT model=${MODEL:-<all>} at $(date) ==="
    python -u attack.py --epoch-study --init $EPOCH_STUDY_INIT \
        ${MODEL:+--models "$MODEL"} \
        --data-root $DATA_DIR_NOISE --model-dir $MODEL_DIR_NOISE \
        --max-samples $EPOCH_STUDY_MAX --out-dir "$OUT_DIR"
}

# Rebuild every figure from the artifacts already on disk. No torch, no models,
# no radon operator — matplotlib and numpy only, so this is a cheap CPU task.
#
# Rendering always happens here, never after fetching: it is 35+ min per run dir
# on a laptop, it parallelises for free across array tasks, and it means
# fetch_results.ps1 can leave the .npz render inputs on the cluster entirely.
# MPLBACKEND=Agg because compute nodes are headless.
render_figures() {
    echo "=== [render] $OUT_DIR at $(date) ==="
    if [ ! -d "$OUT_DIR" ]; then
        echo "[abort] no run directory $OUT_DIR to render"
        return 1
    fi
    MPLBACKEND=Agg python -u visualise.py "$OUT_DIR"
}

# python -m pytest, not bare pytest: the PATH pytest belongs to a different
# python without torch, so every test silently skipped (importorskip) there.
run_tests() {
    python -m pytest -q
}

# Resolve the per-noise paths every stage needs.
set_noise_paths() {
    NOISE=$1
    DATA_DIR_NOISE=$(data_dir_for "$NOISE")
    MODEL_DIR_NOISE=$(model_dir_for "$NOISE")
    OUT_DIR=$(out_dir_for "$NOISE")
}


# =========================================================================== #
# 5. Worker — one array task, or (STAGE=all) every stage in sequence.
# =========================================================================== #
run_worker() {
    mkdir -p logs
    export PYTHONPATH=$REPO_DIR:$PYTHONPATH
    install_traps
    notify "SLURM started: $SLURM_JOB_NAME" \
           "Job $SLURM_JOB_ID started on ${SLURMD_NODENAME:-?} at $(date)"
    setup_env

    echo "[config] stage=$STAGE task=${SLURM_ARRAY_TASK_ID:-<none>}"
    echo "[config] matrix_mode=$MATRIX_MODE pinv_mode=$PINV_MODE variant_tag='${VARIANT_TAG}'"

    [ "$STAGE" = "all" ] && { run_serial; return; }

    local task_id=${SLURM_ARRAY_TASK_ID:?no SLURM_ARRAY_TASK_ID (submit via bash pipeline.sh)}
    local line
    case "$STAGE" in
    prep)
        line=$(nth_task tasks_prep "$task_id")
        [ -n "$line" ] || { echo "[abort] no prep task at index $task_id"; exit 1; }
        set_noise_paths "$line"
        [ "$CREATE_DATA" -eq 1 ] && stage "[noise $NOISE] data generation" create_data
        require_data || finish
        [ "$TRAIN" -eq 1 ] && stage "[noise $NOISE] training" train_models
        ;;
    attack)
        line=$(nth_task tasks_attack "$task_id")
        [ -n "$line" ] || { echo "[abort] no attack task at index $task_id"; exit 1; }
        set_noise_paths "$line"
        require_data || finish
        stage "[noise $NOISE] attack suite" attack_suite
        ;;
    epoch)
        [ "$RUN_EPOCH_STUDY" -eq 1 ] || { echo "[skip] RUN_EPOCH_STUDY=0"; finish; }
        read -r line MODEL <<<"$(nth_task tasks_epoch "$task_id")"
        [ -n "$MODEL" ] || { echo "[abort] no epoch task at index $task_id"; exit 1; }
        set_noise_paths "$line"
        require_data || finish
        stage "[noise $NOISE / $MODEL] epoch study" epoch_study
        ;;
    render)
        # Straight from the table — no data or models needed, so this stage can
        # be submitted on its own against run dirs that already exist.
        OUT_DIR=$(nth_task tasks_render "$task_id")
        [ -n "$OUT_DIR" ] || { echo "[abort] no render task at index $task_id"; exit 1; }
        stage "[render] $OUT_DIR" render_figures
        ;;
    *)
        echo "[abort] unknown STAGE '$STAGE' (expected prep|attack|epoch|render|all)"
        exit 1
        ;;
    esac

    echo "Finished $STAGE task $task_id at: $(date)"
    finish
}

require_data() {
    [ -f "$DATA_DIR_NOISE/summary.json" ] && return 0
    echo "[abort] no data for noise $NOISE at $DATA_DIR_NOISE (summary.json missing)"
    step_failed "[$STAGE] no data at $DATA_DIR_NOISE"
    return 1
}

# STAGE=all: every stage for every noise level, in one job. Slower than the
# arrays by roughly the sum of the stages, and kept for the case where the
# partition cannot take an array.
run_serial() {
    echo "=== Testing before attacking ==="
    # A failure aborts: job 20585 spent six days on unvalidated code because
    # this gate was advisory and pytest was missing from the env.
    if ! run_tests; then
        step_failed "pre-attack tests (pytest exit $?)"
        echo "[abort] pre-attack tests failed — refusing to burn GPU days on unvalidated code."
        finish
    fi
    progress "pre-attack tests finished — starting noise levels: $EPS"

    local total idx=0
    total=$(echo $EPS | wc -w)
    for noise in $EPS; do
        idx=$((idx + 1))
        set_noise_paths "$noise"
        local tag="noise $NOISE ($idx/$total)"

        [ "$CREATE_DATA" -eq 1 ] && stage "[$tag] data generation" create_data
        [ "$TRAIN" -eq 1 ]       && stage "[$tag] training"        train_models

        if ! require_data; then
            progress "[$tag] SKIPPED — no data at $DATA_DIR_NOISE"
            continue
        fi

        stage "[$tag] attack suite" attack_suite
        MODEL=""   # every model, rather than one array task per architecture
        [ "$RUN_EPOCH_STUDY" -eq 1 ] && stage "[$tag] epoch study" epoch_study
        # Render last, so the epoch curves are included.
        stage "[$tag] render figures" render_figures
    done
    echo "Finished all noise levels at: $(date)"
    finish
}


# =========================================================================== #
# 6. Submitter — runs on the login node, chains the arrays with --dependency.
# =========================================================================== #
run_submitter() {
    set -eu

    local dry_run=0 skip_tests=0 serial=0
    # Which stages to submit. Dependencies are formed only between stages
    # actually being submitted, so a narrowed submission starts immediately
    # instead of waiting on jobs that will never exist.
    local stages="prep,attack,epoch,render"
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run)    dry_run=1 ;;
            --skip-tests) skip_tests=1 ;;
            --serial)     serial=1 ;;
            --only)       stages="$2"; shift ;;
            --only=*)     stages="${1#--only=}" ;;
            *) echo "unknown argument: $1" >&2
               echo "usage: bash pipeline.sh [--dry-run] [--skip-tests] [--serial]" >&2
               echo "                        [--only prep,attack,epoch,render]" >&2
               exit 2 ;;
        esac
        shift
    done

    want() { case ",$stages," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

    # Single predicate for "is this stage actually going to be submitted", used
    # both to print the plan and to submit, so the plan cannot disagree with
    # what happens.
    will_submit() {
        want "$1" || return 1
        case "$1" in
            prep)  [ "$CREATE_DATA" -eq 1 ] || [ "$TRAIN" -eq 1 ] ;;
            epoch) [ "$RUN_EPOCH_STUDY" -eq 1 ] ;;
            *)     return 0 ;;
        esac
    }

    local s
    for s in ${stages//,/ }; do
        case "$s" in
            prep|attack|epoch|render) ;;
            *) echo "unknown stage '$s' (expected prep, attack, epoch or render)" >&2; exit 2 ;;
        esac
    done

    local n_prep n_attack n_epoch n_render
    n_prep=$(count_tasks tasks_prep)
    n_attack=$(count_tasks tasks_attack)
    n_epoch=$(count_tasks tasks_epoch)
    n_render=$(count_tasks tasks_render)

    echo "=== Pipeline plan ==========================================="
    echo "  repo         $REPO_DIR"
    echo "  commit       $(git rev-parse --short HEAD 2>/dev/null || echo '?')$(
            test -n "$(git status --porcelain 2>/dev/null)" && echo ' (DIRTY)')"
    echo "  generator    $CREATE_SCRIPT, loader $TYPE"
    echo "  backend      matrix_mode=$MATRIX_MODE $(if [ "$MATRIX_MODE" -eq 1 ]; then echo '(matrix + truncated-SVD pinv)'; else echo '(astra, no pinv init)'; fi)"
    echo "  pinv_mode    $PINV_MODE$(if [ "$MATRIX_MODE" -eq 0 ]; then echo '  (n/a for astra)'; fi)"
    echo "  variant tag  '${VARIANT_TAG:-<baseline>}'"
    echo "  data         $DATA_BASE"
    echo "  models       $MODEL_BASE"
    echo "  noise levels $EPS"
    echo "-------------------------------------------------------------"
    if [ "$serial" -eq 1 ]; then
        echo "  mode         serial (one job, every stage in sequence)"
    else
        echo "  stages       $stages"
    fi
    echo "-------------------------------------------------------------"
    if [ "$serial" -eq 0 ]; then
        # "-" marks a stage that will not be submitted, so the plan shows both
        # what the configuration contains and what this invocation will do.
        mark() { if will_submit "$1"; then echo " "; else echo "-"; fi; }
        printf " %s prep    %2d task(s): %s\n" "$(mark prep)"   "$n_prep"   "$(tasks_prep   | tr '\n' ',' | sed 's/,$//')"
        printf " %s attack  %2d task(s): %s\n" "$(mark attack)" "$n_attack" "$(tasks_attack | tr '\n' ',' | sed 's/,$//')"
        printf " %s epoch   %2d task(s): %s\n" "$(mark epoch)"  "$n_epoch"  "$(tasks_epoch  | tr '\n' ',' | sed 's/,$//')"
        printf " %s render  %2d task(s): %s\n" "$(mark render)" "$n_render" "$(tasks_render | tr '\n' ',' | sed 's/,$//')"
    fi
    echo "  outputs:"
    tasks_render | sed 's/^/    /'
    echo "============================================================="

    if [ "$dry_run" -eq 1 ]; then
        echo "(dry run — nothing submitted)"
        exit 0
    fi

    # ── Pre-submit test gate ─────────────────────────────────────────────────
    # Run once here on the login node rather than inside every array task: it
    # fails in seconds, before anything is queued, instead of 16 times on GPU
    # nodes. A missing pytest is an abort, not a skip. Attacks and training are
    # what the tests cover; a render-only submission does not need them.
    if [ "$skip_tests" -eq 0 ] && \
       { [ "$serial" -eq 1 ] || want attack || want epoch || want prep; }; then
        echo "Running pre-submit tests ..."
        if ! python -m pytest --version >/dev/null 2>&1; then
            echo "pytest is not importable by '$(command -v python || echo python)'." >&2
            echo "Activate the cluster env first (e.g. conda activate data_prox2)," >&2
            echo "or pass --skip-tests to submit without validating." >&2
            exit 1
        fi
        if ! run_tests; then
            echo >&2
            echo "Pre-submit tests failed — refusing to queue GPU days on unvalidated code." >&2
            echo "Fix the failures, or pass --skip-tests if you know what you are doing." >&2
            exit 1
        fi
        echo
    fi

    # Every knob the worker reads is exported, so the array tasks see exactly
    # the configuration printed above rather than re-deriving defaults.
    local exports="ALL,REPO_DIR=$REPO_DIR,MATRIX_MODE=$MATRIX_MODE"
    exports="$exports,PINV_MODE=$PINV_MODE,EPS=$EPS"
    exports="$exports,CREATE_DATA=$CREATE_DATA,TRAIN=$TRAIN,RUN_EPOCH_STUDY=$RUN_EPOCH_STUDY"
    exports="$exports,MAX_SAMPLES=$MAX_SAMPLES,EPOCH_STUDY_MAX=$EPOCH_STUDY_MAX"

    if [ "$serial" -eq 1 ]; then
        # No array, so %A/%a would render as the NO_VAL sentinel — give the log
        # the plain job-id pattern instead.
        local jid
        jid=$(sbatch --parsable --job-name="nsn-serial${VARIANT_TAG}" \
                     --time="$SERIAL_TIME" --mail-type=BEGIN,END,FAIL \
                     --output="logs/%x_%j.out" --error="logs/%x_%j.err" \
                     --export="$exports,STAGE=all" "$REPO_DIR/pipeline.sh")
        echo "submitted serial  job $jid  (every stage, $(echo $EPS | wc -w) noise level(s))"
        print_footer
        return
    fi

    local pct=""
    [ -n "$MAX_CONCURRENT" ] && pct="%$MAX_CONCURRENT"

    submit() {  # <job-name> <n-tasks> <time> <stage> [dependency-job-id ...]
        local name=$1 n=$2 time=$3 stage=$4
        shift 4
        local args=(--job-name="$name" --array="0-$((n - 1))$pct" --time="$time"
                    --export="$exports,STAGE=$stage")
        # afterok:<id>:<id> — every listed job must succeed first. Empty ids (a
        # skipped stage) drop out so the dependency is simply shorter.
        local dep="" id
        for id in "$@"; do
            [ -n "$id" ] && dep="${dep:+$dep:}$id"
        done
        [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
        sbatch --parsable "${args[@]}" "$REPO_DIR/pipeline.sh"
    }

    local prep_id="" attack_id="" epoch_id="" render_id="" deps="" d
    if will_submit prep; then
        prep_id=$(submit "nsn-prep${VARIANT_TAG}" "$n_prep" "$PREP_TIME" prep)
        echo "submitted prep    job $prep_id  ($n_prep tasks)"
    else
        echo "skipping prep — reusing existing data and models"
    fi

    if will_submit attack; then
        attack_id=$(submit "nsn-attack${VARIANT_TAG}" "$n_attack" "$ATTACK_TIME" attack "$prep_id")
        echo "submitted attack  job $attack_id  ($n_attack tasks)${prep_id:+  after $prep_id}"
    fi

    if will_submit epoch; then
        epoch_id=$(submit "nsn-epoch${VARIANT_TAG}" "$n_epoch" "$EPOCH_TIME" epoch "$prep_id")
        echo "submitted epoch   job $epoch_id  ($n_epoch tasks)${prep_id:+  after $prep_id}"
    fi

    # One render task per run dir, after the suite and the epoch study, so both
    # sets of figures are included. With --only render there is nothing to wait
    # for and it starts straight away against what is on disk.
    if will_submit render; then
        render_id=$(submit "nsn-render${VARIANT_TAG}" "$n_render" "$RENDER_TIME" \
                           render "$attack_id" "$epoch_id")
        for d in "$attack_id" "$epoch_id"; do
            [ -n "$d" ] && deps="${deps:+$deps + }$d"
        done
        echo "submitted render  job $render_id  ($n_render tasks)${deps:+  after $deps}"
    fi

    print_footer
}

print_footer() {
    echo
    echo "watch:  squeue -u \$USER"
    echo "logs:   logs/nsn-*_<jobid>_<task>.out"
    echo "fetch:  .\\fetch_results.ps1           # all figures + tables"
    echo "        .\\fetch_results.ps1 -Summary  # skip the per-sample images"
}


# =========================================================================== #
# 7. Entry point.
# =========================================================================== #
if [ -n "${STAGE:-}" ]; then
    run_worker            # an array task, or a hand-run for debugging
else
    run_submitter "$@"    # the login node (the sbatch-misuse case aborted above)
fi
