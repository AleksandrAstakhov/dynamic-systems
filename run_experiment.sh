
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SYSTEM="${SYSTEM:-ring}"
EPOCHS="${EPOCHS:-50}"
N_SAMPLES="${N_SAMPLES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WINDOW="${WINDOW:-32}"
WARMUP="${WARMUP:-8}"
HORIZON="${HORIZON:-4}"
VAE_DIM="${VAE_DIM:-32}"
LR="${LR:-3e-3}"
N_MODES="${N_MODES:-2}"
SEED="${SEED:-0}"

RING_C="${RING_C:-32}"
RING_REGIME_LEN="${RING_REGIME_LEN:-500}"
RING_V_AMP="${RING_V_AMP:-2.0}"
RING_ELL="${RING_ELL:-0.7}"
RING_ALPHA="${RING_ALPHA:-0.05}"
RING_SIGMA="${RING_SIGMA:-0.30}"

EEG_SUBJECT="${EEG_SUBJECT:-1}"
EEG_RUNS="${EEG_RUNS:-4 8 12}"
EEG_SFREQ="${EEG_SFREQ:-160}"


if [ -z "${GPUS:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        ALL=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | xargs || true)
        GPUS=$(echo "$ALL" | awk '{print $1, $2}' | xargs || true)
    fi
fi

if [ -z "${GPUS:-}" ]; then
    DEVICE_ARG="--device cpu"
    GPU_LIST=""
    echo "[warn] no GPUs detected; using CPU (slow)"
else
    DEVICE_ARG="--device cuda"
    GPU_LIST=$(echo "$GPUS" | tr ' ' ',')
fi

echo "============================================================"
echo " spatial_dyn_phase_attn_v3 experiment (DataParallel multi-GPU)"
echo "============================================================"
echo "  SYSTEM    = $SYSTEM"
echo "  N_SAMPLES = $N_SAMPLES"
echo "  EPOCHS    = $EPOCHS"
echo "  BATCH     = $BATCH_SIZE   WINDOW=$WINDOW  WARMUP=$WARMUP  H=$HORIZON"
echo "  GPUS      = ${GPU_LIST:-cpu}"
echo "============================================================"

mkdir -p data results results/logs
LOG_DIR="results/logs"


rm -f data/series_truth.npz

echo
echo "=== [1/4] Generate data ($SYSTEM) ==="
if [ "$SYSTEM" = "ring" ]; then
    $PYTHON -u data_gen.py --system drift_ring \
        --C "$RING_C" --T "$N_SAMPLES" \
        --regime_len "$RING_REGIME_LEN" \
        --v_amp "$RING_V_AMP" --ell "$RING_ELL" \
        --alpha "$RING_ALPHA" --sigma "$RING_SIGMA" \
        --seed "$SEED"
elif [ "$SYSTEM" = "eeg" ]; then
    MAX_SEC=$(awk "BEGIN{printf \"%.3f\", $N_SAMPLES / $EEG_SFREQ}")
    $PYTHON -u data_gen.py --system eeg \
        --eeg_subject "$EEG_SUBJECT" --eeg_runs $EEG_RUNS \
        --eeg_sfreq "$EEG_SFREQ" \
        --eeg_max_seconds "$MAX_SEC"
else
    echo "[error] unknown SYSTEM=$SYSTEM (use ring or eeg)"
    exit 1
fi


echo
echo "=== [2/4] Global Takens (tau, m) ==="
$PYTHON -u takens_global.py --max_lag 40 --m_max 8 --fnn_thresh 0.05


echo
echo "=== [3/4] Train 3 models sequentially on GPUs: ${GPU_LIST:-cpu} ==="
echo "        (DataParallel inside train.py; output tee'd to terminal + logs)"

train_one() {
    local kind=$1
    local logfile="$LOG_DIR/$kind.log"

    echo
    echo "------------------------------------------------------------"
    echo " [$kind] training start  (log: $logfile)"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        $PYTHON -u train.py --spatial "$kind" \
            --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
            --window "$WINDOW" --warmup "$WARMUP" --horizon "$HORIZON" \
            --vae_dim "$VAE_DIM" --lr "$LR" --n_modes "$N_MODES" \
            --seed "$SEED" $DEVICE_ARG \
        2>&1 | tee "$logfile"
}

for KIND in correlation grand_diff grand_full; do
    train_one "$KIND"
done

echo
echo "=== [4/4] Compare ==="
CUDA_VISIBLE_DEVICES="$GPU_LIST" $PYTHON -u compare.py 2>&1 | tee "$LOG_DIR/compare.log"

echo
echo "=== Done ==="
echo " Artifacts:"
echo "   data/series.npz, data/takens.npz"
[ "$SYSTEM" = "ring" ] && echo "   data/series_truth.npz"
echo "   results/{correlation,grand_diff,grand_full}.pt"
echo "   results/{correlation,grand_diff,grand_full}_metrics.json"
echo "   results/comparison.{json,png}"
echo "   results/logs/{correlation,grand_diff,grand_full,compare}.log"
