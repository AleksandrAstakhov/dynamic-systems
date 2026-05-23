
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPU_PHASE="${GPU_PHASE:-0}"
GPU_CORR="${GPU_CORR:-1}"

echo "=== Using GPU $GPU_PHASE for phase_mha, GPU $GPU_CORR for correlation ==="

COMMON_ARGS="--epochs 50 --batch_size 128 \
    --window 64 --warmup 16 --horizon 8 \
    --vae_dim 64 --lr 3e-3 \
    --lam_in 1.0 --lam_rec 0.1 --device cuda"

CUDA_VISIBLE_DEVICES=$GPU_PHASE $PYTHON train.py --spatial phase_mha \
    $COMMON_ARGS --lam_aux 0.05 --n_modes 3 \
    > results/log_phase_mha.txt 2>&1 &
PID_PHASE=$!
echo "  phase_mha   started  (PID $PID_PHASE, GPU $GPU_PHASE, log: results/log_phase_mha.txt)"

CUDA_VISIBLE_DEVICES=$GPU_CORR $PYTHON train.py --spatial correlation \
    $COMMON_ARGS \
    > results/log_correlation.txt 2>&1 &
PID_CORR=$!
echo "  correlation started  (PID $PID_CORR, GPU $GPU_CORR, log: results/log_correlation.txt)"

trap 'kill $PID_PHASE $PID_CORR 2>/dev/null' EXIT
wait $PID_PHASE && echo "  phase_mha   done"
wait $PID_CORR  && echo "  correlation done"
trap - EXIT

echo
echo "=== Running compare.py ==="
CUDA_VISIBLE_DEVICES=$GPU_PHASE $PYTHON compare.py
echo
echo "=== Done. Logs in results/log_*.txt ==="
