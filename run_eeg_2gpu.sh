
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0 1}"
SUBJECT="${SUBJECT:-1}"
RUNS="${RUNS:-4 8 12}"             
SFREQ="${SFREQ:-160}"
N_MODES="${N_MODES:-3}"

echo "=== Config ==="
echo "  PYTHON  = $PYTHON"
echo "  GPUS    = $GPUS"
echo "  SUBJECT = $SUBJECT"
echo "  RUNS    = $RUNS"
echo "  SFREQ   = $SFREQ Hz"
echo "  N_MODES = $N_MODES"
echo

mkdir -p data results

echo "=== [1/4] Load EEG (PhysioNet eegbci) ==="
$PYTHON data_gen.py --system eeg \
    --eeg_subject $SUBJECT --eeg_runs $RUNS \
    --eeg_sfreq $SFREQ \
    --out data/series.npz \
    --eeg_max_seconds 187.5

echo
echo "=== [2/4] Global Takens (tau, m) ==="
$PYTHON takens_global.py --max_lag 80 --m_max 16 --fnn_thresh 0.02

echo
echo "=== [3/4] Train phase_mha on GPUs $GPUS (DataParallel) ==="
$PYTHON train_2gpu.py --spatial phase_mha --gpus $GPUS \
    --epochs 50 --batch_size 128 \
    --window 64 --warmup 16 --horizon 8 \
    --vae_dim 64 --lr 3e-3 \
    --lam_in 1.0 --lam_rec 0.1 --lam_aux 0.05 --n_modes $N_MODES \
    --out results/phase_mha.pt

echo
echo "=== [4/4] Train correlation baseline on GPUs $GPUS (DataParallel) ==="
$PYTHON train_2gpu.py --spatial correlation --gpus $GPUS \
    --epochs 50 --batch_size 128 \
    --window 64 --warmup 16 --horizon 8 \
    --vae_dim 64 --lr 3e-3 \
    --lam_in 1.0 --lam_rec 0.1 \
    --out results/correlation.pt

echo
echo "=== Compare ==="
CUDA_VISIBLE_DEVICES="$(echo $GPUS | cut -d' ' -f1)" \
    $PYTHON compare.py

echo
echo "=== Done. Artifacts: ==="
ls -la results/ data/
