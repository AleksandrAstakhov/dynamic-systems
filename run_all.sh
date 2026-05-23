
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/Users/aleksandr/Downloads/mipt-thesis-master/thesis_takens/test/bin/python}"
echo "Using python: $PYTHON"
$PYTHON -c "import torch, scipy, matplotlib; print('torch', torch.__version__)"

echo
echo "=== [1/4] Generate non-stationary distributed dynamics ==="
$PYTHON data_gen.py \
    --C 32 --T 6000 --regime_len 400 \
    --v_amp 2.0 --ell 0.7 --alpha 0.05 --sigma 0.30 \
    --seed 0

echo
echo "=== [2/4] Global Takens (tau, m) search over the whole field ==="
$PYTHON takens_global.py --max_lag 30 --m_max 6 --fnn_thresh 0.05

echo
echo "=== [3/4] Train phase-conditional MHA spatial model ==="
$PYTHON train.py --spatial phase_mha \
    --epochs 25 --batch_size 64 \
    --window 32 --warmup 8 --horizon 4 \
    --vae_dim 32 --lr 3e-3 \
    --lam_aux 0.05 --n_modes 2

echo
echo "=== [4/4] Train correlation-baseline spatial model ==="
$PYTHON train.py --spatial correlation \
    --epochs 25 --batch_size 64 \
    --window 32 --warmup 8 --horizon 4 \
    --vae_dim 32 --lr 3e-3

echo
echo "=== Compare attention vs correlation ==="
$PYTHON compare.py

echo
echo "=== Done. See results/ ==="
ls -la results/
