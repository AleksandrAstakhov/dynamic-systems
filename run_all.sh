#!/usr/bin/env bash
# run_all.sh — запуск обоих экспериментов с нуля (голый репозиторий)
#
# Использование:
#   bash run_all.sh                        # оба эксперимента
#   bash run_all.sh drift                  # только drift_ring
#   bash run_all.sh eeg                    # только EEG
#
# Переменные окружения:
#   PYTHON   — путь к Python (должен иметь torch, mne, scipy)
#   RESULTS  — базовая директория для результатов (default: results)

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS:-results}"
MODE="${1:-all}"   # all | drift | eeg

# ─── Проверка окружения ───────────────────────────────────────────────────────
echo "=== Проверка окружения ==="
echo "  Python: $PYTHON"
$PYTHON -c "
import torch, scipy, matplotlib
print(f'  torch={torch.__version__}')
try:
    import mne; print(f'  mne={mne.__version__}')
except ImportError:
    print('  mne=НЕТ (нужен для EEG)')
"

# ─── 1. Drift Ring ────────────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "drift" ]]; then
    echo ""
    echo "############################################################"
    echo "### EXPERIMENT 1: Drift Ring (синтетика)                 ###"
    echo "############################################################"
    echo ""
    echo "--- Данные генерируются внутри run_compressed.py (каждый сид отдельно) ---"
    echo ""

    # Параметры:
    #   C=16, T=3000        — 16 сенсоров, 3000 временных шагов на сид
    #   window=24           — длина контекстного окна
    #   warmup=6            — прогрев энкодера
    #   horizon=4           — горизонт прогноза
    #   n_modes=6           — экспертов в MoE-GA (2 истинных матрицы T±,
    #                         n_modes=6 даёт max_rank=6×latent_dim ≥ C=16)
    #   n_seeds=5           — для доверительных интервалов
    #   epochs=50           — обучение

    $PYTHON run_compressed.py \
        --C 16 --T 3000 \
        --window 24 --warmup 6 --horizon 4 \
        --n_modes 6 \
        --n_seeds 5 --epochs 50 \
        --spatials correlation grand_diff grand_full \
        --results_dir "$RESULTS/drift_ring"

    echo ""
    echo "--- Визуализация drift_ring ---"
    $PYTHON plot_training.py --results_dir "$RESULTS/drift_ring" || true
    $PYTHON visualize_matrices.py \
        --mat_dir "$RESULTS/drift_ring/seed_0/matrices" \
        --out_dir  "$RESULTS/drift_ring" || true
fi

# ─── 2. EEG ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "eeg" ]]; then
    echo ""
    echo "############################################################"
    echo "### EXPERIMENT 2: EEG — PhysioNet EEGBCI                ###"
    echo "############################################################"

    # Загрузка данных (если ещё нет)
    EEG_DATA="data/eeg_series.npz"
    if [[ ! -f "$EEG_DATA" ]]; then
        echo ""
        echo "--- Загрузка EEG через MNE (PhysioNet EEGBCI, subject=1, runs=4,8,12) ---"
        echo "    Параметры: sfreq=160 Hz, bandpass 1-40 Hz, max 187.5 с"
        mkdir -p data
        $PYTHON data_gen.py \
            --system eeg \
            --eeg_subject 1 \
            --eeg_runs 4 8 12 \
            --eeg_sfreq 160 \
            --eeg_max_seconds 187.5 \
            --out "$EEG_DATA"
    else
        echo ""
        echo "--- Данные уже есть: $EEG_DATA ---"
    fi

    echo ""
    echo "--- Обучение EEG ---"
    # Параметры:
    #   --no_filter         — данные уже отфильтрованы в data_gen.py (1-40 Hz)
    #   window=64           — 0.4 с при 160 Hz (стандартная эпоха motor imagery)
    #   warmup=16           — 0.1 с прогрев
    #   horizon=8           — 50 мс горизонт прогноза
    #   tau/m — авто        — AMI→tau, FNN→m (запускается один раз для всех сидов)
    #   n_modes — авто      — max(MP, bands, rank_cov):
    #                         MP=4 (4 собств. зн. > λ+=1.62),
    #                         bands=3 (δ/θ/α), rank_cov=ceil(4/latent_dim)
    #                         → n_modes=4, max_rank=4×3=12 ≥ 4 сигн. компонент ✓
    #   Gumbel k=1          — hard routing (Switch Transformer MoE, one-hot forward)
    #   n_seeds=3           — для CI (64 канала = долго, 3 сида разумно)
    #   epochs=50           — обучение

    $PYTHON run_eeg.py \
        --data "$EEG_DATA" \
        --no_filter \
        --window 64 --warmup 16 --horizon 8 \
        --n_seeds 3 --epochs 50 \
        --batch_size 128 \
        --spatials correlation grand_diff grand_full \
        --results_dir "$RESULTS/eeg"

    echo ""
    echo "--- Визуализация EEG ---"
    $PYTHON plot_training.py --results_dir "$RESULTS/eeg" || true
    $PYTHON visualize_matrices.py \
        --mat_dir "$RESULTS/eeg/seed_0/matrices" \
        --out_dir  "$RESULTS/eeg" || true
fi

echo ""
echo "============================================================"
echo "=== Готово ==="
echo "  Drift Ring : $RESULTS/drift_ring/"
echo "  EEG        : $RESULTS/eeg/"
echo "============================================================"
