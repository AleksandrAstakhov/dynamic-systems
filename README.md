# Spatial Dynamics with Phase-Conditional Attention vs. Correlation

Расширение проекта `spatial_dyn_torch` со следующими отличиями:

1. **Глобальный поиск (τ, m)** по всей системе, а не по одному датчику
   (среднее AMI + покомпонентный FNN).
2. **Пространственный блок --- phase-conditional MHA** из нашего
   эксперимента с дрейфующим ядром: банки attention, маршрутизация
   по latentам + entropy-regularization против mode collapse.
3. **Честный baseline --- статическая корреляционная матрица**
   (Pearson по тренировочным данным), используется как графовый
   адъяцентный вес в той же архитектуре.
4. **Цель проверки:** на одном и том же датасете показать, что
   attention восстанавливает пространственную связь точнее
   корреляции и даёт лучший прогноз.

## Структура

```
data/                    artefacts (series.npz, takens.npz)
results/                 metrics, plots
models/
  vae.py                 SequenceVAE (causal Transformer encoder)
  phase_mha.py           Phase-conditional MHA spatial coupler
  correlation.py         Static correlation-based spatial coupler
  forecaster.py          VAE + temporal Transformer + spatial module + decoder
data_gen.py              non-stationary ring with binary-regime drift kernel
takens_global.py         (τ, m) chosen globally over all sensors
train.py                 joint training
compare.py               attention vs correlation on same data
run_all.sh               end-to-end pipeline
```

## Запуск

```bash
./run_all.sh
```

В результате в `results/comparison.json` лежат метрики обеих моделей,
в `results/*.png` --- графики прогноза и выученных матриц связи.
