# MP2 method notes (v2)

## Failure analysis that drove v2

v1 (maj3 gate + hard-only linear scorer, no PCA): **val 87.47%**, test 72.59%.
Remaining val misses (100): wrong maj3 22 / maj2 36 / ties 42.

Probes:
- Soft maj3 override with v1 scorer: **0 fixes**, 1 break → keep/revisit gate only after a stronger scorer.
- Wrong-maj3 gold is often an embedding outlier, but picking outliers is unsafe when maj3 is correct (gold is central ~95%).
- Handcrafted “rich” extras (answer-in-question, numeric support, …) did **not** beat base agreement features once PCA was added.

## Final method (v2)

**Train-fit PCA (90% variance → k=161) + linear listwise softmax over all train questions; always argmax (no majority gate).**

Per candidate features (339-D):
1. **17 handcrafted**: parseable, agreement fraction, unique-mode / tied-top / singleton flags, mode strength, −L2 to mean emb, cos to mean, mean/max pairwise cos, length, question overlap, mode×cos / agreement×cos / minority×cos.
2. **Global PCA**: `(emb − pca_mean) @ components` (k=161).
3. **Local PCA**: `(emb − question_mean) @ components` (same components).

PCA mean/components fit on **train candidate embeddings only** with variance threshold **0.9** (k chosen without validation). Linear weights via listwise softmax CE, cosine LR decay, seed **42**, lr 0.05, epochs 300, l2 1e-4. Standardize with train mean/std.

Inference: score 4 candidates → `argmax`. Checkpoint stores float32 PCA + float64 scorer (~**0.60 MiB**).

## Ablation table (validation)

| Variant | Val acc |
|---|---|
| Majority + emb fallback | 85.46% |
| v1: hard-only linear + gate3 (no PCA) | 87.47% |
| v2 ablate: no PCA (17-D, train-all, gate0) | 87.09% |
| v2 ablate: PCA + gate3 | 90.48% |
| **v2 final: PCA + train-all + gate0** | **91.23%** |
| (exploratory) PCA + rich extras + gate0 | 90.98% (rich extras slightly worse) |

PCA is the main jump (~+4 pp). Dropping the maj3 gate helps once the PCA scorer can fix some wrong maj3 without wrecking easy cases. Seed stability for final config: 5/5 seeds → 90.98–91.23% (seed 42 = 91.23%).

## Official scores (frozen on val, one-shot test)

| Split | Majority baseline | v1 | **v2 final** |
|---|---|---|---|
| Validation | 85.4637% (682/798) | 87.4687% (698/798) | **91.2281% (728/798)** |
| Test | 69.3790% (324/467) | 72.5910% (339/467) | **79.4433% (371/467)** |

Predict time ~0.2–0.3 s with 4 threads on this box (≪ 30 s).

## Artifacts

- `submission.py`, `features.py`, `train.py`
- Checkpoint: `runs/model_bundle/{scorer.npz,meta.json}` (~0.60 MiB)
- `runs/validation-score.json`, `runs/test-score.json`
- Prior v1 checkpoint kept at `runs/model_bundle_v1_gate3_norichpca` (if present)

## Reproduce

```bash
cd code
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # numpy==2.5.3
python check_data.py

python train.py --output runs/model_bundle --seed 42 --lr 0.05 --epochs 300 \
  --l2 1e-4 --pca-var 0.9 --majority-gate 0

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json --output runs/validation-score.json

# Test only after freezing (do not retune)
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Peers may skip training and use the shipped `runs/model_bundle`.

## Remaining risks / failures

v2 val misses (~70): ties ~30, maj2 ~24, maj3 ~18. Wrong maj3 herds and singleton gold with weak PCA geometry still fail. No test-time tuning; audit bank must use the same checkpoint API. Not submitted to the course website from this agent.


## v2 → v2.1 attempt (no improvement; checkpoint unchanged)

Tried on val only (kept 91.2281% best):
- Tiny MLP 32–128: best ~90.98% (< linear)
- Margin abstain → majority: net harmful
- maj3 geometry flip rules: ≤91.23% or worse
- Hard residual MLP blend, seed ensemble, pairwise+listwise, entropy features: no gain

**Did not re-run test.** Final remains val 91.2281% / test 79.4433%.
Report skeleton: `../../report/REPORT.md`.
