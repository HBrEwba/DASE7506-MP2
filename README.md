# DASE7506 Mini-Project 2 — Selecting Reliable LLM Reasoning

**Course:** DASE7506 Advanced Machine Learning Methods · Fall 2026  
**Assignment:** [MP2 — Selecting Reliable LLM Reasoning](https://xudongwu-0.github.io/courses/dase7506/mp2/)  
**Student:** BORUI HAN (3036781090)  
**Protocol:** `dase7506-mp2-open-hard-verifier-v5`

Public submission repository. Upstream course package (data and tools): https://github.com/xudongwu-0/DASE7506-MP2

## Official scores (frozen)

| Split | Accuracy | Correct / N |
|---|---:|---:|
| **Validation** | **91.2281%** | 728 / 798 |
| **Test** (scored once after validation selection) | **79.4433%** | 371 / 467 |

Score JSON: `code/runs/validation-score.json`, `code/runs/test-score.json`  
Checkpoint: `code/runs/model_bundle/` (~**0.60 MiB**, `scorer.npz` + `meta.json`)  
Report: `report/REPORT.pdf` (also `REPORT.md` / `REPORT.docx`)

## Method (short)

1. Build **17 handcrafted** consistency / geometry features with `common.parse_answer` and embedding geometry.
2. Fit PCA on **training** candidate embeddings only (variance threshold **0.9 → k=161**); concatenate global and within-question residual projections → **339-D** per candidate.
3. Train a linear listwise softmax scorer (seed **42**, lr 0.05, epochs 300, L2 1e-4); at inference take `argmax` over the four scores.
4. **No** maj3 hard gate; no external pretrained weights or evaluation-time LLM/API calls. Dependency: `numpy==2.5.3`.

Details: `report/REPORT.pdf` and `code/runs/NOTES.md`.

## Setup

```bash
git clone https://github.com/HBrEwba/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py
```

This repository includes the full `code/data/` package. Keep `code/data/LICENSE.txt` and `NOTICE.txt`.

## Reproduce test score (peer review; no retrain)

```bash
cd code
source .venv/bin/activate
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

For validation, use `--split validation`. The scorer `accuracy` field should match the table above.

## Optional retrain

```bash
cd code
source .venv/bin/activate
python train.py --output runs/model_bundle --seed 42 --lr 0.05 --epochs 300 \
  --l2 1e-4 --pca-var 0.9 --majority-gate 0
```

Hyperparameters are also recorded in `code/runs/model_bundle/meta.json`.

## Layout

```
├── README.md
├── guide/GUIDE.md
├── report/REPORT.pdf | REPORT.md | REPORT.docx
└── code/
    ├── submission.py, features.py, train.py
    ├── predict.py, score_predictions.py, common.py, check_data.py
    ├── requirements.txt
    ├── data/
    └── runs/model_bundle/
```

## Submission links

- Code: https://github.com/HBrEwba/DASE7506-MP2
- Checkpoint: https://github.com/HBrEwba/DASE7506-MP2/tree/main/code/runs/model_bundle
- Report: https://github.com/HBrEwba/DASE7506-MP2/blob/main/report/REPORT.pdf
- Website: https://xudongwu-0.github.io/courses/dase7506/mp2/
