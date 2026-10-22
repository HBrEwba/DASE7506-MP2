# DASE7506 MP2 提交仓库 — LLM 候选推理选择

**课程：** DASE7506 Advanced Machine Learning Methods · Fall 2026  
**作业：** [MP2 — Selecting Reliable LLM Reasoning](https://xudongwu-0.github.io/courses/dase7506/mp2/)  
**学生：** BORUI HAN（学号：【待填】）  
**协议：** `dase7506-mp2-open-hard-verifier-v5`

本仓库在课程官方材料之上实现并冻结了一种训练内 PCA + 列表 Softmax 线性打分方法，供同伴用官方 `predict.py` / `score_predictions.py` **离线复现**（无需重训即可验证分数）。

上游课程仓库（数据与工具来源）：<https://github.com/xudongwu-0/DASE7506-MP2>

---

## 官方分数（冻结后）

| Split | Accuracy | Correct / N |
|---|---:|---:|
| **Validation** | **91.2281%** | 728 / 798 |
| **Test**（验证选型后只评一次） | **79.4433%** | 371 / 467 |

分数 JSON：`code/runs/validation-score.json`、`code/runs/test-score.json`  
检查点：`code/runs/model_bundle/`（约 **0.60 MiB**，含 `scorer.npz` + `meta.json`）  
报告：`report/REPORT.md`

---

## 方法摘要

1. 用 `common.parse_answer` 与嵌入几何构造 **17 维手工特征**（同意度、众数/平票标志、到均值距离与余弦等）。
2. 仅在**训练集**候选嵌入上拟合 PCA（方差阈值 **0.9 → k=161**），拼接全局投影与题内残差投影 → 每候选 **339 维**。
3. 列表 Softmax 交叉熵训练线性打分器（seed **42**，lr 0.05，epochs 300，L2 1e-4）；推断对四候选 `argmax`。
4. **无** maj3 硬门控；无外部预训练权重 / 评测期 LLM API。依赖仅 `numpy==2.5.3`。

详见 `report/REPORT.md` 与 `code/runs/NOTES.md`。

---

## 环境安装

```bash
git clone <本提交仓库 URL>
cd DASE7506-MP2   # 或你的仓库目录名
cd code
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py
```

本仓库已包含完整 `code/data/`（与课程包一致），克隆后即可离线运行。请保留 `code/data/LICENSE.txt` 与 `code/data/NOTICE.txt`。

---

## 仅复现测试分数（推荐同伴评审）

使用已提交检查点，**不必重训**：

```bash
cd code
source .venv/bin/activate
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

验证集同理，将 `--split test` 改为 `validation`。应得到与上表一致的 `accuracy`。

---

## 从零训练（可选）

```bash
cd code
source .venv/bin/activate
python train.py --output runs/model_bundle --seed 42 --lr 0.05 --epochs 300 \
  --l2 1e-4 --pca-var 0.9 --majority-gate 0
```

种子与超参见 `code/runs/model_bundle/meta.json`。

---

## 仓库结构（提交相关）

```
├── README.md                 # 本文件
├── guide/GUIDE.md            # 课程作业说明
├── report/REPORT.md          # ≤10 页报告草稿（Markdown）
└── code/
    ├── submission.py         # predict(bank, arrays, checkpoint)
    ├── features.py           # 手工特征 + PCA
    ├── train.py              # 训练入口
    ├── predict.py / score_predictions.py / common.py / check_data.py
    ├── requirements.txt      # numpy==2.5.3
    ├── data/                 # 完整评测数据 + LICENSE/NOTICE
    └── runs/
        ├── model_bundle/     # 推理资产（必须）
        ├── validation-score.json
        ├── test-score.json
        └── NOTES.md
```

`.venv/`、`__pycache__/`、大型预测 JSON 与消融目录已由 `.gitignore` 排除。

---

## 归属与声明

- 题目与参考答案源自 [GSM8K](https://github.com/openai/grade-school-math)；候选与嵌入由课程方基于 Qwen3-0.6B 制备。详见 `code/data/LICENSE.txt`、`code/data/NOTICE.txt`。
- AI 辅助与复用声明见 `report/REPORT.md` 第 7 节（提交前请核对并签字）。
- 网站提交截止：**2026-10-31 23:59（北京时间）**。
