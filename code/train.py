"""Train listwise-softmax linear scorer with train-fit PCA features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DATA, load_bank, parse_answer, read_jsonl, score_choices
from features import BASE_DIM, candidate_features, fit_pca

SEED = 42
DEFAULT_OUT = Path('runs/model_bundle')


def build_split(split, pca_mean, pca_components, data_dir=DATA):
    bank = load_bank(split, data_dir)
    gold = read_jsonl(Path(data_dir) / f'gsm8k_{split}.jsonl')
    emb = np.load(Path(data_dir) / 'candidates' / f'gsm8k_{split}.npz')['embeddings']
    emb = emb.reshape(len(bank), 4, 1024)
    Xs, Ys, metas = [], [], []
    for q, row, e in zip(gold, bank, emb):
        g = parse_answer(q['answer'])
        feats, parsed, _, _ = candidate_features(row, e, pca_mean, pca_components)
        y = np.array([1.0 if (a is not None and a == g) else 0.0 for a in parsed], dtype=np.float64)
        Xs.append(feats)
        Ys.append(y)
        metas.append(parsed)
    return np.stack(Xs), np.stack(Ys), metas, bank


def standardize_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float64), sd.astype(np.float64)


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def train_softmax(X, Y, lr=0.05, epochs=300, l2=1e-4, seed=SEED):
    rng = np.random.default_rng(seed)
    n, _, f = X.shape
    w = rng.normal(0.0, 0.01, size=f)
    b = 0.0
    idx = np.arange(n)
    base_lr = lr
    for ep in range(epochs):
        rng.shuffle(idx)
        lr_t = base_lr * (0.5 + 0.5 * np.cos(np.pi * ep / max(epochs, 1)))
        for start in range(0, n, 64):
            bi = idx[start:start + 64]
            xb, yb = X[bi], Y[bi]
            s = yb.sum(axis=1, keepdims=True)
            target = np.where(s > 0, yb / np.maximum(s, 1e-9), np.full_like(yb, 0.25))
            scores = xb @ w + b
            scores = scores - scores.max(axis=1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(axis=1, keepdims=True)
            grad = (p - target) / len(bi)
            w -= lr_t * (np.einsum('bof,bo->f', xb, grad) + l2 * w)
            b -= lr_t * grad.sum()
    return w.astype(np.float64), float(b)


def choose(scores, metas, majority_gate=0):
    choices = []
    for i, parsed in enumerate(metas):
        picked = None
        if majority_gate:
            counts = {}
            for a in parsed:
                if a is None:
                    continue
                counts[a] = counts.get(a, 0) + 1
            if counts:
                mode_n = max(counts.values())
                modes = [a for a, n in counts.items() if n == mode_n]
                if len(modes) == 1 and mode_n >= majority_gate:
                    ans = modes[0]
                    for j, a in enumerate(parsed):
                        if a == ans:
                            picked = j
                            break
        if picked is None:
            picked = int(np.argmax(scores[i]))
        choices.append(picked)
    return choices


def majority_baseline(bank, embeddings):
    from collections import Counter
    out = []
    for i, row in enumerate(bank):
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(a for a in parsed if a is not None)
        if counts:
            ans, n = counts.most_common(1)[0]
            if sum(1 for _, c in counts.items() if c == n) == 1:
                for j, a in enumerate(parsed):
                    if a == ans:
                        out.append(j)
                        break
                continue
        e = embeddings[i]
        out.append(int(np.argmin(np.linalg.norm(e - e.mean(0), axis=1))))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--pca-var', type=float, default=0.9,
                        help='Train-only PCA variance threshold for choosing k.')
    parser.add_argument('--pca-k-max', type=int, default=256)
    parser.add_argument('--majority-gate', type=int, default=0,
                        help='0 = always use scorer (default).')
    parser.add_argument('--ablate', choices=('full', 'no_pca', 'gate3'), default='full')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    emb_train = np.load(data_dir / 'candidates' / 'gsm8k_train.npz')['embeddings'].reshape(-1, 1024)

    if args.ablate == 'no_pca':
        pca_mean, pca_components, pca_info = None, None, {'k': 0, 'explained_variance': 0.0, 'var_thresh': args.pca_var}
    else:
        pca_mean, pca_components, pca_info = fit_pca(
            emb_train, var_thresh=args.pca_var, k_max=args.pca_k_max,
        )

    gate = 3 if args.ablate == 'gate3' else args.majority_gate

    Xtr, Ytr, _, _ = build_split('train', pca_mean, pca_components, data_dir)
    Xva, _, meta_va, bank_va = build_split('validation', pca_mean, pca_components, data_dir)
    assert Xtr.shape[-1] == BASE_DIM or pca_components is not None

    mu, sd = standardize_fit(Xtr)
    w, b = train_softmax(
        standardize_apply(Xtr, mu, sd), Ytr,
        lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed,
    )
    scores = standardize_apply(Xva, mu, sd) @ w + b
    choices = choose(scores, meta_va, majority_gate=gate)
    val = score_choices('validation', [r['id'] for r in bank_va], choices, data_dir)

    emb_va = np.load(data_dir / 'candidates' / 'gsm8k_validation.npz')['embeddings']
    emb_va = emb_va.reshape(len(bank_va), 4, 1024)
    base = score_choices(
        'validation', [r['id'] for r in bank_va],
        majority_baseline(bank_va, emb_va), data_dir,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        'weight': w,
        'bias': np.asarray([b], dtype=np.float64),
        'mean': mu,
        'std': sd,
    }
    if pca_mean is not None:
        # float32 PCA to shrink assets; applied in float64 at inference.
        payload['pca_mean'] = pca_mean.astype(np.float32)
        payload['pca_components'] = pca_components.astype(np.float32)
    np.savez_compressed(out / 'scorer.npz', **payload)

    meta = {
        'protocol': 'dase7506-mp2-open-hard-verifier-v5',
        'method': 'pca_listwise_softmax',
        'seed': args.seed,
        'lr': args.lr,
        'epochs': args.epochs,
        'l2': args.l2,
        'majority_gate': gate,
        'train_hard_only': False,
        'pca': pca_info,
        'feature_dim': int(Xtr.shape[-1]),
        'base_dim': BASE_DIM,
        'ablate': args.ablate,
        'validation_accuracy': val['accuracy'],
        'validation_correct': val['correct'],
        'validation_n': val['n'],
        'baseline_validation_accuracy': base['accuracy'],
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
