"""Shared candidate features for training and inference."""
from collections import Counter

import numpy as np

from common import parse_answer

# Base handcrafted dim (before PCA concatenations).
BASE_DIM = 17


def question_meta(parsed):
    counts = Counter(a for a in parsed if a is not None)
    if not counts:
        return counts, 0, False
    mode_n = max(counts.values())
    n_modes = sum(1 for v in counts.values() if v == mode_n)
    return counts, mode_n, n_modes == 1


def _handcrafted(row, embs, parsed, counts, mode_n, unique_mode):
    texts = [c['text'] for c in row['candidates']]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    unit = embs / norms
    mean = embs.mean(axis=0)
    mean_u = mean / (np.linalg.norm(mean) + 1e-9)
    dist = np.linalg.norm(embs - mean, axis=1)
    cos = unit @ mean_u
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    mean_pair = sim.mean(axis=1)
    max_pair = sim.max(axis=1)

    import re
    token_re = re.compile(r'[a-z0-9]+')
    q_tok = set(token_re.findall(str(row['question']).lower()))
    lens = np.array([len(t) for t in texts], dtype=np.float64)
    lens_n = lens / (lens.max() + 1e-9)
    overlap = np.array(
        [len(q_tok & set(token_re.findall(str(t).lower()))) / (len(q_tok) + 1e-9) for t in texts],
        dtype=np.float64,
    )

    feats = []
    for j in range(4):
        a = parsed[j]
        cnt = counts.get(a, 0) if a is not None else 0
        is_mode = 1.0 if (a is not None and unique_mode and cnt == mode_n) else 0.0
        is_tied = 1.0 if (a is not None and (not unique_mode) and counts and cnt == mode_n) else 0.0
        feats.append([
            1.0 if a is not None else 0.0,
            cnt / 4.0,
            is_mode,
            is_tied,
            1.0 if cnt == 1 and a is not None else 0.0,
            1.0 if unique_mode else 0.0,
            mode_n / 4.0 if counts else 0.0,
            -float(dist[j]),
            float(cos[j]),
            float(mean_pair[j]),
            float(max_pair[j]),
            float(lens_n[j]),
            float(overlap[j]),
            float(np.log1p(lens[j]) / 10.0),
            is_mode * float(cos[j]),
            (cnt / 4.0) * float(cos[j]),
            (1.0 - is_mode) * float(cos[j]),
        ])
    return np.asarray(feats, dtype=np.float64), mean


def candidate_features(row, embs, pca_mean=None, pca_components=None):
    """Return [4, F] features, parsed answers, mode_n, unique_mode.

    If PCA arrays are given, append global PCA(emb) and local PCA(emb-mean).
    """
    texts = [c['text'] for c in row['candidates']]
    parsed = [parse_answer(t) for t in texts]
    counts, mode_n, unique_mode = question_meta(parsed)
    base, qmean = _handcrafted(row, embs, parsed, counts, mode_n, unique_mode)
    if pca_mean is None or pca_components is None:
        return base, parsed, mode_n, unique_mode
    global_proj = (embs - pca_mean) @ pca_components
    local_proj = (embs - qmean) @ pca_components
    feats = np.concatenate([base, global_proj, local_proj], axis=1)
    return feats.astype(np.float64), parsed, mode_n, unique_mode


def fit_pca(embeddings_flat, var_thresh=0.9, k_max=256):
    """Fit PCA on [M, 1024] train candidate embeddings. Train-only."""
    mu = embeddings_flat.mean(axis=0)
    X = embeddings_flat - mu
    cov = (X.T @ X) / max(len(X), 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    total = float(evals.sum()) + 1e-12
    cume = np.cumsum(evals) / total
    k = int(np.searchsorted(cume, var_thresh) + 1)
    k = max(8, min(k, k_max, evecs.shape[1]))
    return mu.astype(np.float64), evecs[:, :k].astype(np.float64), {
        'k': k,
        'explained_variance': float(cume[k - 1]),
        'var_thresh': var_thresh,
    }
