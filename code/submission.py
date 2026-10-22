"""Submission: train-fit PCA features + linear listwise scorer."""
from pathlib import Path

import numpy as np

from features import candidate_features, question_meta


def _load_checkpoint(checkpoint):
    path = Path(checkpoint)
    npz_path = path / 'scorer.npz' if path.is_dir() else path
    data = np.load(npz_path)
    meta_path = npz_path.parent / 'meta.json'
    majority_gate = 0
    if meta_path.exists():
        import json
        majority_gate = int(json.loads(meta_path.read_text()).get('majority_gate', 0))
    pca_mean = data['pca_mean'].astype(np.float64) if 'pca_mean' in data.files else None
    pca_components = (
        data['pca_components'].astype(np.float64) if 'pca_components' in data.files else None
    )
    return {
        'weight': data['weight'].astype(np.float64),
        'bias': float(np.asarray(data['bias']).reshape(-1)[0]),
        'mean': data['mean'].astype(np.float64),
        'std': data['std'].astype(np.float64),
        'pca_mean': pca_mean,
        'pca_components': pca_components,
        'majority_gate': majority_gate,
    }


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    Score each candidate with a linear listwise model over handcrafted
    agreement/embedding features plus train-fit PCA projections (global and
    per-question residual). Optional majority_gate from checkpoint meta
    (default 0 = always argmax score).
    """
    if checkpoint is None:
        raise ValueError('This method requires --checkpoint runs/model_bundle')
    ckpt = _load_checkpoint(checkpoint)
    w, b = ckpt['weight'], ckpt['bias']
    mu, sd = ckpt['mean'], ckpt['std']
    gate = ckpt['majority_gate']
    pca_mean, pca_components = ckpt['pca_mean'], ckpt['pca_components']
    embeddings = arrays['embeddings']
    choices = []
    for i, row in enumerate(bank):
        feats, parsed, mode_n, unique_mode = candidate_features(
            row, embeddings[i], pca_mean, pca_components,
        )
        if gate and unique_mode and mode_n >= gate:
            counts, _, _ = question_meta(parsed)
            mode_ans = next(a for a, n in counts.items() if n == mode_n)
            for j, a in enumerate(parsed):
                if a == mode_ans:
                    choices.append(j)
                    break
            continue
        x = (feats - mu) / sd
        scores = x @ w + b
        choices.append(int(np.argmax(scores)))
    return choices
