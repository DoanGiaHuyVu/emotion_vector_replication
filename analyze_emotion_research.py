#!/usr/bin/env python3
"""
Research-grade analysis for emotion-vector experiments.

Designed for two kinds of outputs:
  1) class-level emotion directions: emotion_vectors.npz
  2) contextual dialogue turn activations: turn_vectors.npy + turn_metadata.jsonl

Main goals:
- reproduce the geometry analysis in Sofroniew et al. (2026) more faithfully;
- avoid arbitrary similarity / PCA thresholds;
- quantify uncertainty and held-out generalization;
- detect speaker/topic confounding;
- optionally evaluate paper-style linear projections onto independently extracted emotion directions;
- compare representational geometry across model families without comparing incompatible hidden dimensions directly.

Important limitations:
- Layer sweeps require vectors extracted at multiple layers; a single saved target layer cannot reconstruct them.
- Token-level activation trajectories require token-level activations; mean-pooled turn vectors cannot reconstruct them.
- Causal steering requires running the model with interventions; it cannot be inferred from saved vectors alone.

Example single-model run:
  uv run python analyze_emotion_research.py \
      --turn-dir results_qwen3_1.7b_full \
      --emotion-vectors results_qwen3_emotions/emotion_vectors.npz \
      --name qwen3_1.7b \
      --output-dir analysis/qwen3_1.7b

Example multi-model run with a manifest:
  uv run python analyze_emotion_research.py \
      --manifest analysis_manifest.json \
      --output-dir analysis/all_models

Manifest format:
{
  "runs": [
    {
      "name": "qwen3_1.7b",
      "turn_dir": "results_qwen3_1.7b_full",
      "emotion_vectors": "results_qwen3_emotions/emotion_vectors.npz"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Matplotlib is optional. All numeric analyses still run if unavailable.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


CANONICAL_ALIASES = {
    "anger": "anger", "angry": "anger", "mad": "anger", "furious": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "afraid": "fear", "scared": "fear", "frightened": "fear",
    "joy": "joy", "joyful": "joy", "happy": "joy",
    "neutral": "neutral",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise", "astonished": "surprise",
    "peaceful": "peaceful", "peace": "peaceful", "calm": "peaceful",
    "powerful": "powerful", "power": "powerful", "dominant": "powerful",
}

ERC_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EXTENDED_CLASSES = ERC_CLASSES + ["peaceful", "powerful"]
# Backward-compatible alias.
DEFAULT_CLASSES = ERC_CLASSES


def canonical_emotion(x: str) -> str:
    y = str(x).strip().lower().replace("_", " ")
    return CANONICAL_ALIASES.get(y, y)


def safe_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def unit_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = safe_float_array(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_matrix_rows(x: np.ndarray) -> np.ndarray:
    z = unit_rows(x)
    return z @ z.T


def upper_triangle_values(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: Sequence[float]) -> np.ndarray:
    """Average ranks for ties; minimal scipy-free Spearman helper."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def binary_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney form of ROC AUC; returns NaN if one class is absent."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    auc = (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled_num = (len(a) - 1) * va + (len(b) - 1) * vb
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_metric_by_dialogue(
    y_true: Sequence[str],
    score_matrix: np.ndarray,
    dialogue_ids: Sequence[object],
    classes: Sequence[str],
    bootstraps: int = 200,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Dialogue-resampled CI for one-vs-rest AUROC of independent probe activations."""
    y = np.asarray(y_true, dtype=object)
    dial = np.asarray(dialogue_ids, dtype=object)
    uniq = np.array(sorted(set(dial.tolist()), key=lambda x: str(x)), dtype=object)
    if len(uniq) < 2:
        return {}
    group_rows = {g: np.where(dial == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    values = {c: [] for c in classes}
    for b in range(bootstraps):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([group_rows[g] for g in sampled])
        yy = y[idx]
        ss = score_matrix[idx]
        for j, c in enumerate(classes):
            auc = binary_auc((yy == c).astype(np.int8), ss[:, j])
            if np.isfinite(auc):
                values[c].append(auc)
    out = {}
    for c, vals in values.items():
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            out[c] = {
                "low": float(np.quantile(a, 0.025)),
                "median": float(np.quantile(a, 0.5)),
                "high": float(np.quantile(a, 0.975)),
            }
    return out


def mean_and_std_stream(vectors: np.ndarray, indices: np.ndarray, batch: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    if len(indices) == 0:
        raise ValueError("No indices supplied")
    d = vectors.shape[1]
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros(d, dtype=np.float64)
    n = 0
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = np.asarray(vectors[idx], dtype=np.float32)
        s += x.sum(axis=0, dtype=np.float64)
        ss += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n += len(idx)
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def class_centroids_stream(
    vectors: np.ndarray,
    indices: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    center: Optional[np.ndarray] = None,
    batch: int = 1024,
) -> Tuple[np.ndarray, Dict[str, int]]:
    d = vectors.shape[1]
    sums = {c: np.zeros(d, dtype=np.float64) for c in classes}
    counts = {c: 0 for c in classes}
    label_arr = np.asarray(labels, dtype=object)
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = np.asarray(vectors[idx], dtype=np.float32)
        if center is not None:
            x = x - center[None, :]
        labs = label_arr[idx]
        for c in classes:
            mask = labs == c
            if np.any(mask):
                sums[c] += x[mask].sum(axis=0, dtype=np.float64)
                counts[c] += int(mask.sum())
    centroids = []
    for c in classes:
        if counts[c] == 0:
            centroids.append(np.full(d, np.nan, dtype=np.float32))
        else:
            centroids.append((sums[c] / counts[c]).astype(np.float32))
    return np.stack(centroids), counts


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        if a in pos and b in pos:
            cm[pos[a], pos[b]] += 1
    return cm


def metrics_from_cm(cm: np.ndarray, classes: Sequence[str]) -> Dict[str, object]:
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else float("nan")
    recalls = {}
    f1s = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        rec = float(tp / (tp + fn)) if tp + fn else float("nan")
        prec = float(tp / (tp + fp)) if tp + fp else float("nan")
        f1 = float(2 * prec * rec / (prec + rec)) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else float("nan")
        recalls[c] = rec
        f1s[c] = f1
    finite_rec = [v for v in recalls.values() if np.isfinite(v)]
    finite_f1 = [v for v in f1s.values() if np.isfinite(v)]
    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(finite_rec)) if finite_rec else float("nan"),
        "macro_f1": float(np.mean(finite_f1)) if finite_f1 else float("nan"),
        "recall": recalls,
        "f1": f1s,
        "n": total,
    }


def grouped_kfold_assignments(groups: Sequence[object], k: int, seed: int) -> np.ndarray:
    groups = np.asarray(groups, dtype=object)
    uniq = np.array(sorted(set(groups.tolist()), key=lambda x: str(x)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    mapping = {g: i % k for i, g in enumerate(uniq)}
    return np.array([mapping[g] for g in groups], dtype=np.int16)


def grouped_centroid_cv(
    vectors: np.ndarray,
    labels: Sequence[str],
    indices: np.ndarray,
    groups: Sequence[object],
    classes: Sequence[str],
    k: int = 5,
    seed: int = 42,
    batch: int = 512,
) -> Dict[str, object]:
    if len(indices) == 0:
        return {"error": "no examples"}
    group_arr = np.asarray(groups, dtype=object)
    folds = grouped_kfold_assignments(group_arr[indices], k=k, seed=seed)
    y_true_all, y_pred_all = [], []
    labels_arr = np.asarray(labels, dtype=object)

    for fold in range(k):
        test_idx = indices[folds == fold]
        train_idx = indices[folds != fold]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        train_mean, _ = mean_and_std_stream(vectors, train_idx, batch=batch)
        cents, counts = class_centroids_stream(vectors, train_idx, labels, classes, center=train_mean, batch=batch)
        valid_cls = [i for i, c in enumerate(classes) if counts[c] > 0 and np.isfinite(cents[i]).all()]
        if not valid_cls:
            continue
        cmat = unit_rows(cents[valid_cls])
        valid_names = [classes[i] for i in valid_cls]

        for start in range(0, len(test_idx), batch):
            idx = test_idx[start:start + batch]
            x = np.asarray(vectors[idx], dtype=np.float32) - train_mean[None, :]
            x = unit_rows(x)
            scores = x @ cmat.T
            pred = np.argmax(scores, axis=1)
            y_pred_all.extend(valid_names[i] for i in pred)
            y_true_all.extend(labels_arr[idx].tolist())

    cm = confusion_matrix(y_true_all, y_pred_all, classes)
    out = metrics_from_cm(cm, classes)
    out["confusion_matrix"] = cm.tolist()
    return out


def sampled_pair_separation(
    vectors: np.ndarray,
    labels: Sequence[str],
    indices: np.ndarray,
    dialogue_ids: Optional[Sequence[object]],
    center: np.ndarray,
    n_pairs: int = 50000,
    seed: int = 42,
    pair_batch: int = 512,
) -> Dict[str, float]:
    if len(indices) < 2:
        return {"same": float("nan"), "different": float("nan"), "gap": float("nan"), "same_n": 0, "different_n": 0}
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=object)
    dial = np.asarray(dialogue_ids, dtype=object) if dialogue_ids is not None else None
    same_vals, diff_vals = [], []
    need = n_pairs
    attempts = 0
    max_attempts = n_pairs * 20

    while need > 0 and attempts < max_attempts:
        b = min(pair_batch, max(need * 2, 256))
        ia = rng.choice(indices, size=b, replace=True)
        ib = rng.choice(indices, size=b, replace=True)
        attempts += b
        good = ia != ib
        if dial is not None:
            good &= dial[ia] != dial[ib]
        ia, ib = ia[good], ib[good]
        if len(ia) == 0:
            continue
        xa = unit_rows(np.asarray(vectors[ia], dtype=np.float32) - center[None, :])
        xb = unit_rows(np.asarray(vectors[ib], dtype=np.float32) - center[None, :])
        sims = np.sum(xa * xb, axis=1)
        same = labels_arr[ia] == labels_arr[ib]
        same_vals.extend(sims[same].tolist())
        diff_vals.extend(sims[~same].tolist())
        need -= len(sims)

    same_arr = np.asarray(same_vals[:n_pairs], dtype=np.float64)
    diff_arr = np.asarray(diff_vals[:n_pairs], dtype=np.float64)
    same_mean = float(np.mean(same_arr)) if len(same_arr) else float("nan")
    diff_mean = float(np.mean(diff_arr)) if len(diff_arr) else float("nan")
    return {
        "same": same_mean,
        "different": diff_mean,
        "gap": same_mean - diff_mean if np.isfinite(same_mean) and np.isfinite(diff_mean) else float("nan"),
        "same_n": int(len(same_arr)),
        "different_n": int(len(diff_arr)),
    }


def bootstrap_pair_gap(
    vectors: np.ndarray,
    labels: Sequence[str],
    indices: np.ndarray,
    dialogue_ids: Sequence[object],
    center: np.ndarray,
    bootstraps: int = 200,
    pairs_per_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Dialogue-resampled uncertainty estimate for same-vs-different cosine gap."""
    dial = np.asarray(dialogue_ids, dtype=object)
    uniq = np.array(sorted(set(dial[indices].tolist()), key=lambda x: str(x)), dtype=object)
    if len(uniq) < 2:
        return {"low": float("nan"), "median": float("nan"), "high": float("nan")}
    by_group = {g: indices[dial[indices] == g] for g in uniq}
    rng = np.random.default_rng(seed)
    gaps = []
    for b in range(bootstraps):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        # Keep repeated groups as repeated examples by concatenating their row indices.
        idx = np.concatenate([by_group[g] for g in sampled])
        r = sampled_pair_separation(
            vectors, labels, idx, None, center,
            n_pairs=pairs_per_bootstrap,
            seed=seed + b + 1,
            pair_batch=256,
        )
        if np.isfinite(r["gap"]):
            gaps.append(r["gap"])
    if not gaps:
        return {"low": float("nan"), "median": float("nan"), "high": float("nan")}
    a = np.asarray(gaps)
    return {
        "low": float(np.quantile(a, 0.025)),
        "median": float(np.quantile(a, 0.5)),
        "high": float(np.quantile(a, 0.975)),
    }


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def simple_tokens(text: str) -> List[str]:
    return WORD_RE.findall(str(text).lower())


def grouped_lexical_nb_cv(
    meta: List[dict],
    classes: Sequence[str],
    indices: np.ndarray,
    group_key: str = "dialogue_id",
    k: int = 5,
    seed: int = 42,
    max_vocab: int = 5000,
    alpha: float = 1.0,
) -> Dict[str, object]:
    """
    Surface-form control using a simple multinomial Naive Bayes classifier.

    This is deliberately simple: if it performs very well, especially on neutral,
    part of the task may be recoverable from lexical/style cues alone.
    """
    if len(indices) == 0:
        return {"error": "no examples"}

    labels = np.asarray(
        [canonical_emotion(m.get("emotion", "")) for m in meta],
        dtype=object,
    )
    groups = np.asarray(
        [m.get(group_key, m.get("dialogue_id", i)) for i, m in enumerate(meta)],
        dtype=object,
    )
    texts = [m.get("utterance", "") for m in meta]
    folds = grouped_kfold_assignments(groups[indices], k=k, seed=seed)

    y_true_all, y_pred_all = [], []
    for fold in range(k):
        test_idx = indices[folds == fold]
        train_idx = indices[folds != fold]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        global_counts = Counter()
        class_doc_counts = Counter()
        class_word_counts = {c: Counter() for c in classes}
        class_totals = Counter()

        for ii in train_idx:
            c = labels[ii]
            if c not in class_word_counts:
                continue
            toks = simple_tokens(texts[ii])
            global_counts.update(toks)
            class_word_counts[c].update(toks)
            class_totals[c] += len(toks)
            class_doc_counts[c] += 1

        vocab = [w for w, _ in global_counts.most_common(max_vocab)]
        vocab_set = set(vocab)
        V = max(len(vocab), 1)
        n_docs = max(sum(class_doc_counts.values()), 1)

        logprob = {}
        priors = {}
        for c in classes:
            priors[c] = math.log((class_doc_counts[c] + 1.0) / (n_docs + len(classes)))
            denom = class_totals[c] + alpha * V
            logprob[c] = {
                w: math.log((class_word_counts[c][w] + alpha) / denom)
                for w in vocab
            }
            logprob[c]["__UNK__"] = math.log(alpha / denom)

        for ii in test_idx:
            counts = Counter(simple_tokens(texts[ii]))
            best_c, best_s = None, -float("inf")
            for c in classes:
                sc = priors[c]
                lp = logprob[c]
                unk = lp["__UNK__"]
                for w, cnt in counts.items():
                    sc += cnt * (lp[w] if w in vocab_set else unk)
                if sc > best_s:
                    best_s, best_c = sc, c
            y_true_all.append(labels[ii])
            y_pred_all.append(best_c)

    cm = confusion_matrix(y_true_all, y_pred_all, classes)
    out = metrics_from_cm(cm, classes)
    out["confusion_matrix"] = cm.tolist()
    out["method"] = "multinomial_naive_bayes_unigram_surface_control"
    out["max_vocab"] = int(max_vocab)
    out["group_key"] = group_key
    out["interpretation"] = (
        "This is a surface-form control, not a mechanistic probe. Strong performance "
        "means the synthetic text itself exposes lexical/style cues that may also make "
        "internal representations easy to classify."
    )
    return out


def dataset_lexical_controls(
    meta: List[dict],
    output_dir: Path,
    cv_folds: int,
    seed: int,
) -> Dict[str, object]:
    labels = [canonical_emotion(m.get("emotion", "")) for m in meta]
    speakers = [str(m.get("speaker", "unknown")).lower() for m in meta]
    present = set(labels)
    spaces = {
        "erc7": [c for c in ERC_CLASSES if c in present],
        "extended9": [c for c in EXTENDED_CLASSES if c in present],
    }
    result = {}
    for space_name, classes in spaces.items():
        if len(classes) < 2:
            continue
        class_set = set(classes)
        base = np.array([i for i, lab in enumerate(labels) if lab in class_set], dtype=np.int64)
        result[space_name] = {}
        subsets = {"all": base}
        for sp in sorted(set(speakers)):
            subsets[sp] = np.array([i for i in base if speakers[i] == sp], dtype=np.int64)
        for subset_name, idx in subsets.items():
            if len(idx) < 20:
                continue
            result[space_name][subset_name] = {
                "dialogue_group_cv": grouped_lexical_nb_cv(
                    meta, classes, idx, group_key="dialogue_id",
                    k=cv_folds, seed=seed,
                ),
                "topic_group_cv": grouped_lexical_nb_cv(
                    meta, classes, idx, group_key="topic_idx",
                    k=cv_folds, seed=seed,
                ),
            }
    (output_dir / "dataset_lexical_control.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def pca_from_rows(x: np.ndarray) -> Dict[str, object]:
    x = safe_float_array(x)
    xc = x - x.mean(axis=0, keepdims=True)
    # n_emotions is tiny, so SVD is cheap even for wide residual dimensions.
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratio = var / max(var.sum(), 1e-12)
    scores = xc @ vt.T
    return {
        "scores": scores,
        "components": vt,
        "explained_variance_ratio": ratio,
    }


def pca_variance_null(n: int, d: int, sims: int = 200, seed: int = 42) -> Dict[str, float]:
    """Monte-Carlo null for top-2 PCA concentration with n points in d dimensions."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(sims):
        x = rng.standard_normal((n, d), dtype=np.float32)
        r = pca_from_rows(x)["explained_variance_ratio"]
        vals.append(float(np.sum(r[: min(2, len(r))])))
    a = np.asarray(vals)
    return {
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
    }


def load_affect_ratings(path: Optional[Path]) -> Dict[str, Tuple[float, float]]:
    if path is None:
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        required = {"emotion", "valence", "arousal"}
        if not required.issubset(rd.fieldnames or []):
            raise ValueError(f"ratings CSV must have columns {sorted(required)}")
        for row in rd:
            out[canonical_emotion(row["emotion"])] = (float(row["valence"]), float(row["arousal"]))
    return out


def plot_heatmap(matrix: np.ndarray, labels: Sequence[str], title: str, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
    im = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(scores: np.ndarray, labels: Sequence[str], var: np.ndarray, title: str, path: Path) -> None:
    if plt is None or scores.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores[:, 0], scores[:, 1])
    for i, lab in enumerate(labels):
        ax.annotate(lab, (scores[i, 0], scores[i, 1]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_emotion_vectors(
    path: Path,
    output_dir: Path,
    ratings: Dict[str, Tuple[float, float]],
    pca_null_sims: int,
) -> Dict[str, object]:
    data = np.load(path)
    raw = {canonical_emotion(k): safe_float_array(data[k]) for k in data.files}
    labels = sorted(raw.keys())
    x = np.stack([raw[k] for k in labels])

    finite = bool(np.isfinite(x).all())
    norms = np.linalg.norm(x, axis=1)
    cos = cosine_matrix_rows(x)
    pca = pca_from_rows(x)
    ratio = pca["explained_variance_ratio"]
    top2 = float(np.sum(ratio[: min(2, len(ratio))]))
    null = pca_variance_null(len(labels), x.shape[1], sims=pca_null_sims)

    result = {
        "path": str(path),
        "emotions": labels,
        "n_emotions": len(labels),
        "dimension": int(x.shape[1]),
        "finite": finite,
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "average_pairwise_cosine": float(upper_triangle_values(cos).mean()) if len(labels) > 1 else float("nan"),
        "cosine_matrix": cos.tolist(),
        "explained_variance_ratio": ratio.tolist(),
        "top2_variance": top2,
        "top2_random_null": null,
        "top2_above_random_p95": bool(top2 > null["p95"]),
        "geometry_warning": (
            "With fewer than ~20 emotion concepts, PCA/circumplex claims are exploratory. "
            "Use independent human valence/arousal ratings and a larger emotion vocabulary "
            "for a paper-style geometry replication."
            if len(labels) < 20 else None
        ),
    }

    if ratings:
        common = [e for e in labels if e in ratings]
        result["human_ratings_overlap"] = len(common)
        if len(common) >= 4:
            idx = [labels.index(e) for e in common]
            val = [ratings[e][0] for e in common]
            aro = [ratings[e][1] for e in common]
            pcs = pca["scores"][idx]
            max_pc = min(3, pcs.shape[1])
            val_corr = []
            aro_corr = []
            for j in range(max_pc):
                val_corr.append({
                    "pc": j + 1,
                    "pearson": pearson(pcs[:, j], val),
                    "spearman": spearman(pcs[:, j], val),
                })
                aro_corr.append({
                    "pc": j + 1,
                    "pearson": pearson(pcs[:, j], aro),
                    "spearman": spearman(pcs[:, j], aro),
                })
            result["valence_correlations"] = val_corr
            result["arousal_correlations"] = aro_corr

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "emotion_cosine_matrix.csv", cos, delimiter=",")
    np.save(output_dir / "emotion_direction_geometry.npy", cos)
    (output_dir / "emotion_direction_classes.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    plot_heatmap(cos, labels, "Emotion-vector cosine similarity", output_dir / "emotion_cosine_heatmap.png")
    plot_pca(pca["scores"], labels, ratio, "Emotion-vector PCA", output_dir / "emotion_pca.png")

    with (output_dir / "emotion_geometry.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def load_metadata(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def metadata_fields(meta: List[dict]) -> Tuple[List[str], List[str], List[object], List[object]]:
    labels, speakers, dialogue_ids, topic_ids = [], [], [], []
    for i, m in enumerate(meta):
        lab = canonical_emotion(m.get("emotion", ""))
        labels.append(lab)
        speakers.append(str(m.get("speaker", "unknown")).lower())
        dialogue_ids.append(m.get("dialogue_id", i))
        topic_ids.append(m.get("topic_idx", m.get("topic", m.get("dialogue_id", i))))
    return labels, speakers, dialogue_ids, topic_ids


def probe_scores_batch(x: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Paper-style linear projection: h dot v_hat, not cosine(h, v)."""
    return safe_float_array(x) @ unit_rows(dirs).T


def evaluate_emotion_probes(
    vectors: np.ndarray,
    meta: List[dict],
    emotion_vector_path: Path,
    indices: np.ndarray,
    classes: Sequence[str],
    batch: int = 512,
    bootstraps: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Evaluate independently extracted emotion directions on held-out dialogue turns.

    Primary diagnostics:
      - continuous paper-style linear probe activation h @ v_hat;
      - one-vs-rest AUROC per direction (does not require cross-direction calibration);
      - effect size of matched-vs-unmatched activations;
      - true-class margin over the strongest competing direction.

    Argmax classification is retained only as a secondary application diagnostic.
    """
    data = np.load(emotion_vector_path)
    direction_map = {canonical_emotion(k): safe_float_array(data[k]) for k in data.files}
    common = [c for c in classes if c in direction_map]
    missing = [c for c in classes if c not in direction_map]
    if len(common) < 2:
        return {
            "error": "fewer than two overlapping emotion directions",
            "overlap": common,
            "missing_directions": missing,
        }

    dirs = np.stack([direction_map[c] for c in common])
    if dirs.shape[1] != vectors.shape[1]:
        return {
            "error": "hidden dimension mismatch",
            "turn_dim": int(vectors.shape[1]),
            "emotion_dim": int(dirs.shape[1]),
            "overlap": common,
        }

    labels_all = np.asarray([canonical_emotion(m.get("emotion", "")) for m in meta], dtype=object)
    valid_idx = np.array([i for i in indices if labels_all[i] in common], dtype=np.int64)
    if len(valid_idx) == 0:
        return {"error": "no indexed turns have labels overlapping the supplied directions"}

    # Compute scores for all valid rows in deterministic index order.
    score_chunks = []
    for start in range(0, len(valid_idx), batch):
        idx = valid_idx[start:start + batch]
        x = np.asarray(vectors[idx], dtype=np.float32)
        score_chunks.append(probe_scores_batch(x, dirs))
    scores = np.concatenate(score_chunks, axis=0)
    y = labels_all[valid_idx]
    dialogue_ids = np.asarray(
        [meta[i].get("dialogue_id", i) for i in valid_idx],
        dtype=object,
    )

    # Secondary top-1 diagnostic. Raw direction projections need not be perfectly
    # calibrated across concepts, so AUROC/effect-size results are primary.
    pred_idx = np.argmax(scores, axis=1)
    pred = [common[i] for i in pred_idx]
    cm = confusion_matrix(y.tolist(), pred, common)
    metrics = metrics_from_cm(cm, common)
    metrics["classes"] = common
    metrics["missing_directions"] = missing
    metrics["confusion_matrix"] = cm.tolist()
    metrics["note"] = (
        "Argmax classification is secondary. The paper-style quantity is the "
        "continuous linear projection onto each independently extracted direction."
    )

    per_class = {}
    for j, c in enumerate(common):
        is_pos = y == c
        pos_scores = scores[is_pos, j]
        neg_scores = scores[~is_pos, j]
        auc = binary_auc(is_pos.astype(np.int8), scores[:, j])
        per_class[c] = {
            "n_positive": int(is_pos.sum()),
            "n_negative": int((~is_pos).sum()),
            "mean_activation_positive": float(np.mean(pos_scores)) if len(pos_scores) else float("nan"),
            "mean_activation_negative": float(np.mean(neg_scores)) if len(neg_scores) else float("nan"),
            "activation_gap": (
                float(np.mean(pos_scores) - np.mean(neg_scores))
                if len(pos_scores) and len(neg_scores) else float("nan")
            ),
            "cohens_d": cohens_d(pos_scores, neg_scores),
            "auroc": auc,
        }

    aucs = [v["auroc"] for v in per_class.values() if np.isfinite(v["auroc"])]
    metrics["macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["per_class_probe"] = per_class
    metrics["auroc_bootstrap_95ci_by_dialogue"] = bootstrap_metric_by_dialogue(
        y, scores, dialogue_ids, common, bootstraps=bootstraps, seed=seed
    )

    # True-class activation margin: score(true) - best score(other).
    class_pos = {c: j for j, c in enumerate(common)}
    margins = []
    for row, lab in enumerate(y):
        j = class_pos[lab]
        true_score = scores[row, j]
        if len(common) > 1:
            other = np.delete(scores[row], j)
            margins.append(float(true_score - np.max(other)))
    if margins:
        marr = np.asarray(margins, dtype=np.float64)
        metrics["true_class_margin"] = {
            "mean": float(marr.mean()),
            "median": float(np.median(marr)),
            "positive_fraction": float(np.mean(marr > 0)),
        }

    # Mean activation matrix by true label. This is directly useful for checking
    # selectivity vs psychologically related confusions.
    mean_activation = {}
    for lab in common:
        mask = y == lab
        if not np.any(mask):
            continue
        mean_activation[lab] = {
            c: float(np.mean(scores[mask, j])) for j, c in enumerate(common)
        }
    metrics["mean_probe_activation_by_true_label"] = mean_activation

    return metrics


def finite_row_mask_memmap(vectors: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Chunked finite-row check that is safe for memory-mapped large arrays."""
    mask = np.ones(vectors.shape[0], dtype=bool)
    for start in range(0, vectors.shape[0], batch):
        end = min(vectors.shape[0], start + batch)
        x = np.asarray(vectors[start:end], dtype=np.float32)
        mask[start:end] = np.isfinite(x).all(axis=1)
    return mask


def analyze_label_space(
    vectors: np.ndarray,
    labels: Sequence[str],
    speakers: Sequence[str],
    dialogue_ids: Sequence[object],
    topic_ids: Sequence[object],
    classes: Sequence[str],
    eligible_mask: np.ndarray,
    pair_samples: int,
    bootstraps: int,
    cv_folds: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Analyze one explicit label space (e.g. ERC-7 or extended 9-state)."""
    class_set = set(classes)
    indices_all = np.array(
        [i for i, lab in enumerate(labels) if eligible_mask[i] and lab in class_set],
        dtype=np.int64,
    )
    subsets = {"all": indices_all}
    for sp in sorted(set(speakers)):
        subsets[sp] = np.array(
            [i for i in indices_all if speakers[i] == sp],
            dtype=np.int64,
        )

    subset_results = {}
    geometries = {}
    for subset_name, idx in subsets.items():
        if len(idx) < 10:
            continue
        center, _ = mean_and_std_stream(vectors, idx)
        sep = sampled_pair_separation(
            vectors, labels, idx, dialogue_ids, center,
            n_pairs=pair_samples, seed=seed,
        )
        ci = bootstrap_pair_gap(
            vectors, labels, idx, dialogue_ids, center,
            bootstraps=bootstraps,
            pairs_per_bootstrap=min(5000, pair_samples),
            seed=seed,
        )
        dialogue_cv = grouped_centroid_cv(
            vectors, labels, idx, dialogue_ids, classes,
            k=cv_folds, seed=seed,
        )
        unique_topics = len(set(np.asarray(topic_ids, dtype=object)[idx].tolist()))
        topic_cv = grouped_centroid_cv(
            vectors, labels, idx, topic_ids, classes,
            k=min(cv_folds, max(2, unique_topics)),
            seed=seed,
        )
        cents, counts = class_centroids_stream(
            vectors, idx, labels, classes, center=center
        )
        valid = [i for i, c in enumerate(classes) if counts[c] > 0]
        centroid_cos = np.full((len(classes), len(classes)), np.nan, dtype=np.float32)
        if len(valid) >= 2:
            cc = cosine_matrix_rows(cents[valid])
            for a, ia in enumerate(valid):
                for b, ib in enumerate(valid):
                    centroid_cos[ia, ib] = cc[a, b]

        subset_results[subset_name] = {
            "n": int(len(idx)),
            "same_vs_different_emotion": sep,
            "gap_bootstrap_95ci": ci,
            "dialogue_group_cv": dialogue_cv,
            "topic_group_cv": topic_cv,
            "centroid_counts": counts,
            "centroid_cosine_matrix": centroid_cos.tolist(),
        }
        geometries[subset_name] = centroid_cos

    return {
        "classes": list(classes),
        "n": int(len(indices_all)),
        "coverage_fraction_of_finite_rows": float(len(indices_all) / max(int(eligible_mask.sum()), 1)),
        "subsets": subset_results,
    }, geometries


def analyze_turn_dir(
    turn_dir: Path,
    output_dir: Path,
    name: str,
    emotion_vectors: Optional[Path],
    pair_samples: int,
    bootstraps: int,
    cv_folds: int,
    seed: int,
) -> Dict[str, object]:
    vec_path = turn_dir / "turn_vectors.npy"
    meta_path = turn_dir / "turn_metadata.jsonl"
    config_path = turn_dir / "run_config.json"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Expected {vec_path} and {meta_path}")

    vectors = np.load(vec_path, mmap_mode="r")
    meta = load_metadata(meta_path)
    if len(meta) != vectors.shape[0]:
        raise ValueError(f"metadata rows ({len(meta)}) != vector rows ({vectors.shape[0]})")

    labels, speakers, dialogue_ids, topic_ids = metadata_fields(meta)
    finite_mask = finite_row_mask_memmap(vectors)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    emotion_provenance = {
        "status": "not_supplied" if emotion_vectors is None else "unverified_no_sidecar",
        "checked_sidecar": None,
    }
    emotion_sidecar_config = None
    if emotion_vectors is not None and emotion_vectors.exists():
        # Prefer the explicit emotion-vector sidecar over a generic run_config.json.
        for sidecar_name in ("emotion_vector_config.json", "run_config.json"):
            sidecar = emotion_vectors.parent / sidecar_name
            if sidecar.exists():
                ev_cfg = json.loads(sidecar.read_text(encoding="utf-8"))
                emotion_sidecar_config = ev_cfg
                emotion_provenance = {
                    "status": "verified_from_sidecar",
                    "checked_sidecar": str(sidecar),
                    "config": ev_cfg,
                }
                mismatches = []
                for key in ("model", "target_layer", "hidden_size"):
                    if key in config and key in ev_cfg and config[key] != ev_cfg[key]:
                        mismatches.append(
                            f"{key}: turn={config[key]!r} emotion={ev_cfg[key]!r}"
                        )
                if mismatches:
                    raise ValueError(
                        f"{name}: emotion-vector provenance mismatch; "
                        + "; ".join(mismatches)
                    )
                break

    # Stable metadata fingerprint helps confirm that cross-model runs use the same examples.
    h = hashlib.sha256()
    for m in meta:
        key = (
            m.get("dialogue_id"),
            m.get("turn_idx"),
            str(m.get("speaker", "")).lower(),
            canonical_emotion(m.get("emotion", "")),
            m.get("topic_idx"),
        )
        h.update(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    metadata_fp = h.hexdigest()

    # If directions were derived from a training split of these same dialogue vectors,
    # evaluate them ONLY on the saved held-out rows. This is essential for ERC: using
    # the same labeled turns to construct and test class directions would leak labels.
    independent_probe_eval_indices = None
    independent_probe_scope = {
        "mode": "all_eligible_rows",
        "reason": "emotion directions are external or no held-out split was declared",
    }
    if emotion_vectors is not None and emotion_sidecar_config is not None:
        src_type = emotion_sidecar_config.get("source_type")
        if src_type == "dialogue_turn_vectors_train_split":
            sidecar_fp = emotion_sidecar_config.get("metadata_fingerprint_sha256")
            if sidecar_fp and sidecar_fp != metadata_fp:
                raise ValueError(
                    f"{name}: emotion directions were derived from different turn metadata "
                    f"(fingerprint mismatch)."
                )
            eval_name = emotion_sidecar_config.get("eval_indices_file", "eval_indices.npy")
            eval_path = emotion_vectors.parent / eval_name
            if not eval_path.exists():
                raise FileNotFoundError(
                    f"{name}: dialogue-derived emotion directions require held-out eval indices, "
                    f"but {eval_path} does not exist."
                )
            independent_probe_eval_indices = np.asarray(np.load(eval_path), dtype=np.int64)
            if independent_probe_eval_indices.ndim != 1:
                raise ValueError(f"{name}: {eval_path} must be a 1-D index array")
            if len(independent_probe_eval_indices) == 0:
                raise ValueError(f"{name}: held-out eval index array is empty")
            if independent_probe_eval_indices.min() < 0 or independent_probe_eval_indices.max() >= len(meta):
                raise ValueError(f"{name}: held-out eval indices are out of bounds")
            if len(np.unique(independent_probe_eval_indices)) != len(independent_probe_eval_indices):
                raise ValueError(f"{name}: held-out eval indices contain duplicates")

            # Optional leakage audit: train and eval row-index files must be disjoint.
            train_name = emotion_sidecar_config.get("train_indices_file", "train_indices.npy")
            train_path = emotion_vectors.parent / train_name
            overlap_n = None
            if train_path.exists():
                train_idx_audit = np.asarray(np.load(train_path), dtype=np.int64)
                overlap_n = int(np.intersect1d(train_idx_audit, independent_probe_eval_indices).size)
                if overlap_n:
                    raise ValueError(
                        f"{name}: DATA LEAKAGE detected: {overlap_n} rows occur in both "
                        "direction-training and held-out evaluation indices."
                    )
            independent_probe_scope = {
                "mode": "held_out_rows_from_emotion_vector_split",
                "source_type": src_type,
                "eval_indices_file": str(eval_path),
                "n_eval_rows_before_label_filter": int(len(independent_probe_eval_indices)),
                "train_eval_overlap_rows": overlap_n,
                "split_by": emotion_sidecar_config.get("split_by"),
                "split_sha256": emotion_sidecar_config.get("split_sha256"),
            }

    label_counts = dict(Counter(labels))
    present = set(labels)
    erc_classes = [c for c in ERC_CLASSES if c in present]
    extended_classes = [c for c in EXTENDED_CLASSES if c in present]

    excluded_erc = {
        k: v for k, v in label_counts.items()
        if k not in set(erc_classes)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name,
        "turn_dir": str(turn_dir),
        "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "dtype": str(vectors.dtype),
        "finite_rows": int(finite_mask.sum()),
        "nonfinite_rows": int((~finite_mask).sum()),
        "metadata_fingerprint_sha256": metadata_fp,
        "run_config": config,
        "emotion_vector_provenance": emotion_provenance,
        "label_counts": label_counts,
        "speaker_counts": dict(Counter(speakers)),
        "speaker_emotion_counts": {},
        "erc7_excluded_label_counts": excluded_erc,
        "erc7_excluded_fraction": float(
            sum(excluded_erc.values()) / max(len(labels), 1)
        ),
    }

    for sp in sorted(set(speakers)):
        result["speaker_emotion_counts"][sp] = dict(
            Counter(labels[i] for i, s in enumerate(speakers) if s == sp)
        )

    analyses = {}
    geometry_files = []

    if len(erc_classes) >= 2:
        erc_analysis, erc_geometries = analyze_label_space(
            vectors=vectors,
            labels=labels,
            speakers=speakers,
            dialogue_ids=dialogue_ids,
            topic_ids=topic_ids,
            classes=erc_classes,
            eligible_mask=finite_mask,
            pair_samples=pair_samples,
            bootstraps=bootstraps,
            cv_folds=cv_folds,
            seed=seed,
        )
        analyses["erc7"] = erc_analysis
        # Backward-compatible fields for older notebooks/scripts.
        result["classes"] = erc_classes
        result["subsets"] = erc_analysis["subsets"]

        for subset_name in ("ai", "all", "human"):
            if subset_name in erc_geometries:
                np.save(
                    output_dir / f"cross_model_geometry_erc7_{subset_name}.npy",
                    erc_geometries[subset_name],
                )
                (output_dir / f"cross_model_classes_erc7_{subset_name}.json").write_text(
                    json.dumps(erc_classes), encoding="utf-8"
                )
                geometry_files.append(f"erc7_{subset_name}")
        # Legacy AI-preferred files so older code keeps working.
        preferred_geom = erc_geometries.get("ai")
        if preferred_geom is None:
            preferred_geom = erc_geometries.get("all")
        if preferred_geom is not None:
            np.save(output_dir / "cross_model_geometry.npy", preferred_geom)
            (output_dir / "cross_model_classes.json").write_text(
                json.dumps(erc_classes), encoding="utf-8"
            )

    if len(extended_classes) >= 2:
        ext_analysis, ext_geometries = analyze_label_space(
            vectors=vectors,
            labels=labels,
            speakers=speakers,
            dialogue_ids=dialogue_ids,
            topic_ids=topic_ids,
            classes=extended_classes,
            eligible_mask=finite_mask,
            pair_samples=pair_samples,
            bootstraps=bootstraps,
            cv_folds=cv_folds,
            seed=seed + 1000,
        )
        analyses["extended9"] = ext_analysis
        for subset_name in ("ai", "all", "human"):
            if subset_name in ext_geometries:
                np.save(
                    output_dir / f"cross_model_geometry_extended9_{subset_name}.npy",
                    ext_geometries[subset_name],
                )
                (output_dir / f"cross_model_classes_extended9_{subset_name}.json").write_text(
                    json.dumps(extended_classes), encoding="utf-8"
                )
                geometry_files.append(f"extended9_{subset_name}")

    result["analyses"] = analyses
    result["saved_contextual_geometries"] = geometry_files

    # Fixed emotion directions are evaluated separately from the supervised centroid CV.
    # For dialogue-derived ERC directions, only the held-out split is used here.
    if emotion_vectors is not None:
        if not emotion_vectors.exists():
            raise FileNotFoundError(
                f"Emotion-vector file was specified for {name} but does not exist: "
                f"{emotion_vectors}. Fix the manifest/path instead of silently skipping it."
            )
        result["paper_style_probe"] = {}
        result["independent_probe_evaluation_scope"] = independent_probe_scope
        for label_space_name, class_list in (
            ("erc7", erc_classes),
            ("extended9", extended_classes),
        ):
            if len(class_list) < 2:
                continue
            result["paper_style_probe"][label_space_name] = {}
            class_set = set(class_list)
            if independent_probe_eval_indices is None:
                candidate_indices = np.arange(len(labels), dtype=np.int64)
            else:
                candidate_indices = independent_probe_eval_indices
            base_indices = np.array(
                [
                    int(i) for i in candidate_indices
                    if finite_mask[int(i)] and labels[int(i)] in class_set
                ],
                dtype=np.int64,
            )
            probe_subsets = {"all": base_indices}
            for sp in sorted(set(speakers)):
                probe_subsets[sp] = np.array(
                    [i for i in base_indices if speakers[i] == sp],
                    dtype=np.int64,
                )
            for subset_name, idx in probe_subsets.items():
                if len(idx) >= 10:
                    result["paper_style_probe"][label_space_name][subset_name] = (
                        evaluate_emotion_probes(
                            vectors,
                            meta,
                            emotion_vectors,
                            idx,
                            class_list,
                            bootstraps=bootstraps,
                            seed=seed,
                        )
                    )

    with (output_dir / "turn_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def rsa_values(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        return np.array([]), np.array([])
    mask = np.triu(np.ones_like(a, dtype=bool), k=1) & np.isfinite(a) & np.isfinite(b)
    return a[mask].astype(np.float64), b[mask].astype(np.float64)


def rsa_similarity(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    x, y = rsa_values(a, b)
    if len(x) < 3:
        return float("nan")
    if method == "pearson":
        return pearson(x, y)
    if method == "spearman":
        return spearman(x, y)
    raise ValueError(f"Unknown RSA method: {method}")


def rsa_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    permutations: int = 2000,
    seed: int = 42,
) -> float:
    """Two-sided class-label permutation test for Pearson RSA."""
    obs = rsa_similarity(a, b, method="pearson")
    if not np.isfinite(obs) or permutations <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    extreme = 0
    valid = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        bp = b[np.ix_(perm, perm)]
        r = rsa_similarity(a, bp, method="pearson")
        if np.isfinite(r):
            valid += 1
            if abs(r) >= abs(obs):
                extreme += 1
    return float((extreme + 1) / (valid + 1)) if valid else float("nan")


def load_and_align_geometries(
    analyses: List[Tuple[str, Path]],
    matrix_filename: str,
    classes_filename: str,
) -> Tuple[List[str], List[np.ndarray], List[str]]:
    names, mats, class_lists = [], [], []
    for name, analysis_dir in analyses:
        mp = analysis_dir / matrix_filename
        cp = analysis_dir / classes_filename
        if mp.exists() and cp.exists():
            names.append(name)
            mats.append(np.load(mp))
            class_lists.append(json.loads(cp.read_text(encoding="utf-8")))
    if len(mats) < 2:
        return [], [], []

    common = [c for c in class_lists[0] if all(c in cls for cls in class_lists[1:])]
    if len(common) < 3:
        return [], [], []

    aligned = []
    for mat, cls in zip(mats, class_lists):
        idx = [cls.index(c) for c in common]
        aligned.append(mat[np.ix_(idx, idx)])
    return names, aligned, common


def cross_model_rsa_generic(
    analyses: List[Tuple[str, Path]],
    output_dir: Path,
    stem: str,
    matrix_filename: str,
    classes_filename: str,
    permutations: int,
    seed: int,
    make_legacy_alias: bool = False,
) -> Optional[Dict[str, object]]:
    names, mats, classes = load_and_align_geometries(
        analyses, matrix_filename, classes_filename
    )
    if len(mats) < 2:
        return None

    n = len(mats)
    pear = np.eye(n, dtype=np.float32)
    spear = np.eye(n, dtype=np.float32)
    pvals = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            pear[i, j] = pear[j, i] = rsa_similarity(mats[i], mats[j], "pearson")
            spear[i, j] = spear[j, i] = rsa_similarity(mats[i], mats[j], "spearman")
            p = rsa_permutation_pvalue(
                mats[i], mats[j],
                permutations=permutations,
                seed=seed + i * 1009 + j * 9176,
            )
            pvals[i, j] = pvals[j, i] = p

    tri = np.triu_indices(n, k=1)
    pear_vals = pear[tri]
    spear_vals = spear[tri]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"{stem}_pearson.csv", pear, delimiter=",")
    np.savetxt(output_dir / f"{stem}_spearman.csv", spear, delimiter=",")
    np.savetxt(output_dir / f"{stem}_permutation_p.csv", pvals, delimiter=",")
    plot_heatmap(
        pear, names,
        f"{stem.replace('_', ' ')} — Pearson RSA",
        output_dir / f"{stem}_pearson.png",
    )

    out = {
        "models": names,
        "classes": classes,
        "n_pairwise_emotion_relationships": int(len(classes) * (len(classes) - 1) / 2),
        "pearson_rsa_matrix": pear.tolist(),
        "spearman_rsa_matrix": spear.tolist(),
        "pearson_permutation_p_matrix": pvals.tolist(),
        "mean_pairwise_pearson_rsa": float(np.nanmean(pear_vals)),
        "mean_pairwise_spearman_rsa": float(np.nanmean(spear_vals)),
        "permutations": int(permutations),
        "warning": (
            "Models were evaluated on the same labelled synthetic dialogue corpus. "
            "High contextual-centroid RSA demonstrates convergent relational geometry "
            "on this dataset, but independent-corpus and independent-direction RSA are "
            "stronger controls."
        ),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    # Preserve the filenames the user already expects. These aliases correspond
    # specifically to the primary ERC-7 AI contextual-centroid Pearson RSA.
    if make_legacy_alias:
        np.savetxt(output_dir / "cross_model_rsa.csv", pear, delimiter=",")
        plot_heatmap(
            pear, names,
            "Cross-model representational similarity (ERC-7 AI)",
            output_dir / "cross_model_rsa.png",
        )
        legacy = {
            "models": names,
            "classes": classes,
            "rsa_matrix": pear.tolist(),
            "spearman_rsa_matrix": spear.tolist(),
            "permutation_p_matrix": pvals.tolist(),
            "mean_pairwise_pearson_rsa": out["mean_pairwise_pearson_rsa"],
            "mean_pairwise_spearman_rsa": out["mean_pairwise_spearman_rsa"],
            "permutations": int(permutations),
        }
        (output_dir / "cross_model_rsa.json").write_text(
            json.dumps(legacy, indent=2), encoding="utf-8"
        )
    return out


def cross_model_rsa_suite(
    analyses: List[Tuple[str, Path]],
    output_dir: Path,
    permutations: int,
    seed: int,
) -> Dict[str, object]:
    """Run contextual ERC-7/extended-9 RSA and independent-direction RSA."""
    specs = [
        (
            "contextual_erc7_ai",
            "cross_model_geometry_erc7_ai.npy",
            "cross_model_classes_erc7_ai.json",
            True,
        ),
        (
            "contextual_erc7_human",
            "cross_model_geometry_erc7_human.npy",
            "cross_model_classes_erc7_human.json",
            False,
        ),
        (
            "contextual_extended9_ai",
            "cross_model_geometry_extended9_ai.npy",
            "cross_model_classes_extended9_ai.json",
            False,
        ),
        (
            "contextual_extended9_human",
            "cross_model_geometry_extended9_human.npy",
            "cross_model_classes_extended9_human.json",
            False,
        ),
        (
            "independent_emotion_direction_rsa",
            "emotion_direction_geometry.npy",
            "emotion_direction_classes.json",
            False,
        ),
    ]
    out = {}
    for k, matrix_file, class_file, legacy in specs:
        r = cross_model_rsa_generic(
            analyses=analyses,
            output_dir=output_dir,
            stem=k,
            matrix_filename=matrix_file,
            classes_filename=class_file,
            permutations=permutations,
            seed=seed,
            make_legacy_alias=legacy,
        )
        out[k] = r
    return out


def parse_manifest(path: Path) -> List[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    runs = obj.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must contain a non-empty 'runs' list")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, help="JSON manifest for multi-model analysis")
    ap.add_argument("--turn-dir", type=Path, help="Single run directory containing turn_vectors.npy")
    ap.add_argument("--emotion-vectors", type=Path, help="Fixed class-level emotion_vectors.npz for the SAME model/layer. Dialogue-derived directions should include held-out eval_indices.npy sidecar data.")
    ap.add_argument("--name", default="model", help="Name for single-model mode")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis_research_v3"))
    ap.add_argument("--ratings-csv", type=Path, help="Independent human affect ratings: emotion,valence,arousal")
    ap.add_argument("--pair-samples", type=int, default=50000)
    ap.add_argument("--bootstraps", type=int, default=200)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--pca-null-sims", type=int, default=500)
    ap.add_argument("--rsa-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-lexical-control",
        action="store_true",
        help="Skip the dataset-only unigram lexical baseline.",
    )
    ap.add_argument(
        "--require-emotion-vectors",
        action="store_true",
        help=(
            "Fail unless every run supplies a valid independent emotion_vectors.npz. "
            "Recommended for the paper-style replication run."
        ),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratings = load_affect_ratings(args.ratings_csv)

    if args.manifest:
        runs = parse_manifest(args.manifest)
    elif args.turn_dir:
        runs = [{
            "name": args.name,
            "turn_dir": str(args.turn_dir),
            "emotion_vectors": str(args.emotion_vectors) if args.emotion_vectors else None,
        }]
    else:
        ap.error("Provide --manifest or --turn-dir")

    # Validate everything up front so a typo cannot silently create emotion_geometry=null.
    validation_errors = []
    for r in runs:
        if "name" not in r or "turn_dir" not in r:
            validation_errors.append(f"Malformed run entry: {r}")
            continue
        td = Path(r["turn_dir"])
        if not (td / "turn_vectors.npy").exists():
            validation_errors.append(f"{r['name']}: missing {td / 'turn_vectors.npy'}")
        if not (td / "turn_metadata.jsonl").exists():
            validation_errors.append(f"{r['name']}: missing {td / 'turn_metadata.jsonl'}")
        ev_raw = r.get("emotion_vectors")
        if ev_raw:
            ev = Path(ev_raw)
            if not ev.exists():
                validation_errors.append(
                    f"{r['name']}: emotion_vectors path does not exist: {ev}"
                )
        elif args.require_emotion_vectors:
            validation_errors.append(
                f"{r['name']}: no emotion_vectors path supplied, but --require-emotion-vectors was set"
            )

    if validation_errors:
        raise FileNotFoundError(
            "Input validation failed before analysis:\n  - "
            + "\n  - ".join(validation_errors)
        )

    all_summary = []
    rsa_inputs = []

    for r in runs:
        name = r["name"]
        turn_dir = Path(r["turn_dir"])
        ev = Path(r["emotion_vectors"]) if r.get("emotion_vectors") else None
        out = args.output_dir / name
        print(f"\n=== {name} ===")

        turn_result = analyze_turn_dir(
            turn_dir=turn_dir,
            output_dir=out,
            name=name,
            emotion_vectors=ev,
            pair_samples=args.pair_samples,
            bootstraps=args.bootstraps,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )

        geometry_result = None
        if ev is not None:
            geometry_result = analyze_emotion_vectors(
                ev, out, ratings, args.pca_null_sims
            )
        else:
            print(
                "  WARNING: no independent emotion_vectors.npz supplied; "
                "paper-style direction generalization and direction RSA are unavailable."
            )

        all_summary.append({
            "name": name,
            "turn_analysis": turn_result,
            "emotion_geometry": geometry_result,
        })
        rsa_inputs.append((name, out))

        erc_ai = (
            turn_result.get("analyses", {})
            .get("erc7", {})
            .get("subsets", {})
            .get("ai")
        )
        erc_all = (
            turn_result.get("analyses", {})
            .get("erc7", {})
            .get("subsets", {})
            .get("all")
        )
        preferred = erc_ai or erc_all
        if preferred:
            sep = preferred["same_vs_different_emotion"]
            cv = preferred["dialogue_group_cv"]
            print(f"  ERC-7 same-vs-different cosine gap: {sep.get('gap', float('nan')):.4f}")
            print(f"  ERC-7 dialogue CV accuracy:        {cv.get('accuracy', float('nan')):.4f}")
            print(f"  ERC-7 dialogue CV macro-F1:        {cv.get('macro_f1', float('nan')):.4f}")

        ext_ai = (
            turn_result.get("analyses", {})
            .get("extended9", {})
            .get("subsets", {})
            .get("ai")
        )
        if ext_ai:
            cv9 = ext_ai["dialogue_group_cv"]
            print(f"  9-state dialogue CV macro-F1:      {cv9.get('macro_f1', float('nan')):.4f}")

        if ev is not None:
            probe = (
                turn_result.get("paper_style_probe", {})
                .get("erc7", {})
                .get("ai")
            )
            if probe and "error" not in probe:
                print(f"  independent-direction macro-AUROC: {probe.get('macro_auroc', float('nan')):.4f}")
                print(f"  independent-direction top1 acc:    {probe.get('accuracy', float('nan')):.4f}")

        if geometry_result:
            print(f"  emotion-vector top-2 PCA variance: {geometry_result['top2_variance']:.4f}")
            print(f"  random-null 95th percentile:       {geometry_result['top2_random_null']['p95']:.4f}")

    # Dataset-only lexical/style control (computed once because every model uses
    # the same dialogues). This directly tests whether neutral or other classes
    # are trivially recoverable from surface wording.
    lexical_control = None
    if not args.skip_lexical_control and runs:
        first_meta = load_metadata(Path(runs[0]["turn_dir"]) / "turn_metadata.jsonl")
        print("\n=== Dataset lexical/style control ===")
        lexical_control = dataset_lexical_controls(
            first_meta, args.output_dir, cv_folds=args.cv_folds, seed=args.seed
        )
        try:
            neutral_rec = (
                lexical_control["erc7"]["all"]["dialogue_group_cv"]["recall"]["neutral"]
            )
            print(f"  lexical baseline neutral recall: {neutral_rec:.4f}")
        except Exception:
            pass

    # Confirm that cross-model comparisons use identically aligned turn metadata.
    fingerprints = [
        r["turn_analysis"].get("metadata_fingerprint_sha256")
        for r in all_summary
    ]
    dataset_alignment = {
        "all_metadata_fingerprints_identical": (
            len(set(fingerprints)) == 1 if fingerprints else False
        ),
        "fingerprints_by_model": {
            r["name"]: r["turn_analysis"].get("metadata_fingerprint_sha256")
            for r in all_summary
        },
    }

    rsa_suite = cross_model_rsa_suite(
        rsa_inputs,
        args.output_dir,
        permutations=args.rsa_permutations,
        seed=args.seed,
    )

    final = {
        "runs": all_summary,
        "dataset_alignment": dataset_alignment,
        "cross_model_rsa": rsa_suite,
        "dataset_lexical_control": lexical_control,
        "interpretation_guardrails": {
            "contextual_rsa": (
                "Contextual-centroid RSA is computed from the same labelled dialogue corpus "
                "across models; use it as evidence of convergent relational geometry on this dataset."
            ),
            "independent_direction_rsa": (
                "Independent emotion-direction RSA is the stronger cross-model control because "
                "the geometry comes from separately extracted emotion directions."
            ),
            "causality": (
                "No saved-vector analysis establishes causal influence. Causal claims require "
                "model interventions/steering."
            ),
        },
    }
    with (args.output_dir / "all_models_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis written to: {args.output_dir}")
    print(
        f"Metadata aligned across all models: "
        f"{dataset_alignment['all_metadata_fingerprints_identical']}"
    )
    print(
        "Important: layer dynamics, token-locality, and causal steering require "
        "additional targeted extraction/intervention runs."
    )


if __name__ == "__main__":
    main()
