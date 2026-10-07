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
  uv run python analyze_emotion_research_old.py \
      --turn-dir results_qwen3_1.7b_full \
      --emotion-vectors results_qwen3_emotions/emotion_vectors.npz \
      --name qwen3_1.7b \
      --output-dir analysis/qwen3_1.7b

Example multi-model run with a manifest:
  uv run python analyze_emotion_research_old.py \
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
}

DEFAULT_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


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
) -> Dict[str, object]:
    data = np.load(emotion_vector_path)
    direction_map = {canonical_emotion(k): safe_float_array(data[k]) for k in data.files}
    common = [c for c in classes if c in direction_map]
    if len(common) < 2:
        return {"error": "fewer than two overlapping emotion directions", "overlap": common}
    dirs = np.stack([direction_map[c] for c in common])
    if dirs.shape[1] != vectors.shape[1]:
        return {
            "error": "hidden dimension mismatch",
            "turn_dim": int(vectors.shape[1]),
            "emotion_dim": int(dirs.shape[1]),
        }

    true, pred = [], []
    activation_sum = {c: defaultdict(float) for c in common}
    activation_n = {c: defaultdict(int) for c in common}
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = np.asarray(vectors[idx], dtype=np.float32)
        scores = probe_scores_batch(x, dirs)
        p = np.argmax(scores, axis=1)
        for row, ii in enumerate(idx):
            label = canonical_emotion(meta[ii].get("emotion", ""))
            if label in common:
                true.append(label)
                pred.append(common[p[row]])
                for j, c in enumerate(common):
                    activation_sum[label][c] += float(scores[row, j])
                    activation_n[label][c] += 1
    cm = confusion_matrix(true, pred, common)
    metrics = metrics_from_cm(cm, common)
    metrics["classes"] = common
    metrics["confusion_matrix"] = cm.tolist()
    metrics["note"] = "Argmax classification is an application diagnostic; the paper primarily treats these as continuous linear probe activations."
    metrics["mean_probe_activation_by_true_label"] = {
        y: {c: activation_sum[y][c] / max(activation_n[y][c], 1) for c in common}
        for y in common
    }
    return metrics


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
    classes = [c for c in DEFAULT_CLASSES if c in set(labels)]
    indices_all = np.array([i for i, lab in enumerate(labels) if lab in classes], dtype=np.int64)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name,
        "turn_dir": str(turn_dir),
        "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "dtype": str(vectors.dtype),
        "run_config": config,
        "classes": classes,
        "label_counts": dict(Counter(labels)),
        "speaker_counts": dict(Counter(speakers)),
        "speaker_emotion_counts": {},
    }

    for sp in sorted(set(speakers)):
        result["speaker_emotion_counts"][sp] = dict(Counter(labels[i] for i, s in enumerate(speakers) if s == sp))

    # Analyze all turns and each speaker separately. This explicitly surfaces speaker/emotion confounding.
    subsets = {"all": indices_all}
    for sp in sorted(set(speakers)):
        subsets[sp] = np.array([i for i in indices_all if speakers[i] == sp], dtype=np.int64)

    subset_results = {}
    geometry_for_cross_model = None
    labels_arr = np.asarray(labels, dtype=object)

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
        topic_cv = grouped_centroid_cv(
            vectors, labels, idx, topic_ids, classes,
            k=min(cv_folds, max(2, len(set(np.asarray(topic_ids, dtype=object)[idx].tolist())))),
            seed=seed,
        )
        cents, counts = class_centroids_stream(vectors, idx, labels, classes, center=center)
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

        # Prefer AI geometry for cross-model RSA when present; otherwise all turns.
        if subset_name == "ai" or (geometry_for_cross_model is None and subset_name == "all"):
            geometry_for_cross_model = centroid_cos

    result["subsets"] = subset_results

    if emotion_vectors is not None and emotion_vectors.exists():
        result["paper_style_probe"] = {}
        for subset_name, idx in subsets.items():
            if len(idx) >= 10:
                result["paper_style_probe"][subset_name] = evaluate_emotion_probes(
                    vectors, meta, emotion_vectors, idx, classes
                )

    with (output_dir / "turn_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if geometry_for_cross_model is not None:
        np.save(output_dir / "cross_model_geometry.npy", geometry_for_cross_model)
        (output_dir / "cross_model_classes.json").write_text(json.dumps(classes), encoding="utf-8")

    return result


def rsa_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mask = np.triu(np.ones_like(a, dtype=bool), k=1) & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return pearson(a[mask], b[mask])


def cross_model_rsa(analyses: List[Tuple[str, Path]], output_dir: Path) -> Optional[Dict[str, object]]:
    names, mats, classes_list = [], [], []
    for name, analysis_dir in analyses:
        mp = analysis_dir / "cross_model_geometry.npy"
        cp = analysis_dir / "cross_model_classes.json"
        if mp.exists() and cp.exists():
            names.append(name)
            mats.append(np.load(mp))
            classes_list.append(json.loads(cp.read_text(encoding="utf-8")))
    if len(mats) < 2:
        return None
    if any(c != classes_list[0] for c in classes_list[1:]):
        return {"error": "models do not have identical canonical class ordering"}
    n = len(mats)
    rsa = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            rsa[i, j] = rsa[j, i] = rsa_similarity(mats[i], mats[j])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "cross_model_rsa.csv", rsa, delimiter=",")
    plot_heatmap(rsa, names, "Cross-model representational similarity", output_dir / "cross_model_rsa.png")
    out = {"models": names, "classes": classes_list[0], "rsa_matrix": rsa.tolist()}
    (output_dir / "cross_model_rsa.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
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
    ap.add_argument("--emotion-vectors", type=Path, help="Optional class-level emotion_vectors.npz for the SAME model and SAME layer")
    ap.add_argument("--name", default="model", help="Name for single-model mode")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis_research_v2"))
    ap.add_argument("--ratings-csv", type=Path, help="Independent human affect ratings: emotion,valence,arousal")
    ap.add_argument("--pair-samples", type=int, default=50000)
    ap.add_argument("--bootstraps", type=int, default=100)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--pca-null-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
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
        if ev is not None and ev.exists():
            geometry_result = analyze_emotion_vectors(ev, out, ratings, args.pca_null_sims)

        all_summary.append({
            "name": name,
            "turn_analysis": turn_result,
            "emotion_geometry": geometry_result,
        })
        rsa_inputs.append((name, out))

        # Compact console summary.
        preferred = turn_result.get("subsets", {}).get("ai") or turn_result.get("subsets", {}).get("all")
        if preferred:
            sep = preferred["same_vs_different_emotion"]
            cv = preferred["dialogue_group_cv"]
            print(f"  same-vs-different cosine gap: {sep.get('gap', float('nan')):.4f}")
            print(f"  dialogue-group CV accuracy:  {cv.get('accuracy', float('nan')):.4f}")
            print(f"  dialogue-group CV macro-F1:  {cv.get('macro_f1', float('nan')):.4f}")
        if geometry_result:
            print(f"  emotion-vector top-2 PCA variance: {geometry_result['top2_variance']:.4f}")
            print(f"  random-null 95th percentile:      {geometry_result['top2_random_null']['p95']:.4f}")

    rsa = cross_model_rsa(rsa_inputs, args.output_dir)
    final = {"runs": all_summary, "cross_model_rsa": rsa}
    with (args.output_dir / "all_models_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis written to: {args.output_dir}")
    print("Important: layer dynamics, token-locality, and causal steering require additional targeted extraction/intervention runs.")


if __name__ == "__main__":
    main()
