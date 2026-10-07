#!/usr/bin/env python3
"""Derive ERC emotion directions directly from contextual dialogue turn vectors.

This is for the dialogue-only Emotion Vector -> ERC experiment.
It DOES NOT use synthetic stories and DOES NOT run the language model again.

Inputs (from extract_dialogue_turn_vectors_macos.py):
    <turn-run-dir>/turn_vectors.npy
    <turn-run-dir>/turn_metadata.jsonl
    <turn-run-dir>/run_config.json

Method:
  1) Split data at the DIALOGUE level into direction-training and held-out eval sets.
     This prevents turns from the same conversation appearing in both construction
     and evaluation.
  2) Within the training split, average turns per (dialogue, speaker). This prevents
     long dialogues from receiving more weight simply because they contain more turns.
  3) Average those dialogue-speaker units within each emotion.
  4) Subtract the equal-weight mean across emotion class means:
         v_e = mean_e - mean_over_classes(mean_class)
  5) Save emotion_vectors.npz plus train/eval row indices and full provenance.

The companion analyze_emotion_research_old.py detects eval_indices.npy and evaluates
these directions ONLY on held-out dialogue turns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ALIASES = {
    "anger": "anger", "angry": "anger", "mad": "anger", "furious": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "afraid": "fear", "scared": "fear", "frightened": "fear",
    "joy": "joy", "joyful": "joy", "happy": "joy",
    "neutral": "neutral",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise", "astonished": "surprise",
    "peaceful": "peaceful",
    "powerful": "powerful",
}

ERC7 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EXTENDED9 = ERC7 + ["peaceful", "powerful"]


def canonical(x: object) -> str:
    s = str(x if x is not None else "").strip().lower().replace("_", " ")
    return ALIASES.get(s, s)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{ln}")
            out.append(obj)
    return out


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def metadata_fingerprint(meta: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for m in meta:
        key = (
            m.get("dialogue_id"),
            m.get("turn_idx"),
            str(m.get("speaker", "")).lower(),
            canonical(m.get("emotion", "")),
            m.get("topic_idx"),
        )
        h.update(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def dialogue_records(meta: Sequence[dict], eligible_classes: set[str]) -> Dict[object, dict]:
    """Collect one stratification record per dialogue."""
    d: Dict[object, dict] = {}
    for i, m in enumerate(meta):
        did = m.get("dialogue_id")
        if did is None:
            raise ValueError(f"Metadata row {i} has no dialogue_id")
        rec = d.setdefault(did, {
            "dialogue_id": did,
            "topic_idx": m.get("topic_idx"),
            "person_emotion": canonical(m.get("person_emotion", "")),
            "ai_emotion": canonical(m.get("ai_emotion", "")),
            "rows": [],
        })
        lab = canonical(m.get("emotion", ""))
        if lab in eligible_classes:
            rec["rows"].append(i)
    return d


def stratified_dialogue_split(
    records: Dict[object, dict],
    eval_fraction: float,
    seed: int,
    split_by: str,
) -> Tuple[set, set, dict]:
    """Deterministic grouped split, stratified by person/AI emotion pair.

    split_by='dialogue' samples individual dialogue IDs within each emotion-pair stratum.
    split_by='topic' holds out whole topic_idx groups, preventing topic overlap as well.
    """
    if not 0.05 <= eval_fraction <= 0.5:
        raise ValueError("--eval-fraction must be between 0.05 and 0.5")

    rng = np.random.default_rng(seed)
    valid = [r for r in records.values() if r["rows"]]
    if not valid:
        raise ValueError("No dialogues contain requested emotion classes")

    if split_by == "dialogue":
        strata: Dict[tuple, List[object]] = defaultdict(list)
        for r in valid:
            key = (r["person_emotion"], r["ai_emotion"])
            strata[key].append(r["dialogue_id"])

        eval_ids: set = set()
        for key, ids in sorted(strata.items(), key=lambda kv: str(kv[0])):
            ids = list(ids)
            rng.shuffle(ids)
            if len(ids) == 1:
                # Keep singleton strata in training; otherwise the vector for that
                # emotion combination would have no construction example.
                n_eval = 0
            else:
                n_eval = int(round(len(ids) * eval_fraction))
                n_eval = min(max(n_eval, 1), len(ids) - 1)
            eval_ids.update(ids[:n_eval])
        all_ids = {r["dialogue_id"] for r in valid}
        train_ids = all_ids - eval_ids
        detail = {
            "split_by": "dialogue",
            "stratification": "(person_emotion, ai_emotion)",
            "num_strata": len(strata),
        }
        return train_ids, eval_ids, detail

    # Topic-held-out split: choose whole topics. Exact pair-stratification is impossible
    # while keeping topics intact, so use a deterministic random topic split and report it.
    topics = sorted({r["topic_idx"] for r in valid if r["topic_idx"] is not None}, key=str)
    if len(topics) < 2:
        raise ValueError("Need at least two topic_idx values for --split-by topic")
    rng.shuffle(topics)
    n_eval_topics = min(max(int(round(len(topics) * eval_fraction)), 1), len(topics) - 1)
    eval_topics = set(topics[:n_eval_topics])
    eval_ids = {r["dialogue_id"] for r in valid if r["topic_idx"] in eval_topics}
    all_ids = {r["dialogue_id"] for r in valid}
    train_ids = all_ids - eval_ids
    detail = {
        "split_by": "topic",
        "num_topics": len(topics),
        "num_eval_topics": len(eval_topics),
        "eval_topics": sorted(eval_topics, key=str),
    }
    return train_ids, eval_ids, detail


def rows_for_dialogues(
    meta: Sequence[dict], dialogue_ids: set, classes: set[str], speaker: str
) -> np.ndarray:
    idx = []
    for i, m in enumerate(meta):
        if m.get("dialogue_id") not in dialogue_ids:
            continue
        if canonical(m.get("emotion", "")) not in classes:
            continue
        sp = str(m.get("speaker", "unknown")).lower()
        if speaker != "all" and sp != speaker:
            continue
        idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def finite_rows(vectors: np.ndarray, indices: np.ndarray, batch: int = 4096) -> np.ndarray:
    keep = []
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = np.asarray(vectors[idx], dtype=np.float32)
        keep.append(idx[np.isfinite(x).all(axis=1)])
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


def build_dialogue_speaker_units(
    vectors: np.ndarray,
    meta: Sequence[dict],
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Average turn activations within (dialogue_id, speaker, emotion) first."""
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i in train_indices.tolist():
        m = meta[i]
        key = (
            m.get("dialogue_id"),
            str(m.get("speaker", "unknown")).lower(),
            canonical(m.get("emotion", "")),
        )
        groups[key].append(i)

    unit_vecs = []
    labels = []
    unit_meta = []
    for (did, speaker, label), idxs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        x = np.asarray(vectors[np.asarray(idxs, dtype=np.int64)], dtype=np.float32)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            continue
        unit_vecs.append(x.mean(axis=0))
        labels.append(label)
        unit_meta.append({
            "dialogue_id": did,
            "speaker": speaker,
            "emotion": label,
            "num_turns": len(x),
        })
    if not unit_vecs:
        raise ValueError("No finite training units could be constructed")
    return np.stack(unit_vecs).astype(np.float32), labels, unit_meta


def derive_directions(
    unit_vectors: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    min_train_units: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    labels_arr = np.asarray(labels, dtype=object)
    means: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    missing = []
    for c in classes:
        idx = np.where(labels_arr == c)[0]
        counts[c] = int(len(idx))
        if len(idx) < min_train_units:
            missing.append(f"{c}={len(idx)}")
        else:
            means[c] = unit_vectors[idx].mean(axis=0).astype(np.float32)
    if missing:
        raise ValueError(
            "Insufficient dialogue-speaker training units for classes: "
            + ", ".join(missing)
            + f"; need at least {min_train_units} each"
        )

    # Equal-weight class baseline avoids class-frequency imbalance.
    global_mean = np.stack([means[c] for c in classes], axis=0).mean(axis=0).astype(np.float32)
    directions = {c: (means[c] - global_mean).astype(np.float32) for c in classes}
    return directions, means, global_mean, counts


def cosine_matrix(direction_map: Dict[str, np.ndarray], classes: Sequence[str]) -> np.ndarray:
    x = np.stack([direction_map[c] for c in classes]).astype(np.float32)
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return z @ z.T


def class_counts(meta: Sequence[dict], indices: np.ndarray) -> Dict[str, int]:
    return dict(Counter(canonical(meta[i].get("emotion", "")) for i in indices.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive held-out-safe ERC emotion directions from existing dialogue turn vectors."
    )
    ap.add_argument("--turn-run-dir", type=Path, required=True,
                    help="Directory with turn_vectors.npy, turn_metadata.jsonl, run_config.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--eval-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-by", choices=["dialogue", "topic"], default="dialogue")
    ap.add_argument("--speaker", choices=["all", "human", "ai"], default="all",
                    help="Which speaker turns construct directions; default all. Evaluation indices still cover both speakers unless analyzer subsets them.")
    ap.add_argument("--label-space", choices=["erc7", "extended9"], default="erc7")
    ap.add_argument("--min-train-units", type=int, default=20,
                    help="Minimum (dialogue,speaker) units per class in direction-training split")
    ap.add_argument("--restart", action="store_true",
                    help="Overwrite an existing emotion_vectors.npz/config in output-dir")
    args = ap.parse_args()

    run_dir = args.turn_run_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = run_dir / "turn_vectors.npy"
    meta_path = run_dir / "turn_metadata.jsonl"
    cfg_path = run_dir / "run_config.json"
    for p in (vec_path, meta_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    final_path = out_dir / "emotion_vectors.npz"
    if final_path.exists() and not args.restart:
        raise FileExistsError(
            f"{final_path} already exists. Use --restart to replace it or choose another --output-dir."
        )

    vectors = np.load(vec_path, mmap_mode="r")
    meta = load_jsonl(meta_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if len(meta) != vectors.shape[0]:
        raise ValueError(f"Metadata rows {len(meta)} != vector rows {vectors.shape[0]}")

    classes = ERC7 if args.label_space == "erc7" else EXTENDED9
    class_set = set(classes)
    records = dialogue_records(meta, class_set)
    train_dialogues, eval_dialogues, split_detail = stratified_dialogue_split(
        records, eval_fraction=args.eval_fraction, seed=args.seed, split_by=args.split_by
    )
    if not train_dialogues or not eval_dialogues:
        raise ValueError("Train/eval dialogue split is empty")

    # Rows used to construct directions may be speaker-restricted.
    train_idx = rows_for_dialogues(meta, train_dialogues, class_set, args.speaker)
    train_idx = finite_rows(vectors, train_idx)

    # Evaluation rows intentionally include both speakers; analyzer will also report
    # human and AI subsets independently.
    eval_idx = rows_for_dialogues(meta, eval_dialogues, class_set, "all")
    eval_idx = finite_rows(vectors, eval_idx)

    # Save a full eligible index too, useful for audits.
    all_idx = rows_for_dialogues(meta, train_dialogues | eval_dialogues, class_set, "all")
    all_idx = finite_rows(vectors, all_idx)

    train_units, unit_labels, unit_meta = build_dialogue_speaker_units(vectors, meta, train_idx)
    directions, means, global_mean, unit_counts = derive_directions(
        train_units, unit_labels, classes, min_train_units=args.min_train_units
    )

    np.savez(final_path, **directions)
    np.savez(out_dir / "emotion_class_means.npz", **means)
    np.save(out_dir / "emotion_global_mean.npy", global_mean)
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "eval_indices.npy", eval_idx)
    np.save(out_dir / "all_eligible_indices.npy", all_idx)
    np.save(out_dir / "train_unit_vectors.npy", train_units.astype(np.float32))

    with (out_dir / "train_unit_metadata.jsonl").open("w", encoding="utf-8") as f:
        for i, m in enumerate(unit_meta):
            rec = dict(m)
            rec["unit_row"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cos = cosine_matrix(directions, classes)
    np.savetxt(out_dir / "emotion_direction_cosine.csv", cos, delimiter=",")

    fingerprint = metadata_fingerprint(meta)
    train_d_sorted = sorted(train_dialogues, key=str)
    eval_d_sorted = sorted(eval_dialogues, key=str)
    split_hash = hashlib.sha256(
        json.dumps({"train": train_d_sorted, "eval": eval_d_sorted}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    ev_cfg = {
        "method": "dialogue ERC direction = equal-weight class mean of training (dialogue,speaker) activation units minus equal-weight mean across class means",
        "source_type": "dialogue_turn_vectors_train_split",
        "turn_run_dir": str(run_dir),
        "model": cfg.get("model"),
        "target_layer": cfg.get("target_layer"),
        "hidden_size": int(vectors.shape[1]),
        "num_layers": cfg.get("num_layers"),
        "pooling_source": cfg.get("pooling"),
        "context_source": cfg.get("context"),
        "label_space": args.label_space,
        "classes": classes,
        "direction_source_speaker": args.speaker,
        "split_by": args.split_by,
        "eval_fraction_requested": args.eval_fraction,
        "seed": args.seed,
        "num_train_dialogues": len(train_dialogues),
        "num_eval_dialogues": len(eval_dialogues),
        "num_train_turn_rows": int(len(train_idx)),
        "num_eval_turn_rows": int(len(eval_idx)),
        "train_class_turn_counts": class_counts(meta, train_idx),
        "eval_class_turn_counts": class_counts(meta, eval_idx),
        "train_direction_unit_counts": unit_counts,
        "metadata_fingerprint_sha256": fingerprint,
        "split_sha256": split_hash,
        "split_detail": split_detail,
        "train_indices_file": "train_indices.npy",
        "eval_indices_file": "eval_indices.npy",
        "all_eligible_indices_file": "all_eligible_indices.npy",
        "emotion_vectors_file": "emotion_vectors.npz",
        "leakage_control": "emotion directions use training dialogues only; paper-style probe must evaluate eval_indices.npy only",
        "neutral_handling": "neutral is a target ERC class; no neutral-PCA projection is applied",
    }
    save_json(out_dir / "emotion_vector_config.json", ev_cfg)
    save_json(out_dir / "run_config.json", ev_cfg)

    diagnostics = {
        "classes": classes,
        "direction_norms": {c: float(np.linalg.norm(directions[c])) for c in classes},
        "direction_cosine_matrix": cos.tolist(),
        "train_direction_unit_counts": unit_counts,
        "train_class_turn_counts": class_counts(meta, train_idx),
        "eval_class_turn_counts": class_counts(meta, eval_idx),
        "split_sha256": split_hash,
        "notes": [
            "No story data is used.",
            "No model forward pass is required: the directions are derived from the already extracted contextual turn residual vectors.",
            "The train/eval split is performed before direction construction and at dialogue/topic group level to prevent conversation leakage.",
            "Each (dialogue, speaker) is averaged before class averaging so longer dialogues do not dominate the direction.",
            "Neutral remains an ERC target class, so neutral-PC nuisance removal is intentionally not used.",
        ],
    }
    save_json(out_dir / "diagnostics.json", diagnostics)

    print("=== Dialogue-derived ERC Emotion Directions ===")
    print(f"Turn run:            {run_dir}")
    print(f"Model:               {cfg.get('model')}")
    print(f"Target layer:        {cfg.get('target_layer')}")
    print(f"Hidden size:         {vectors.shape[1]}")
    print(f"Label space:         {args.label_space}: {', '.join(classes)}")
    print(f"Direction speaker:   {args.speaker}")
    print(f"Split:               {args.split_by}, eval_fraction={args.eval_fraction}, seed={args.seed}")
    print(f"Train dialogues:     {len(train_dialogues)}")
    print(f"Eval dialogues:      {len(eval_dialogues)}")
    print(f"Train turn rows:     {len(train_idx)}")
    print(f"Eval turn rows:      {len(eval_idx)}")
    print(f"Direction units:     {unit_counts}")
    print(f"Saved:               {final_path}")
    print(f"Held-out indices:    {out_dir / 'eval_indices.npy'}")
    print("IMPORTANT: analyze_emotion_research_old.py must use eval_indices.npy for the independent-direction probe.")


if __name__ == "__main__":
    main()
