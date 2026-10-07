#!/usr/bin/env python3
"""Extract one contextual residual-stream vector per Person/AI utterance.

Designed for Apple Silicon Macs (including M2 Pro with 16 GB unified memory).

Key idea
--------
For a causal language model, the hidden state of an earlier token cannot depend on
future tokens. Therefore, for normal-length dialogues we can run ONE forward pass
for the whole dialogue and mean-pool only the token span belonging to each turn.
That gives each utterance its preceding conversational context without requiring
one model call per turn.

For unusually long dialogues, the script automatically falls back to per-turn
left-context windows bounded by --max-context.

Outputs
-------
results_turn_vectors/
    turn_vectors.npy       float16 array [num_turns, hidden_size]
    turn_metadata.jsonl    one JSON object per vector row
    run_config.json        model/layer/configuration used
    progress.json          resume checkpoint

IMPORTANT: if you plan to compare these turn vectors with learned emotion
vectors/directions, extract BOTH with the exact same model and layer.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Let unsupported MPS ops fall back to CPU instead of failing outright.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "google/gemma-4-E4B-it" #"google/gemma-3-1b-it"
DEFAULT_MAX_CONTEXT = 2048
DEFAULT_LAYER_FRACTION = 2.0 / 3.0

TURN_MARKER = re.compile(r"(?m)^(Person|AI):\s*")


def choose_device(requested: str = "auto") -> torch.device:
    """Choose MPS on Apple Silicon, CUDA when available, otherwise CPU."""
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    """Use float16 on MPS for memory safety; float32 on CPU."""
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        # bfloat16 is generally a good default on recent CUDA GPUs.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def parse_turns(text: str) -> List[Dict]:
    """Parse Person:/AI: turns and preserve exact character spans of utterances."""
    matches = list(TURN_MARKER.finditer(text))
    turns: List[Dict] = []

    for i, match in enumerate(matches):
        speaker_raw = match.group(1)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        raw = text[content_start:content_end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())

        start_char = content_start + left_trim
        end_char = content_end - right_trim
        utterance = text[start_char:end_char]

        if not utterance:
            continue

        turns.append(
            {
                "speaker_raw": speaker_raw,
                "speaker": "human" if speaker_raw == "Person" else "ai",
                "utterance": utterance,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return turns


def get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    """Locate decoder blocks for common Hugging Face causal-LM architectures."""
    candidates = []

    # Gemma 4 style nesting (kept for compatibility if a user runs elsewhere).
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            candidates.append(lm.layers)
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(lm.model.layers)

    # Gemma 3, Qwen, Llama, Mistral and many decoder-only HF models.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)

    # GPT-style models.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        candidates.append(model.transformer.h)

    # GPT-NeoX style.
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        candidates.append(model.gpt_neox.layers)

    for layers in candidates:
        try:
            if len(layers) > 0:
                return layers
        except TypeError:
            pass

    raise RuntimeError(
        "Could not locate decoder layers for this model architecture. "
        "Inspect the model and extend get_decoder_layers()."
    )


def get_hidden_size(model) -> int:
    """Read hidden size from common HF config layouts."""
    configs = [model.config, getattr(model.config, "text_config", None)]
    for cfg in configs:
        if cfg is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)
    raise RuntimeError("Could not determine model hidden size from config.")


class TargetLayerCapture:
    """Capture ONLY one decoder layer to minimize memory use."""

    def __init__(self, layer: torch.nn.Module):
        self.hidden: Optional[torch.Tensor] = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        # Some modules can return structured outputs; decoder blocks normally
        # return a Tensor or tuple whose first item is the hidden state.
        if not torch.is_tensor(hidden):
            raise TypeError(f"Unexpected layer output type: {type(hidden)}")

        # Copy only the selected layer to CPU in float16. This avoids keeping
        # every layer's activation around and saves unified memory.
        self.hidden = hidden.detach().to(device="cpu", dtype=torch.float16)

    def clear(self) -> None:
        self.hidden = None

    def close(self) -> None:
        self.handle.remove()


def char_span_to_token_positions(
    offsets: torch.Tensor, start_char: int, end_char: int
) -> List[int]:
    """Find tokenizer positions whose character offsets overlap an utterance."""
    positions: List[int] = []
    for idx, (start, end) in enumerate(offsets.tolist()):
        # Special tokens commonly have (0, 0).
        if end <= start:
            continue
        if end > start_char and start < end_char:
            positions.append(idx)
    return positions


def forward_hidden(
    model,
    capture: TargetLayerCapture,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one inference pass and return target-layer hidden states on CPU."""
    capture.clear()

    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }

    with torch.inference_mode():
        model(**inputs, use_cache=False)

    if capture.hidden is None:
        raise RuntimeError("Target-layer hook did not capture any hidden states.")

    hidden = capture.hidden

    # Release input tensors from the accelerator promptly.
    del inputs
    return hidden


def pool_positions(hidden: torch.Tensor, positions: Sequence[int]) -> np.ndarray:
    """Mean-pool selected token positions into one float16 utterance vector."""
    if not positions:
        raise ValueError("Cannot pool an empty token span.")

    idx = torch.tensor(list(positions), dtype=torch.long)
    vector = hidden[0].index_select(0, idx).float().mean(dim=0)
    return vector.numpy().astype(np.float16, copy=False)


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize without truncation so we can locate every turn precisely."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )

    offsets = encoded.pop("offset_mapping")[0].cpu()
    input_ids = encoded["input_ids"].cpu()
    attention_mask = encoded["attention_mask"].cpu()
    return input_ids, attention_mask, offsets


def extract_dialogue_vectors(
    model,
    tokenizer,
    capture: TargetLayerCapture,
    text: str,
    turns: Sequence[Dict],
    device: torch.device,
    max_context: int,
) -> Tuple[List[Optional[np.ndarray]], int, bool]:
    """Extract contextual vectors for every turn in a dialogue.

    Returns:
        vectors: one vector (or None) per parsed turn
        token_count: tokenized full-dialogue length
        used_window_fallback: whether per-turn context windows were required
    """
    input_ids, attention_mask, offsets = tokenize_with_offsets(tokenizer, text)
    token_count = int(input_ids.shape[1])

    turn_token_positions: List[List[int]] = []
    for turn in turns:
        positions = char_span_to_token_positions(
            offsets, turn["start_char"], turn["end_char"]
        )
        turn_token_positions.append(positions)

    # Fast path: one causal forward pass for the whole dialogue.
    if token_count <= max_context:
        try:
            hidden = forward_hidden(
                model, capture, input_ids, attention_mask, device
            )
            vectors: List[Optional[np.ndarray]] = []
            for positions in turn_token_positions:
                vectors.append(pool_positions(hidden, positions) if positions else None)
            del hidden
            return vectors, token_count, False
        except RuntimeError as exc:
            # On a memory-constrained Mac, gracefully retry using smaller windows.
            message = str(exc).lower()
            if "memory" not in message and "mps" not in message and "alloc" not in message:
                raise
            print(
                f"    Whole-dialogue pass hit a device-memory error; "
                f"falling back to per-turn windows.",
                file=sys.stderr,
            )
            clear_device_cache(device)

    # Long-dialogue / memory fallback. Each target turn gets at most max_context
    # tokens of LEFT context and no future turns.
    vectors = []

    for positions in turn_token_positions:
        if not positions:
            vectors.append(None)
            continue

        target_start = min(positions)
        target_end = max(positions) + 1

        # End the window at the end of the current utterance, so there is no
        # future context. Keep as much preceding conversation as fits.
        window_end = target_end
        requested_context = max_context
        vector: Optional[np.ndarray] = None

        # Back off automatically if MPS cannot allocate the requested window.
        for context_limit in (requested_context, min(1024, requested_context), min(512, requested_context)):
            window_start = max(0, window_end - context_limit)

            if target_start < window_start:
                # An individual utterance longer than the window is extremely
                # unlikely here; keep its most recent tokens if it happens.
                local_positions = list(range(0, window_end - window_start))
            else:
                local_positions = [p - window_start for p in positions if p >= window_start]

            ids_slice = input_ids[:, window_start:window_end]
            mask_slice = attention_mask[:, window_start:window_end]

            try:
                hidden = forward_hidden(
                    model, capture, ids_slice, mask_slice, device
                )
                vector = pool_positions(hidden, local_positions)
                del hidden
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                if "memory" not in message and "mps" not in message and "alloc" not in message:
                    raise
                clear_device_cache(device)

        vectors.append(vector)

    return vectors, token_count, True


def load_dialogues(path: Path, limit: Optional[int] = None) -> List[Dict]:
    dialogues = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            dialogues.append(json.loads(line))
    return dialogues


def prepare_metadata(dialogues: Sequence[Dict], metadata_path: Path) -> Tuple[List[List[Dict]], int]:
    """Parse all turns once, assign stable row IDs, and write metadata."""
    parsed_by_dialogue: List[List[Dict]] = []
    row = 0

    with metadata_path.open("w", encoding="utf-8") as out:
        for dialogue_id, d in enumerate(dialogues):
            turns = parse_turns(d["text"])
            parsed_by_dialogue.append(turns)

            for turn_idx, turn in enumerate(turns):
                emotion = (
                    d.get("person_emotion")
                    if turn["speaker"] == "human"
                    else d.get("ai_emotion")
                )

                record = {
                    "row": row,
                    "dialogue_id": dialogue_id,
                    "dialogue_idx": d.get("dialogue_idx"),
                    "topic_idx": d.get("topic_idx"),
                    "topic": d.get("topic"),
                    "turn_idx": turn_idx,
                    "speaker": turn["speaker"],
                    "emotion": emotion,
                    "person_emotion": d.get("person_emotion"),
                    "ai_emotion": d.get("ai_emotion"),
                    "utterance": turn["utterance"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                row += 1

    return parsed_by_dialogue, row


def save_json(path: Path, obj: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one contextual residual-stream vector per Person/AI turn."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "emotion_dialogues.jsonl",
        help="Path to emotion_dialogues.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_turn_vectors",
        help="Directory for vectors, metadata, config, and progress",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Execution device; auto prefers MPS on Apple Silicon",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=f"Maximum tokens per forward pass (default: {DEFAULT_MAX_CONTEXT})",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Exact zero-based decoder layer. Default: approximately 2/3 depth.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N dialogues (useful for a quick test).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard an existing compatible run and start from dialogue 0.",
    )
    parser.add_argument(
        "--cache-every",
        type=int,
        default=25,
        help="Clear accelerator cache every N dialogues (default: 25).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.max_context < 128:
        raise ValueError("--max-context should be at least 128 tokens.")

    device = choose_device(args.device)
    dtype = choose_dtype(device)

    print("=== Contextual Dialogue Turn Vector Extraction ===")
    print(f"Input:  {input_path}")
    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    print(f"Max context per forward pass: {args.max_context} tokens")

    if device.type == "mps":
        print("Apple Silicon MPS detected.")
    elif device.type == "cpu":
        print("WARNING: MPS/CUDA not detected; CPU extraction will be much slower.")

    print("\nLoading dataset...")
    dialogues = load_dialogues(input_path, args.limit)
    print(f"Loaded {len(dialogues)} dialogues.")

    metadata_path = output_dir / "turn_metadata.jsonl"
    vectors_path = output_dir / "turn_vectors.npy"
    config_path = output_dir / "run_config.json"
    progress_path = output_dir / "progress.json"

    print("Parsing turns and preparing metadata...")
    parsed_by_dialogue, total_turns = prepare_metadata(dialogues, metadata_path)
    print(f"Found {total_turns} Person/AI turns.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "This script needs a fast tokenizer for offset mappings. "
            "Choose a model with a fast Hugging Face tokenizer."
        )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    model.to(device)

    layers = get_decoder_layers(model)
    num_layers = len(layers)
    target_layer = (
        args.layer
        if args.layer is not None
        else min(num_layers - 1, int(num_layers * DEFAULT_LAYER_FRACTION))
    )
    if not 0 <= target_layer < num_layers:
        raise ValueError(
            f"--layer {target_layer} is invalid; model has {num_layers} layers (0..{num_layers - 1})."
        )

    hidden_size = get_hidden_size(model)
    print(
        f"Model ready: {num_layers} decoder layers, hidden size {hidden_size}, "
        f"target layer {target_layer}."
    )

    capture = TargetLayerCapture(layers[target_layer])

    run_config = {
        "input": str(input_path),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "max_context": args.max_context,
        "num_dialogues": len(dialogues),
        "num_turns": total_turns,
        "num_layers": num_layers,
        "target_layer": target_layer,
        "hidden_size": hidden_size,
        "vector_dtype": "float16",
        "pooling": "mean over tokens belonging only to the target utterance",
        "context": "all preceding tokens when full dialogue fits; otherwise bounded left context",
    }

    start_dialogue = 0
    start_row = 0

    if args.restart:
        for path in (vectors_path, config_path, progress_path):
            if path.exists():
                path.unlink()

    if vectors_path.exists() and config_path.exists() and progress_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            old_config = json.load(f)

        compatibility_keys = [
            "model",
            "max_context",
            "num_dialogues",
            "num_turns",
            "target_layer",
            "hidden_size",
        ]
        incompatible = [
            key
            for key in compatibility_keys
            if old_config.get(key) != run_config.get(key)
        ]
        if incompatible:
            raise RuntimeError(
                "Existing output is incompatible with this run for: "
                + ", ".join(incompatible)
                + ". Use --restart or a different --output-dir."
            )

        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
        start_dialogue = int(progress.get("next_dialogue", 0))
        start_row = int(progress.get("next_row", 0))

        vectors_mm = np.lib.format.open_memmap(
            vectors_path,
            mode="r+",
            dtype=np.float16,
            shape=(total_turns, hidden_size),
        )
        print(
            f"Resuming at dialogue {start_dialogue}/{len(dialogues)}, "
            f"vector row {start_row}/{total_turns}."
        )
    else:
        save_json(config_path, run_config)
        vectors_mm = np.lib.format.open_memmap(
            vectors_path,
            mode="w+",
            dtype=np.float16,
            shape=(total_turns, hidden_size),
        )
        vectors_mm[:] = np.nan
        vectors_mm.flush()
        save_json(progress_path, {"next_dialogue": 0, "next_row": 0})

    # Validate row position implied by the parsed data.
    expected_start_row = sum(len(x) for x in parsed_by_dialogue[:start_dialogue])
    if start_row != expected_start_row:
        raise RuntimeError(
            f"Resume checkpoint row mismatch: progress says {start_row}, "
            f"but parsed data implies {expected_start_row}. Use --restart."
        )

    row = start_row
    fallback_dialogues = 0
    missing_vectors = 0

    print("\nExtracting vectors...")

    try:
        for dialogue_id in range(start_dialogue, len(dialogues)):
            d = dialogues[dialogue_id]
            turns = parsed_by_dialogue[dialogue_id]

            vectors, token_count, used_fallback = extract_dialogue_vectors(
                model=model,
                tokenizer=tokenizer,
                capture=capture,
                text=d["text"],
                turns=turns,
                device=device,
                max_context=args.max_context,
            )

            if used_fallback:
                fallback_dialogues += 1

            for vector in vectors:
                if vector is None:
                    missing_vectors += 1
                else:
                    vectors_mm[row] = vector
                row += 1

            vectors_mm.flush()
            save_json(
                progress_path,
                {
                    "next_dialogue": dialogue_id + 1,
                    "next_row": row,
                    "fallback_dialogues_in_this_session": fallback_dialogues,
                    "missing_vectors_in_this_session": missing_vectors,
                },
            )

            done = dialogue_id + 1
            if done % 10 == 0 or done == len(dialogues):
                print(
                    f"  [{done}/{len(dialogues)} dialogues] "
                    f"rows={row}/{total_turns} last_tokens={token_count} "
                    f"fallback_dialogues={fallback_dialogues} missing={missing_vectors}"
                )

            if args.cache_every > 0 and done % args.cache_every == 0:
                clear_device_cache(device)

    finally:
        capture.close()
        vectors_mm.flush()
        clear_device_cache(device)

    save_json(
        progress_path,
        {
            "next_dialogue": len(dialogues),
            "next_row": row,
            "complete": True,
            "fallback_dialogues_in_this_session": fallback_dialogues,
            "missing_vectors_in_this_session": missing_vectors,
        },
    )

    print("\n=== COMPLETE ===")
    print(f"Vectors:  {vectors_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Shape:    ({total_turns}, {hidden_size})")
    print("Row i in turn_vectors.npy corresponds to row i in turn_metadata.jsonl.")

    if missing_vectors:
        print(
            f"WARNING: {missing_vectors} rows could not be extracted and remain NaN."
        )


if __name__ == "__main__":
    main()
