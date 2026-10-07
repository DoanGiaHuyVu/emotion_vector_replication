# Emotion Vector Replication

Scripts and saved results for extracting contextual dialogue activations, learning emotion directions, and comparing emotion representation across small language models.

## Project contents

- `generating_story_dialogue.py`: generate emotion-labeled synthetic dialogues through OpenRouter using `OPENROUTER_API_KEY` from the environment.
- `emotion_dialogues.jsonl`: saved synthetic dialogue dataset.
- `extract_dialogue_turn_vectors_macos.py`: extract one contextual vector per dialogue turn, with support for Apple Silicon MPS, CPU, and CUDA.
- `extract_emotion_directions_macos.py`: learn emotion directions with held-out splits.
- `analyze_emotion_research.py`: analyze geometry, uncertainty, generalization, and cross-model comparisons.
- `analysis_manifest.json`: configure the six saved model runs.
- `analysis/` and `results_*/`: saved reports, metadata, configurations, and smaller vector artifacts.

## Running the scripts

Use a Python environment with NumPy, PyTorch, Transformers, Accelerate, and SentencePiece installed. Each extraction and analysis script provides command-line options through `--help`. Dialogue generation also requires `curl` and an OpenRouter API key.

Full-run `results_*_full/turn_vectors.npy` arrays are excluded from Git because each exceeds GitHub's regular file-size limit. The original arrays remain in the local project folder. Regenerate these arrays before rerunning analyses from a fresh clone, using the model and layer recorded in each run's `run_config.json`.

For example, regenerate the Qwen3 run:

```sh
python extract_dialogue_turn_vectors_macos.py \
  --input emotion_dialogues.jsonl \
  --model Qwen/Qwen3-1.7B \
  --layer 18 \
  --output-dir results_qwen3_1.7b_full \
  --restart
```

After all required full-run arrays are available:

```sh
python analyze_emotion_research.py \
  --manifest analysis_manifest.json \
  --output-dir analysis/all_models
```

Use the same model and layer for contextual vectors and their corresponding emotion directions. See the script docstrings for analysis limitations.
