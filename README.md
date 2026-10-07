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

## Results

These results summarize the saved runs in [analysis/all_models_summary.json](analysis/all_models_summary.json). All six models processed the same **4,500 synthetic dialogues and 58,555 turns**, with identical metadata fingerprints and no nonfinite activation rows. The dialogues contain fictional Person and AI speakers; the labels describe their prompted emotional states.

The main evaluation uses seven classes (ERC-7): **anger, disgust, fear, joy, neutral, sadness, and surprise**. This retains **45,777 turns**; the additional `peaceful` and `powerful` labels account for the remaining 21.82% of turns and are included in separate nine-state contextual analyses.

### Held-out emotion directions

Directions were learned from **36,668 turns in 3,417 dialogues**, with evaluation on **9,109 turns in 851 different dialogues**. All models use the same dialogue split (seed 42), with zero overlapping train/evaluation rows. Each training `(dialogue, speaker)` unit is averaged before computing class means, and directions are centered by the equal-weight mean of the seven class means. Both speaker types are pooled for the results below.

The primary score is **macro one-vs-rest AUROC**, computed from continuous projections of turn activations onto unit emotion directions. Accuracy and macro F1 use the highest projection as the predicted class. Layer indices are zero-based.

| Model | Layer | Macro AUROC | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 18 | 0.8965 | 50.96% | 0.5117 |
| Qwen2.5-3B-Instruct | 24 | 0.8933 | 50.86% | 0.5042 |
| Granite-3.3-2B-Instruct | 26 | 0.8859 | 62.00% | 0.6300 |
| SmolLM2-1.7B-Instruct | 16 | 0.9107 | 49.81% | 0.5155 |
| Llama-3.2-3B-Instruct | 18 | 0.9003 | 62.13% | 0.6363 |
| Phi-3.5-mini-instruct | 21 | 0.9100 | 22.35% | 0.1732 |

All models show useful within-class ranking on this synthetic dataset. SmolLM2 has the highest AUROC point estimate, while Llama has the highest argmax accuracy and macro F1. Phi illustrates why these measures must be reported separately: high one-vs-rest AUROC does not guarantee that scores are calibrated across emotion directions for multiclass prediction. These point estimates do not establish statistically significant differences between models. Per-class AUROCs, dialogue-bootstrap 95% confidence intervals, and confusion matrices are available in each model's `analysis/<model>/turn_analysis.json`.

### Grouped classification and lexical control

A separate nearest-centroid classifier is evaluated across the full ERC-7 subset using cross-validation grouped by dialogue or by topic. Centering and class centroids are fitted within each training fold. The table reports **macro F1** and includes a text-only multinomial naive Bayes baseline with up to 5,000 unigram features.

| Representation / baseline | Dialogue-group CV | Topic-group CV |
| --- | ---: | ---: |
| Qwen3-1.7B activations | 0.6358 | 0.6359 |
| Qwen2.5-3B activations | 0.6262 | 0.6239 |
| Granite-3.3-2B activations | 0.6037 | 0.6033 |
| SmolLM2-1.7B activations | 0.6660 | 0.6644 |
| Llama-3.2-3B activations | 0.6426 | 0.6425 |
| Phi-3.5-mini activations | 0.6672 | 0.6660 |
| Text-only unigram baseline | 0.6593 | 0.6523 |

Topic-group results are close to dialogue-group results for these saved runs. The strong [lexical baseline](analysis/dataset_lexical_control.json) shows that word choice and style already expose much of the synthetic emotion signal. This limits what activation classification alone can establish about emotion representation. The grouped-CV classifier uses a different scoring procedure and evaluation protocol from the held-out direction probes above.

### Cross-model geometry

Representational similarity analysis (RSA) correlates the **21 pairwise relationships among seven emotions**, or 36 relationships for nine states, across models. It compares relational geometry rather than directly comparing coordinates in different hidden spaces. The values below average over the 15 distinct model pairs.

| Geometry | Mean Pearson RSA | Mean Spearman RSA |
| --- | ---: | ---: |
| ERC-7 contextual centroids, AI turns | 0.9721 | 0.9556 |
| ERC-7 contextual centroids, Person turns | 0.9675 | 0.9620 |
| Nine-state contextual centroids, AI turns | 0.9681 | 0.9443 |
| Nine-state contextual centroids, Person turns | 0.9730 | 0.9633 |
| ERC-7 training-derived emotion directions | 0.9776 | 0.9810 |

The models have strongly aligned emotion relationships on this shared corpus. Full matrices are available in [contextual ERC-7 AI RSA](analysis/contextual_erc7_ai.json), [Person RSA](analysis/contextual_erc7_human.json), and [emotion-direction RSA](analysis/independent_emotion_direction_rsa.json). Despite the last file's `independent` name, its direction provenance is the training split of the same synthetic dialogue corpus, so it is not an independent-corpus replication.

Across models, same-emotion turns have higher centered cosine similarity than different-emotion turns, with gaps of **0.0768–0.0894** and dialogue-bootstrap 95% intervals above zero. The first two principal components explain **74.67–79.81%** of variance among the seven emotion directions, exceeding the saved random-vector null's 95th percentile (**35.61–36.17%**). These are exploratory geometry findings: seven concepts and no saved comparison against independent human valence/arousal ratings do not establish a psychological circumplex.

### Scope of the findings

The saved results support emotion-label separability and convergent relational geometry on this synthetic corpus. They do not establish felt emotion, generalization to natural conversations, or causal control of model behavior. Directions are derived from contextual training activations, and causal steering was not tested. Only one layer per model is represented here; layer sweeps, independent corpora, and intervention experiments remain additional work.

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
