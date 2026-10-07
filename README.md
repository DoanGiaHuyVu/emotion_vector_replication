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

## Interpretation in relation to the Anthropic paper

This project is a **partial adaptation and cross-model extension**, rather than a full replication of [*Emotion Concepts and their Function in a Large Language Model* (Sofroniew et al., 2026, v1)](https://arxiv.org/html/2604.07729v1). The paper studies Claude Sonnet 4.5 and distinguishes emotion representations from evidence that those representations influence behavior. Our strongest evidence concerns representation on a shared synthetic corpus.

### Which experiments are comparable?

| Aspect | Anthropic paper | This project |
| --- | --- | --- |
| Direction construction | 171 story-based emotion concepts; centered means with neutral-PC nuisance removal ([§1.1](https://arxiv.org/html/2604.07729v1#S1.SS1)). | Seven dialogue-derived directions; neutral is a target class and its PCs are retained. |
| Generalization | External corpora, implicit emotional scenarios, and controlled semantic changes ([§1.2](https://arxiv.org/html/2604.07729v1#S1.SS2)). | Held-out dialogues and topic-group CV within the same synthetic corpus. |
| Speaker binding | Separate present-speaker and other-speaker probes ([§2.3](https://arxiv.org/html/2604.07729v1#S2.SS3)). | Current-speaker labels, pooled directions, and separate Person/AI analysis subsets. |
| Geometry | Human valence/arousal comparisons using 45 overlapping concepts ([§2.1.2](https://arxiv.org/html/2604.07729v1#S2.SS1.SSS2)). | Seven-direction PCA and comparisons across six models. |
| Behavioral function | Steering interventions on preferences and alignment-related behavior ([§1.3](https://arxiv.org/html/2604.07729v1#S1.SS3), [Part 3](https://arxiv.org/html/2604.07729v1#S3)). | Observational analysis of saved activations; no steering interventions. |

The closest methodological comparison is the paper's dialogue-based speaker analysis. However, this extractor feeds the raw fictional dialogue text to each model and averages each utterance's token activations. It does not measure a model-generated assistant response or retain token-level trajectories. The `ai` subset therefore concerns a fictional AI speaker's text, rather than evidence that the evaluated model itself experiences or enacts that emotion.

### Emotion information is present, but the probe is not a calibrated classifier

The held-out AUROCs of **0.8859–0.9107** support the narrow claim that the training-derived directions rank matching emotional content on new dialogues. `Neutral` is especially easy: its per-class AUROC ranges from **0.9976 to 0.9997**. Averaging only the other six saved one-vs-rest AUROCs yields **0.8672–0.8958**, so the signal remains substantial without that easy class. This calculation changes the averaging set, not the fitted directions or evaluation examples.

Phi provides a concrete explanation for the discrepancy between AUROC and multiclass accuracy: it predicts **sadness for 8,390 of 9,109 held-out turns (92.11%)**, although only **1,486 turns (16.31%)** have that label. A direction can rank positive examples well while its absolute projection scores dominate competing directions. This is consistent with a cross-direction score offset or calibration problem; the reports do not establish the cause. Training-only centering or calibration is an experiment to test, rather than a correction already validated here. Phi's topic-group centroid macro F1 of **0.6660** also shows that its poor direction-argmax score does not imply an absence of class information in its activations.

The lexical baseline is a more demanding interpretive control than random guessing. On topic-group CV, SmolLM2 and Phi exceed its macro F1 by only **0.0121** and **0.0137**, respectively; the other four activation classifiers score below it. These differences are descriptive, without paired uncertainty estimates. The data therefore supports accessible emotion-label information, but provides limited evidence that the tested activation classifiers capture information beyond the corpus's lexical and stylistic cues.

### Shared geometry survives a neutral-class sensitivity check

High RSA could reflect both shared emotion structure and shared dataset biases. To test one obvious source of agreement, we dropped all six relationships involving neutral from each saved seven-class cosine matrix and recomputed Pearson RSA over the remaining **15 emotion pairs**. Mean cross-model direction RSA decreases only from **0.9776 to 0.9692**; contextual RSA remains **0.9788 for AI turns** and **0.9707 for Person turns**. Thus, shared geometry is not explained solely by the neutral relationships. This is a sensitivity check on fixed vectors, not a six-class refit: their original centering still includes neutral. Calculations and prediction counts are recorded in [paper_comparison_sensitivity.json](analysis/paper_comparison_sensitivity.json).

The PCA result deserves more restraint. In our runs, PC1 alone explains **59.69–68.76%** of direction variance, and the neutral direction has a norm **2.19–2.48 times** that of the largest other direction. These observations are consistent with a strong neutral-versus-emotional contrast, but identifying the axis requires inspecting its loadings. The paper's illustrative PCs explain **26% and 15%**, with human-rating validation; those percentages are not directly comparable to a seven-class, differently constructed set ([§2.1.2](https://arxiv.org/html/2604.07729v1#S2.SS1.SSS2)). Exceeding an isotropic random-vector null shows concentrated geometry, but cannot identify its axes as valence and arousal.

### What remains necessary for a stronger replication?

The paper emphasizes locally operative emotion concepts rather than a continuously active emotional state ([§2.2.3](https://arxiv.org/html/2604.07729v1#S2.SS2.SSS3)). Our turn averages and fixed dialogue labels cannot distinguish those alternatives or demonstrate separate present/other-speaker representations. More importantly, the paper's functional interpretation depends on behavioral interventions; classification and RSA alone do not supply that evidence.

The most informative next experiments are:

1. **Independent transfer:** freeze the dialogue-trained directions and evaluate on independently labeled natural conversations and implicit-emotion scenarios. Include controls that change semantic implications while keeping wording similar, and compare against the lexical baseline.
2. **Speaker and position controls:** construct the full two-by-two speaker-label/token-position probe grid; compare raw transcripts with model-native chat formatting, response-boundary activations, token trajectories, and generated responses across layers.
3. **Geometry validation:** expand the emotion vocabulary, compare axes with independent human valence/arousal ratings, and test nuisance removal using a separate neutral corpus without discarding the neutral ERC target.
4. **Causal evaluation:** test positive and negative steering strengths against unsteered, norm-matched random-direction, and unrelated-concept controls. Measure behavior and task quality on independent scenarios, rather than only changes in emotional word choice.

Together, the current results make emotion directions plausible candidates for further testing in small models. They extend the comparison across model families, while leaving independent semantic generalization and causal behavioral function unresolved.

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
