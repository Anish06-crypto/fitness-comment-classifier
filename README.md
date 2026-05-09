# Fitness YouTube Comment Classifier

Fine-tuned `distilbert-base-uncased` that classifies YouTube comments from fitness influencer videos into 5 categories: `fitness`, `nutrition`, `motivational`, `challenge`, `product`.

---

## Quick Start

```python
from transformers import pipeline

classifier = pipeline(
    'text-classification',
    model='Krat6s/fitness-comment-classifier'
)

classifier("This protein shake changed my life, amazing with oat milk")
# [{'label': 'nutrition', 'score': 0.956}]

classifier("Day 7 of the squat challenge complete 🔥")
# [{'label': 'challenge', 'score': 0.508}]
```

---

## Model Description

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Task:** Multi-class text classification (5 classes)
- **Domain:** YouTube comments from fitness influencer channels
- **Language:** English (non-English comments present in dataset but not handled)

---

## Dataset

Self-scraped YouTube comments collected via the YouTube Data API v3 for MSc dissertation research on fitness influencer sentiment and thematic analysis.

- **Total dataset size:** 92,223 comments
- **YouTubers:** 94 fitness influencer channels
- **Top channels:** Noel Deyzel, Browney, Jeff Nippard, Renaissance Periodization, ATHLEAN-X
- **HuggingFace dataset:** [Krat6s/fitness-youtube-comments](https://huggingface.co/datasets/Krat6s/fitness-youtube-comments)

### Class Distribution (Full Dataset)

| Class | Count |
|-------|-------|
| challenge | 20,923 |
| nutrition | 20,506 |
| fitness | 19,990 |
| motivational | 19,928 |
| product | 10,749 |

---

## Training

### Data Splits (20,000 row stratified sample)

| Split | Size |
|-------|------|
| Train | 14,000 |
| Validation | 3,000 |
| Test | 3,000 |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Max sequence length | 128 |
| Warmup steps | 50 |
| Weight decay | 0.01 |
| Optimizer | AdamW |

### Training Curve

| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|-------|-----------|----------|----------|----|
| 1 | 2.594 | 2.262 | 0.549 | 0.553 |
| 2 | 2.037 | 2.150 | 0.581 | 0.583 |
| 3 | 1.757 | 2.163 | 0.583 | 0.584 |

**Hardware:** Kaggle T4 x2 GPU  
**Training time:** 327 seconds (~5.5 minutes)

---

## Evaluation Results (Test Set — 3,000 samples)

### Overall

| Metric | Score |
|--------|-------|
| Accuracy | 60.4% |
| F1 (weighted) | 60.4% |

### Per-Class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| challenge | 0.61 | 0.58 | 0.60 | 685 |
| fitness | 0.64 | 0.62 | 0.63 | 647 |
| motivational | 0.53 | 0.64 | 0.58 | 641 |
| nutrition | 0.64 | 0.62 | 0.63 | 671 |
| product | 0.63 | 0.50 | 0.56 | 356 |

### Baseline Comparison

| Model | Accuracy |
|-------|----------|
| Majority class (always predict challenge) | 22.8% |
| Fine-tuned DistilBERT (20K rows) | 60.4% |
| Improvement | +37.6pp |

---

## Data Scaling Results

Trained two versions to measure the effect of data volume:

| Model | Training Data | Accuracy | F1 |
|-------|--------------|----------|----|
| DistilBERT | 5,000 rows | 53.6% | 53.8% |
| DistilBERT | 20,000 rows | 60.4% | 60.4% |

4x more data produced +6.8pp accuracy improvement with linear increase in training time (81s → 327s).

---

## Inference Examples

| Comment | Predicted | Confidence |
|---------|-----------|------------|
| "This protein shake recipe changed my life, tastes amazing with oat milk" | nutrition | 95.6% |
| "I've been doing this workout for 30 days and I can see abs forming!" | fitness | 93.1% |
| "Never give up on your dreams, the grind is worth it" | motivational | 79.8% |
| "Is this pre-workout worth buying? I've heard mixed reviews" | product | 81.4% |
| "Day 7 of the squat challenge complete 🔥" | challenge | 50.8% |

---

## Limitations

**Challenge/motivational confusion** is the largest source of error (136 challenge comments predicted as motivational in the 20K test set). Both classes share workout encouragement language. Without video context, the boundary is inherently ambiguous — this is a label ambiguity issue, not a data quantity issue. More training data did not reduce this confusion proportionally.

**Product/nutrition overlap** is the second largest error pattern (64 nutrition comments predicted as product). Supplement and protein content sits on the boundary between these two classes.

**Non-English comments** default to incorrect predictions. Approximately 15% of the full dataset contains non-English comments (Russian, German, Spanish, Korean). VADER assigned these neutral scores in the original dissertation — DistilBERT similarly has no signal for them.

**Product class underrepresentation** — product has roughly half the examples of other classes in the full dataset. Despite this, product achieved competitive F1 (0.56) due to distinctive brand/review vocabulary.

---

## Next Steps

- Fine-tune `roberta-base` (125M parameters) on 20K rows — model size comparison
- YouTuber-stratified train/test split — test generalisation to unseen creators
- Sentiment classification using human-labelled subset replacing VADER baseline

---

## Citation

If you use this model or dataset, please credit:

```
Dataset: Self-scraped YouTube comments from 94 fitness influencer channels
Collected via YouTube Data API v3 for MSc dissertation research
HuggingFace: Krat6s/fitness-youtube-comments
```

## HuggingFace Links
- https://huggingface.co/Krat6s/fitness-comment-classifier
- https://huggingface.co/datasets/Krat6s/fitness-youtube-comments


