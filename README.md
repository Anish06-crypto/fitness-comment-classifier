# Fitness YouTube Comment Classifier — RoBERTa

Fine-tuned `roberta-base` that classifies YouTube comments from fitness influencer videos into 5 categories: `fitness`, `nutrition`, `motivational`, `challenge`, `product`.

Part of a three-experiment study measuring the effect of data volume and model size on a self-scraped fitness influencer comment dataset.

---

## Quick Start

```python
from transformers import pipeline

classifier = pipeline(
    'text-classification',
    model='Krat6s/fitness-comment-classifier-roberta'
)

classifier("This protein shake changed my life, amazing with oat milk")
# [{'label': 'nutrition', 'score': 0.956}]

classifier("I've been doing this workout for 30 days and I can see abs forming!")
# [{'label': 'fitness', 'score': 0.965}]
```

---

## Model Description

- **Base model:** `roberta-base` (FacebookAI, 125M parameters)
- **Task:** Multi-class text classification (5 classes)
- **Domain:** YouTube comments from fitness influencer channels
- **Language:** English (non-English comments present in dataset but not handled)

---

## Dataset

Self-scraped YouTube comments collected via the YouTube Data API v3 for MSc dissertation research on fitness influencer sentiment and thematic analysis.

- **Total dataset:** 92,223 comments across 94 fitness influencer channels
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
| 1 | 2.495 | 2.126 | 0.592 | 0.595 |
| 2 | 1.934 | 2.059 | 0.607 | 0.609 |
| 3 | 1.638 | 2.102 | 0.614 | 0.614 |

**Hardware:** Kaggle T4 x2 GPU
**Training time:** 643 seconds (~10.7 minutes)

---

## Evaluation Results (Test Set — 3,000 samples)

### Overall

| Metric | Score |
|--------|-------|
| Accuracy | 62.5% |
| F1 (weighted) | 62.5% |

### Per-Class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| challenge | 0.62 | 0.58 | 0.60 | 685 |
| fitness | 0.63 | 0.67 | 0.65 | 647 |
| motivational | 0.56 | 0.66 | 0.61 | 641 |
| nutrition | 0.69 | 0.65 | 0.67 | 671 |
| product | 0.65 | 0.53 | 0.58 | 356 |

### Baseline Comparisons

| Model | Accuracy |
|-------|----------|
| Majority class baseline | 22.8% |
| Pretrained RoBERTa (no fine-tuning) | 21.6% |
| Fine-tuned RoBERTa (this model) | 62.5% |
| Improvement over baseline | +39.7pp |
| Improvement from fine-tuning | +40.9pp |

---

## Experiment Comparison — Data Scaling + Model Scaling

Three experiments run on the same dataset and evaluation pipeline, changing one variable at a time.

| Model | Parameters | Training Data | Accuracy | F1 | Train Time |
|-------|-----------|--------------|----------|----|------------|
| DistilBERT | 66M | 5,000 rows | 53.6% | 53.8% | 81s |
| DistilBERT | 66M | 20,000 rows | 60.4% | 60.4% | 327s |
| RoBERTa (this model) | 125M | 20,000 rows | 62.5% | 62.5% | 643s |

**Key findings:**
- Data scaling (5K → 20K rows): +6.8pp accuracy, 4x training time
- Model scaling (DistilBERT → RoBERTa): +2.1pp accuracy, 2x training time
- Data volume had a larger impact than model size on this task

---

## Per-Class F1 Across All Experiments

| Class | DistilBERT 5K | DistilBERT 20K | RoBERTa 20K |
|-------|--------------|----------------|-------------|
| challenge | 0.48 | 0.60 | 0.60 |
| fitness | 0.54 | 0.63 | 0.65 |
| motivational | 0.51 | 0.58 | 0.61 |
| nutrition | 0.62 | 0.63 | 0.67 |
| product | 0.54 | 0.56 | 0.58 |

---

## Inference Examples

| Comment | Predicted | Confidence |
|---------|-----------|------------|
| "This protein shake recipe changed my life, tastes amazing with oat milk" | nutrition | 95.6% |
| "I've been doing this workout for 30 days and I can see abs forming!" | fitness | 96.5% |
| "Never give up on your dreams, the grind is worth it" | motivational | 86.0% |
| "Is this pre-workout worth buying? I've heard mixed reviews" | product | 90.6% |
| "Day 7 of the squat challenge complete 🔥" | fitness ✗ | 89.3% |

Note: the final example is a known failure case. "Day 7 of the squat challenge" is correctly a challenge comment, but RoBERTa predicts fitness at high confidence. "Squat" has strong fitness associations in the training data. This illustrates a known failure mode of larger models — higher confidence on incorrect predictions. DistilBERT correctly predicted challenge here at lower confidence (50.8%).

---

## Limitations

**Challenge/motivational confusion** persists across all three model variants. 129 challenge comments were predicted as motivational in the test set despite the larger model and more training data. This is a label ambiguity problem intrinsic to the task — challenge and motivational videos share workout encouragement language. The confusion is not resolvable by more data or a larger model without incorporating video title or metadata alongside the comment text.

**Product class underrepresentation** — product has roughly half the examples of other classes. F1 of 0.58 is the lowest across classes despite competitive precision (0.65), driven by low recall (0.53) — the model misses nearly half of actual product comments.

**High-confidence errors** — RoBERTa's stronger language associations produce higher confidence scores overall, including on incorrect predictions. The challenge → fitness misclassification at 89.3% confidence is an example.

**Non-English comments** — approximately 15% of the dataset contains non-English comments. These produce unreliable predictions.

---

## Next Steps

- YouTuber-stratified train/test split — train on 80 channels, test on 14 held-out channels to measure generalisation to unseen creators
- Sentiment classification using human-labelled subset to replace VADER dissertation baseline
- Incorporate video title as additional input feature to resolve challenge/motivational ambiguity

---

## Citation

```
Dataset: Self-scraped YouTube comments from 94 fitness influencer channels
Collected via YouTube Data API v3 for MSc dissertation research
HuggingFace dataset: Krat6s/fitness-youtube-comments
```

---

## HuggingFace Roberta Fine-Tuned model link
- https://huggingface.co/Krat6s/fitness-comment-classifier-roberta

---

## Related Models

- [Krat6s/fitness-comment-classifier](https://huggingface.co/Krat6s/fitness-comment-classifier) — DistilBERT version trained on 20K rows (60.4% accuracy)

