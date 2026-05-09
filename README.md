# Fitness YouTube Comment Classifier

## Model Description
Fine-tuned `distilbert-base-uncased` for classifying YouTube comments from fitness 
influencer videos into 5 categories: fitness, nutrition, motivational, challenge, product.

## Dataset
- Self-scraped YouTube comments from 94 fitness influencer channels
- 92,223 total comments, 5,000 used for this training run
- Scraped using YouTube Data API v3 for dissertation research (2024)
- YouTubers include: Noel Deyzel, Browney, Jeff Nippard, Renaissance Periodization, ATHLEAN-X

## Training
- Base model: distilbert-base-uncased (66M parameters)
- Training samples: 3,500 | Validation: 750 | Test: 750
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Hardware: Kaggle T4 x2 GPU
- Training time: ~81 seconds

## Evaluation Results (Test Set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| challenge | 0.49 | 0.47 | 0.48 |
| fitness | 0.54 | 0.55 | 0.54 |
| motivational | 0.46 | 0.57 | 0.51 |
| nutrition | 0.67 | 0.57 | 0.62 |
| product | 0.58 | 0.51 | 0.54 |
| **overall** | | | **0.54** |

| Baseline | Accuracy |
|----------|----------|
| Majority class (always predict challenge) | 22.9% |
| Fine-tuned DistilBERT | 53.6% |
| Improvement | +30.7pp |

## Usage
```python
from transformers import pipeline
classifier = pipeline('text-classification', 
                      model='Krat6s/fitness-comment-classifier')
classifier("This protein shake recipe is amazing with oat milk")
# {'label': 'nutrition', 'score': 0.873}
```

## Limitations
- Challenge and motivational comments frequently confused — 
  these classes share workout encouragement language
- Product comments phrased as questions get misclassified as fitness
- Model trained on English comments only — non-English comments 
  (~15% of dataset) default to neutral/incorrect predictions
- Training on 5,000 of 92,223 available rows — performance 
  expected to improve with full dataset

## HuggingFace Links
- https://huggingface.co/Krat6s/fitness-comment-classifier
- https://huggingface.co/datasets/Krat6s/fitness-youtube-comments

## Next Steps
- Scale training to 20,000 rows
- Compare against roberta-base (125M parameters)
- YouTuber-stratified train/test split to test generalisation

