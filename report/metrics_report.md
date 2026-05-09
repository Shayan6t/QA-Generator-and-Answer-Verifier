# RACE Reading Comprehension - Final Metrics Report

Generated automatically by `src/evaluate.py`.


## Model A - Answer Verification (TEST split)

| model     |   binary_acc |   binary_f1m |   precision |   recall |   mcq_acc |
|:----------|-------------:|-------------:|------------:|---------:|----------:|
| LR        |       0.7495 |       0.4298 |      0.3065 |   0.0014 |    0.3516 |
| SVM       |       0.75   |       0.4286 |      0      |   0      |    0.3543 |
| NB        |       0.7499 |       0.4287 |      0.2857 |   0.0002 |    0.3452 |
| Soft-Vote |       0.75   |       0.4287 |      1      |   0.0002 |    0.3608 |

_MCQ accuracy = pick option with highest P(correct) per question (4 options); random baseline = 25%._


## Model A - Unsupervised, Semi-Supervised, Question Generation

| model             |   val_binary_acc |   val_binary_f1 |   test_binary_acc |   test_binary_f1 |   val_mcq_acc |   test_mcq_acc |   purity |   silhouette |   val_acc |   val_f1_macro |   labeled_n |   unlabeled_n |   val_auc |
|:------------------|-----------------:|----------------:|------------------:|-----------------:|--------------:|---------------:|---------:|-------------:|----------:|---------------:|------------:|--------------:|----------:|
| kmeans_k4         |              nan |             nan |            nan    |         nan      |      nan      |       nan      |     0.75 |       0.0206 |  nan      |       nan      |         nan |           nan |  nan      |
| kmeans_k8         |              nan |             nan |            nan    |         nan      |      nan      |       nan      |     0.75 |      -0.0302 |  nan      |       nan      |         nan |           nan |  nan      |
| label_propagation |              nan |             nan |            nan    |         nan      |      nan      |       nan      |   nan    |     nan      |    0.7414 |         0.4431 |        3000 |          7000 |  nan      |
| ensemble_softvote |              nan |             nan |              0.75 |           0.4287 |        0.3576 |         0.3608 |   nan    |     nan      |  nan      |       nan      |         nan |           nan |  nan      |
| ensemble_stacking |              nan |             nan |              0.75 |           0.4287 |      nan      |         0.3567 |   nan    |     nan      |  nan      |       nan      |         nan |           nan |  nan      |
| qg_ranker_rf      |              nan |             nan |            nan    |         nan      |      nan      |       nan      |   nan    |     nan      |    0.8032 |       nan      |         nan |           nan |    0.7941 |

## Model B - Distractor & Hint Generation

| model              |   val_acc |   val_prec |   val_recall |   val_f1 |   val_auc |   test_precision_overlap |   test_recall_overlap |   test_f1_overlap |   top1_not_answer_acc |   diversity |   test_precision_at_1 |   test_precision_at_3 |   n_evaluated |
|:-------------------|----------:|-----------:|-------------:|---------:|----------:|-------------------------:|----------------------:|------------------:|----------------------:|------------:|----------------------:|----------------------:|--------------:|
| model_b_ranker_rf  |    0.9797 |     0.9804 |       0.9583 |   0.9692 |     0.997 |                   nan    |                nan    |            nan    |                   nan |    nan      |              nan      |              nan      |           nan |
| model_b_end_to_end |  nan      |   nan      |     nan      | nan      |   nan     |                     0.16 |                  0.16 |              0.16 |                     1 |      0.9469 |              nan      |              nan      |           nan |
| hint_ranker_lr     |  nan      |   nan      |     nan      | nan      |   nan     |                   nan    |                nan    |            nan    |                   nan |    nan      |                0.5608 |                0.7804 |           296 |

## Notes

- Binary accuracy looks high (~75%) because 75% of (article, question, option) rows are negatives by construction (1 correct of 4). The MCQ accuracy is the meaningful metric.

- Soft voting beats every individual base model on MCQ accuracy.

- Distractor end-to-end overlap F1 is low because RACE gold distractors are full sentences while the classical extractor produces short noun phrases - this is an expected limitation of the no-NLP-tools, no-neural-models constraint.
