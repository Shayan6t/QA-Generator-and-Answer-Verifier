# RACE Reading Comprehension — AI Lab Project

## Project Summary
- **Dataset**: RACE (Kaggle: ankitdhiman7/race-dataset) — ~100K MCQs
- **Model A**: Q&A Generator / Answer Verifier (Supervised + Unsupervised + Ensemble)
- **Model B**: Distractor & Hint Generator (Ranking + Extraction)
- **UI**: Streamlit — 4 screens (Article Input, Quiz, Hints, Analytics Dashboard)
- **ML Stack**: One-Hot Encoding, TF-IDF (optional), Logistic Regression, SVM, Naive Bayes, Random Forest, K-Means, Label Propagation

---

## Commit Plan (15–20 Commits)

| #  | Commit Message                                      | Status      |
|----|-----------------------------------------------------|-------------|
| 1  | `Initial project structure & requirements.txt`      | ✅ Done     |
| 2  | `Add RACE dataset to data/raw/`                     | ⬜ Pending  |
| 3  | `EDA notebook — distributions & visualizations`     | ⬜ Pending  |
| 4  | `preprocessing.py — cleaning & feature engineering` | ⬜ Pending  |
| 5  | `Model A — Logistic Regression answer verifier`     | ⬜ Pending  |
| 6  | `Model A — SVM answer verifier`                     | ⬜ Pending  |
| 7  | `Model A — Unsupervised (K-Means + Label Prop)`     | ⬜ Pending  |
| 8  | `Model A — Ensemble (soft vote / stacking)`         | ⬜ Pending  |
| 9  | `Model A — Template-based question generation`      | ⬜ Pending  |
| 10 | `Model B — Distractor candidate extraction & ranking` | ⬜ Pending |
| 11 | `Model B — Hint extraction with ML scoring`         | ⬜ Pending  |
| 12 | `evaluate.py — all metrics for Model A & B`         | ⬜ Pending  |
| 13 | `inference.py — unified inference API`              | ⬜ Pending  |
| 14 | `UI — Screen 1 & 2 (Article Input + Quiz View)`    | ⬜ Pending  |
| 15 | `UI — Screen 3 (Hint Panel)`                        | ⬜ Pending  |
| 16 | `UI — Screen 4 (Analytics Dashboard)`               | ⬜ Pending  |
| 17 | `Unit tests & integration tests`                    | ⬜ Pending  |
| 18 | `README.md with full setup & run instructions`      | ⬜ Pending  |
| 19 | `Final polish — error handling, UX, caching`        | ⬜ Pending  |
| 20 | `Final report PDF & submission checklist`            | ⬜ Pending  |

---

## Dataset Schema
| Column | Type   | Description                          |
|--------|--------|--------------------------------------|
| id     | String | Unique identifier                    |
| article| String | Full reading passage                 |
| question| String| Multiple-choice question             |
| A–D    | String | Answer options                       |
| answer | String | Correct label (A, B, C, or D)        |

Files: `train.csv` (~87,866 rows), `test.csv`, `val.csv`

---

## Required Folder Structure
```
AI Project/
├── data/raw/                  # train.csv, test.csv, val.csv
├── models/model_a/traditional/
├── models/model_b/traditional/
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   └── evaluate.py
├── ui/
│   ├── app.py
│   └── components/
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
├── tests/test_inference.py
├── requirements.txt
├── README.md
└── report/
```

## Grading Breakdown (100 marks)
- EDA & Preprocessing: 10
- Model A — Traditional ML: 15
- Model A — Unsupervised/Semi-Supervised: 20
- Model A — Ensemble: 5
- Model B — Distractor Gen: 15
- Model B — Hint Gen: 10
- User Interface: 15
- Final Report: 5
- Code Quality: 5
