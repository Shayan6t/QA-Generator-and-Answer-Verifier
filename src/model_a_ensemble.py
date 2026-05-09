"""
model_a_ensemble.py — Combine Model A base classifiers (LR + SVM + NB) into:

  1. Soft voting   - average P(is_correct) across base models
  2. Stacking      - train a Logistic Regression meta-learner on the
                     [P_lr, P_svm, P_nb] vectors of the validation split,
                     then evaluate on the test split.

Reports binary AND MCQ-level accuracy for both ensembles.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")


def load_split(split: str):
    X = sparse.load_npz(os.path.join(PROCESSED_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(PROCESSED_DIR, f"y_{split}.npy"))
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}_verification.csv"))
    return X, y, df


def _clip_nonneg(X):
    X = X.copy()
    X.data = np.clip(X.data, 0.0, None)
    return X


def mcq_accuracy(probs: np.ndarray, df: pd.DataFrame) -> float:
    """Same MCQ metric as in model_a_train: argmax across each block of 4 options."""
    df = df.copy()
    df["score"] = probs
    n_q = len(df) // 4
    df["q_id"] = np.repeat(np.arange(n_q), 4)
    pred, gold = [], []
    for _, g in df.groupby("q_id", sort=False):
        i = g["score"].values.argmax()
        pred.append(g["option_label"].values[i])
        gold.append(g.loc[g["is_correct"] == 1, "option_label"].values[0])
    return float(np.mean(np.array(pred) == np.array(gold)))


def base_probs(model, X):
    """Return P(class=1) from a sklearn classifier."""
    return model.predict_proba(X)[:, 1]


def main():
    print("=== Loading base models ===")
    lr  = joblib.load(os.path.join(MODELS_DIR, "model_a_lr.joblib"))
    svm = joblib.load(os.path.join(MODELS_DIR, "model_a_svm.joblib"))
    nb  = joblib.load(os.path.join(MODELS_DIR, "model_a_nb.joblib"))

    print("=== Loading splits ===")
    X_val,  y_val,  val_df  = load_split("val")
    X_test, y_test, test_df = load_split("test")

    # NB needs clipped inputs
    X_val_nb  = _clip_nonneg(X_val)
    X_test_nb = _clip_nonneg(X_test)

    print("=== Generating base-model probabilities ===")
    p_val = np.column_stack([
        base_probs(lr,  X_val),
        base_probs(svm, X_val),
        base_probs(nb,  X_val_nb),
    ])
    p_test = np.column_stack([
        base_probs(lr,  X_test),
        base_probs(svm, X_test),
        base_probs(nb,  X_test_nb),
    ])

    # ── 1. SOFT VOTING ───────────────────────────────────────────────────
    print("\n=== Soft Voting (mean of base probabilities) ===")
    soft_val  = p_val.mean(axis=1)
    soft_test = p_test.mean(axis=1)

    soft_val_pred  = (soft_val  >= 0.5).astype(int)
    soft_test_pred = (soft_test >= 0.5).astype(int)

    soft_val_mcq  = mcq_accuracy(soft_val,  val_df)
    soft_test_mcq = mcq_accuracy(soft_test, test_df)

    print(f"  VAL  | binary acc {accuracy_score(y_val,  soft_val_pred):.4f}"
          f"  | F1 macro {f1_score(y_val,  soft_val_pred,  average='macro', zero_division=0):.4f}"
          f"  | MCQ acc {soft_val_mcq:.4f}")
    print(f"  TEST | binary acc {accuracy_score(y_test, soft_test_pred):.4f}"
          f"  | F1 macro {f1_score(y_test, soft_test_pred, average='macro', zero_division=0):.4f}"
          f"  | MCQ acc {soft_test_mcq:.4f}")

    # ── 2. STACKING (meta = LR over base probabilities) ──────────────────
    print("\n=== Stacking (meta-LR on base probs, fit on VAL, eval on TEST) ===")
    meta = LogisticRegression(C=1.0, max_iter=500)
    meta.fit(p_val, y_val)
    print(f"  Meta-LR coefs: {dict(zip(['lr','svm','nb'], np.round(meta.coef_[0], 4)))}")

    stack_test_proba = meta.predict_proba(p_test)[:, 1]
    stack_test_pred  = (stack_test_proba >= 0.5).astype(int)
    stack_test_mcq   = mcq_accuracy(stack_test_proba, test_df)

    print(f"  TEST | binary acc {accuracy_score(y_test, stack_test_pred):.4f}"
          f"  | F1 macro {f1_score(y_test, stack_test_pred, average='macro', zero_division=0):.4f}"
          f"  | MCQ acc {stack_test_mcq:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, stack_test_pred)}")

    # ── Persist ──────────────────────────────────────────────────────────
    joblib.dump(meta, os.path.join(MODELS_DIR, "model_a_stack_meta.joblib"))

    # Append summary to results log
    rows = [
        {"model": "ensemble_softvote",
         "val_mcq_acc": soft_val_mcq, "test_mcq_acc": soft_test_mcq,
         "test_binary_acc": accuracy_score(y_test, soft_test_pred),
         "test_binary_f1":  f1_score(y_test, soft_test_pred, average="macro", zero_division=0)},
        {"model": "ensemble_stacking",
         "test_mcq_acc": stack_test_mcq,
         "test_binary_acc": accuracy_score(y_test, stack_test_pred),
         "test_binary_f1":  f1_score(y_test, stack_test_pred, average="macro", zero_division=0)},
    ]
    log_path = os.path.join(MODELS_DIR, "results_log.csv")
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        keep = existing[~existing["model"].isin([r["model"] for r in rows])]
        out = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    else:
        out = pd.DataFrame(rows)
    out.to_csv(log_path, index=False)
    print(f"\nUpdated {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
