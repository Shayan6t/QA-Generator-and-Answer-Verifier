"""
model_a_train.py — Train classical ML answer-verifier(s) for Model A.

Tasks:
  1. Load pre-built verification features (X, y) from data/processed/.
  2. Train classifier(s) chosen via --model flag (default: lr).
  3. Evaluate at TWO levels:
       a) Binary verification:  is_correct vs predicted on (article, question, option) triples
       b) MCQ accuracy:         pick the option with highest P(correct) per question; match gold answer
  4. Persist the trained model under models/model_a/traditional/.

Supported models (one per commit, accumulated over commits 5-7-8):
  * lr   — Logistic Regression       (Commit 5)
  * svm  — Linear SVM (calibrated)   (Commit 6)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Data loading ─────────────────────────────────────────────────────────
def load_features(split: str):
    X = sparse.load_npz(os.path.join(PROCESSED_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(PROCESSED_DIR, f"y_{split}.npy"))
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}_verification.csv"))
    return X, y, df


# ── Evaluation ───────────────────────────────────────────────────────────
def evaluate_binary(y_true, y_pred, name: str):
    print(f"\n[{name}] Binary verification metrics")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 (bin) : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 macro : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")


def evaluate_mcq(probs: np.ndarray, df: pd.DataFrame, name: str):
    """
    MCQ-level accuracy: every 4 consecutive rows (in option order A,B,C,D)
    correspond to one original question.  Pick the option with the highest
    P(is_correct=1); compare to the gold label.
    """
    df = df.copy()
    df["score"] = probs

    # Each question has 4 rows in order A,B,C,D as built by explode_to_verification
    n_questions = len(df) // 4
    df["q_id"] = np.repeat(np.arange(n_questions), 4)

    pred_labels, gold_labels = [], []
    for _, group in df.groupby("q_id", sort=False):
        # Max-score option label
        best_idx = group["score"].values.argmax()
        pred_labels.append(group["option_label"].values[best_idx])
        # Gold = whichever row has is_correct == 1
        gold_labels.append(group.loc[group["is_correct"] == 1, "option_label"].values[0])

    acc = np.mean(np.array(pred_labels) == np.array(gold_labels))
    print(f"[{name}] MCQ accuracy: {acc:.4f}  ({n_questions:,} questions)")
    return acc


# ── Trainers ─────────────────────────────────────────────────────────────
def train_lr(X_train, y_train):
    print("Training Logistic Regression ...")
    clf = LogisticRegression(
        solver="liblinear",   # good for sparse high-dim binary
        C=1.0,
        max_iter=1000,
        n_jobs=None,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Main ─────────────────────────────────────────────────────────────────
def main(model_name: str):
    print(f"=== Training Model A | {model_name.upper()} ===")
    X_train, y_train, _ = load_features("train")
    X_val,   y_val,   val_df  = load_features("val")
    X_test,  y_test,  test_df = load_features("test")
    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    print(f"Positive class ratio (train): {y_train.mean():.3f}")

    if model_name == "lr":
        clf = train_lr(X_train, y_train)
        out_path = os.path.join(MODELS_DIR, "model_a_lr.joblib")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Binary metrics on val and test ───────────────────────────────────
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    evaluate_binary(y_val,  val_pred,  "VAL")
    evaluate_binary(y_test, test_pred, "TEST")

    # ── MCQ metrics ──────────────────────────────────────────────────────
    val_probs  = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    val_acc  = evaluate_mcq(val_probs,  val_df,  "VAL")
    test_acc = evaluate_mcq(test_probs, test_df, "TEST")

    # ── Persist ──────────────────────────────────────────────────────────
    joblib.dump(clf, out_path)
    print(f"\nSaved model to {out_path}")

    # Append a result row to a shared results log
    results_path = os.path.join(MODELS_DIR, "results_log.csv")
    row = {
        "model": model_name,
        "val_binary_acc":  accuracy_score(y_val,  val_pred),
        "val_binary_f1":   f1_score(y_val,  val_pred, average="macro"),
        "test_binary_acc": accuracy_score(y_test, test_pred),
        "test_binary_f1":  f1_score(y_test, test_pred, average="macro"),
        "val_mcq_acc":  val_acc,
        "test_mcq_acc": test_acc,
    }
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != model_name]
        df_log = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])
    df_log.to_csv(results_path, index=False)
    print(f"Updated results log: {results_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lr", choices=["lr"],
                   help="Which classical model to train (more added in later commits)")
    args = p.parse_args()
    main(args.model)
