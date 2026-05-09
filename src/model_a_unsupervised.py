"""
model_a_unsupervised.py — Unsupervised & semi-supervised methods for Model A.

Implements the project requirement to "explore at least one unsupervised or
semi-supervised technique for answer verification and clustering":

  1. MiniBatch K-Means      - cluster verification samples; report purity & silhouette
  2. Label Propagation      - semi-supervised: small labeled set propagates labels
                              to a large unlabeled pool; evaluated on val.

Notes on scale:
  * Full train verification = ~246K samples * 15K features.
  * K-Means (MiniBatch) handles this directly.
  * Label Propagation builds an O(n^2) graph -> we subsample to ~3K labeled
    + ~7K unlabeled and reduce dimensionality with TruncatedSVD.

Outputs metrics and persists clusterer / semi-supervised model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, f1_score, accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_split(split: str):
    X = sparse.load_npz(os.path.join(PROCESSED_DIR, f"X_{split}.npz"))
    y = np.load(os.path.join(PROCESSED_DIR, f"y_{split}.npy"))
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}_verification.csv"))
    return X, y, df


# ── 1. K-MEANS CLUSTERING ───────────────────────────────────────────────
def cluster_purity(labels: np.ndarray, gold: np.ndarray) -> float:
    """Fraction of samples whose cluster majority class matches them."""
    purity = 0
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() == 0:
            continue
        purity += np.bincount(gold[mask]).max()
    return purity / len(gold)


def run_kmeans(k: int = 4, sil_sample: int = 5000):
    print(f"\n=== K-Means (k={k}) ===")
    X_train, y_train, _ = load_split("train")

    # L2-normalize so MiniBatchKMeans behaves like spherical k-means (cosine-like)
    X_norm = normalize(X_train, norm="l2", copy=False)

    km = MiniBatchKMeans(
        n_clusters=k, batch_size=4096, n_init=5,
        max_iter=200, random_state=42, verbose=0,
    )
    labels = km.fit_predict(X_norm)

    purity = cluster_purity(labels, y_train)

    # Silhouette is O(n^2) — sample for tractability
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), size=min(sil_sample, len(labels)), replace=False)
    sil = silhouette_score(X_norm[idx], labels[idx], metric="cosine")

    print(f"  Train purity vs is_correct : {purity:.4f}")
    print(f"  Silhouette (sample n={len(idx)}): {sil:.4f}")
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")

    joblib.dump(km, os.path.join(MODELS_DIR, f"model_a_kmeans_k{k}.joblib"))
    return {"k": k, "purity": purity, "silhouette": sil}


# ── 2. LABEL PROPAGATION (semi-supervised) ───────────────────────────────
def run_label_propagation(
    labeled_n: int = 3000,
    unlabeled_n: int = 7000,
    svd_dim: int = 100,
):
    """
    Semi-supervised setup:
      - Take `labeled_n` samples with their is_correct labels.
      - Take `unlabeled_n` samples and mark labels as -1.
      - Fit Label Propagation; evaluate predictions on the validation set.
    """
    print(f"\n=== Label Propagation (labeled={labeled_n}, unlabeled={unlabeled_n}) ===")
    X_train, y_train, _ = load_split("train")
    X_val,   y_val,   _ = load_split("val")

    rng = np.random.default_rng(42)
    n_train = X_train.shape[0]
    perm = rng.permutation(n_train)
    lab_idx = perm[:labeled_n]
    unl_idx = perm[labeled_n: labeled_n + unlabeled_n]

    # Combine labeled + unlabeled
    X_combined = sparse.vstack([X_train[lab_idx], X_train[unl_idx]])
    y_combined = np.concatenate([y_train[lab_idx], -np.ones(unlabeled_n, dtype=np.int8)])

    # LP requires dense; reduce dim via TruncatedSVD (LSA on TF-IDF/OHE)
    print(f"  Reducing dim to {svd_dim} via TruncatedSVD ...")
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_combined_dense = svd.fit_transform(X_combined)
    X_val_dense = svd.transform(X_val)

    print("  Fitting LabelPropagation ...")
    lp = LabelPropagation(kernel="knn", n_neighbors=10, max_iter=50)
    lp.fit(X_combined_dense, y_combined)

    val_pred = lp.predict(X_val_dense)
    acc = accuracy_score(y_val, val_pred)
    f1m = f1_score(y_val, val_pred, average="macro", zero_division=0)
    print(f"  VAL accuracy: {acc:.4f}")
    print(f"  VAL F1 macro: {f1m:.4f}")

    joblib.dump({"svd": svd, "lp": lp},
                os.path.join(MODELS_DIR, "model_a_label_prop.joblib"))
    return {
        "labeled_n": labeled_n, "unlabeled_n": unlabeled_n,
        "val_acc": acc, "val_f1_macro": f1m,
    }


def append_results(rows):
    path = os.path.join(MODELS_DIR, "results_log.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        # remove any earlier rows with the same `model` key
        existing = existing[~existing["model"].isin(new_df["model"])]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)
    print(f"\nUpdated results log -> {path}")


if __name__ == "__main__":
    rows = []

    # K-Means with a couple of K values
    for k in (4, 8):
        m = run_kmeans(k=k)
        rows.append({
            "model": f"kmeans_k{k}",
            "purity": m["purity"],
            "silhouette": m["silhouette"],
        })

    # Label Propagation
    lp_m = run_label_propagation()
    rows.append({
        "model": "label_propagation",
        "val_acc": lp_m["val_acc"],
        "val_f1_macro": lp_m["val_f1_macro"],
        "labeled_n": lp_m["labeled_n"],
        "unlabeled_n": lp_m["unlabeled_n"],
    })

    append_results(rows)
    print("Done.")
