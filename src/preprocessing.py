"""
preprocessing.py — Text cleaning & feature engineering for RACE.

Implements the feature pipeline required by Model A (Answer Verifier) and
Model B (Distractor Ranker):

  1. Text cleaning           - lowercasing, punctuation/whitespace normalization
  2. One-Hot Encoding (OHE)  - primary feature representation (CountVectorizer binary=True)
  3. TF-IDF (optional)       - alternative representation, also useful for cosine sim
  4. Cosine similarity feats - between (article, question, option) triples
  5. Handcrafted features    - keyword overlap, char-match, lengths, etc.
  6. Verification builder    - explodes each row into 4 (article, question, option, label)
                              samples; label = 1 if option matches the correct one.

All vectorizers are fit on TRAIN ONLY and saved with joblib.
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Stopwords (small inline list — avoids extra dependency) ──────────────
STOPWORDS = set(
    """a an the is are was were be been being have has had do does did
       will would shall should can could may might must of in on at to
       for with by from as and or but if then so than that this these
       those i you he she it we they me him her us them my your his its
       our their what which who whom whose how when where why not no""".split()
)

# ── 1. TEXT CLEANING ─────────────────────────────────────────────────────
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokens(text: str, drop_stopwords: bool = True):
    """Return a list of clean tokens."""
    toks = clean_text(text).split()
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return toks


# ── 2. VECTORIZERS ───────────────────────────────────────────────────────
def build_one_hot_vectorizer(max_features: int = 15000) -> CountVectorizer:
    """Binary CountVectorizer = One-Hot Encoding over a fixed vocabulary."""
    return CountVectorizer(
        max_features=max_features,
        binary=True,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )


def build_tfidf_vectorizer(max_features: int = 15000) -> TfidfVectorizer:
    """TF-IDF vectorizer (optional but used for cosine similarity features)."""
    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )


# ── 3. HANDCRAFTED LEXICAL FEATURES ──────────────────────────────────────
def keyword_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap of content tokens."""
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_match_ratio(a: str, b: str) -> float:
    """Fraction of characters in `b` that also appear in `a`."""
    if not a or not b:
        return 0.0
    sa = set(a.lower())
    return sum(c in sa for c in b.lower()) / max(len(b), 1)


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    return lb / max(la, 1)


# ── 4. VERIFICATION FEATURE BUILDER ──────────────────────────────────────
def explode_to_verification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each row (article, question, A/B/C/D, answer) into 4 verification
    samples: (article, question, option_text, option_label, is_correct).
    """
    rows = []
    for _, r in df.iterrows():
        for opt in ("A", "B", "C", "D"):
            rows.append({
                "article": r["article"],
                "question": r["question"],
                "option_text": r[opt],
                "option_label": opt,
                "is_correct": int(r["answer"] == opt),
            })
    return pd.DataFrame(rows)


def build_features(
    ver_df: pd.DataFrame,
    ohe_vec: CountVectorizer,
    tfidf_vec: TfidfVectorizer,
    fit: bool = False,
):
    """
    Build the feature matrix for the verification task.

    Returns
    -------
    X : scipy.sparse.csr_matrix   shape (n_samples, n_features)
    y : np.ndarray (n_samples,)   binary labels
    """
    article = ver_df["article"].astype(str).map(clean_text).values
    question = ver_df["question"].astype(str).map(clean_text).values
    option = ver_df["option_text"].astype(str).map(clean_text).values

    # Combined text: weight article higher by repeating, append question + option
    combined = [f"{a} {a} {q} {o}" for a, q, o in zip(article, question, option)]

    if fit:
        X_ohe = ohe_vec.fit_transform(combined)
        # Fit TF-IDF on articles alone (used for cosine sim across A/Q/O)
        tfidf_vec.fit(np.concatenate([article, question, option]))
    else:
        X_ohe = ohe_vec.transform(combined)

    # ── Cosine similarity features ───────────────────────────────────────
    art_v = tfidf_vec.transform(article)
    q_v = tfidf_vec.transform(question)
    opt_v = tfidf_vec.transform(option)

    # Pairwise cosine similarities (per-row, diagonal only)
    cos_aq = _row_cosine(art_v, q_v)
    cos_ao = _row_cosine(art_v, opt_v)
    cos_qo = _row_cosine(q_v, opt_v)

    # ── Handcrafted features ─────────────────────────────────────────────
    n = len(ver_df)
    handcrafted = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        handcrafted[i, 0] = keyword_overlap(ver_df["article"].iat[i], ver_df["option_text"].iat[i])
        handcrafted[i, 1] = keyword_overlap(ver_df["question"].iat[i], ver_df["option_text"].iat[i])
        handcrafted[i, 2] = char_match_ratio(ver_df["article"].iat[i], ver_df["option_text"].iat[i])
        handcrafted[i, 3] = length_ratio(ver_df["question"].iat[i], ver_df["option_text"].iat[i])
        handcrafted[i, 4] = len(str(ver_df["option_text"].iat[i]).split())

    # Stack all features into a single sparse matrix
    extra = np.column_stack([cos_aq, cos_ao, cos_qo, handcrafted])  # (n, 8)
    X = sparse.hstack([X_ohe, sparse.csr_matrix(extra)]).tocsr()

    y = ver_df["is_correct"].values.astype(np.int8)
    return X, y


def _row_cosine(A: sparse.csr_matrix, B: sparse.csr_matrix) -> np.ndarray:
    """Per-row cosine similarity between two sparse matrices of equal shape."""
    a_norm = sparse.linalg.norm(A, axis=1)
    b_norm = sparse.linalg.norm(B, axis=1)
    dot = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    denom = np.maximum(a_norm * b_norm, 1e-9)
    return (dot / denom).astype(np.float32)


# ── 5. PIPELINE ENTRY POINT ──────────────────────────────────────────────
def run_pipeline(sample_train: int | None = None):
    """
    Build verification feature matrices for train / val / test and persist
    them along with the fitted vectorizers.

    Parameters
    ----------
    sample_train : int or None
        If given, sub-sample the train set to this many *original* rows
        (each row -> 4 verification samples).  Useful for quick smoke tests.
    """
    print("Loading processed splits ...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    if sample_train:
        train_df = train_df.sample(sample_train, random_state=42).reset_index(drop=True)
        print(f"  (sub-sampled train to {sample_train} rows)")

    print("Exploding rows -> verification samples ...")
    train_ver = explode_to_verification(train_df)
    val_ver = explode_to_verification(val_df)
    test_ver = explode_to_verification(test_df)
    print(f"  train: {len(train_ver):,}  |  val: {len(val_ver):,}  |  test: {len(test_ver):,}")
    print(f"  positive class ratio (train): {train_ver['is_correct'].mean():.3f}")

    print("Fitting vectorizers on TRAIN and building features ...")
    ohe_vec = build_one_hot_vectorizer()
    tfidf_vec = build_tfidf_vectorizer()
    X_train, y_train = build_features(train_ver, ohe_vec, tfidf_vec, fit=True)
    X_val,   y_val   = build_features(val_ver,   ohe_vec, tfidf_vec, fit=False)
    X_test,  y_test  = build_features(test_ver,  ohe_vec, tfidf_vec, fit=False)
    print(f"  X_train shape: {X_train.shape}  ({X_train.nnz:,} non-zeros)")

    # Persist vectorizers
    joblib.dump(ohe_vec,   os.path.join(MODELS_DIR, "ohe_vectorizer.joblib"))
    joblib.dump(tfidf_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    # Persist feature matrices
    sparse.save_npz(os.path.join(PROCESSED_DIR, "X_train.npz"), X_train)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "X_val.npz"),   X_val)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "X_test.npz"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    # Also save the verification dataframes (handy for downstream MCQ scoring)
    train_ver.to_csv(os.path.join(PROCESSED_DIR, "train_verification.csv"), index=False)
    val_ver.to_csv(os.path.join(PROCESSED_DIR, "val_verification.csv"), index=False)
    test_ver.to_csv(os.path.join(PROCESSED_DIR, "test_verification.csv"), index=False)

    print(f"\nArtifacts saved to:\n  {PROCESSED_DIR}\n  {MODELS_DIR}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Sub-sample train set to N rows (for quick testing)")
    args = p.parse_args()
    run_pipeline(sample_train=args.sample)
