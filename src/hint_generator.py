"""
hint_generator.py - Extractive hint generation with classical ML scoring.

Two strategies (per the project spec):

  1. Extractive cosine    - score each passage sentence by TF-IDF cosine
                            similarity to the (question + correct answer).
  2. ML-scored ranking    - Logistic Regression trained on handcrafted sentence
                            features:
                              * keyword overlap with question
                              * keyword overlap with answer
                              * position in passage (normalized)
                              * sentence length
                              * proper-noun density

Weak supervision target:
  positive  = sentence with the highest TF-IDF cosine sim to (question+answer)
  negative  = three random other sentences from the same article

Hint output is GRADUATED:
  - Hint 1: most general (lowest-scored relevant sentence)
  - Hint 2: medium specificity
  - Hint 3: most specific / near-explicit (highest-scored sentence)

Evaluation:
  * Precision@1, Precision@3 on whether the LR-ranked top sentence(s) match
    the cosine-pseudo-gold sentence (token-overlap >= 0.5).
"""

import os
import re
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_B_DIR = os.path.join(BASE_DIR, "models", "model_b", "traditional")
os.makedirs(MODELS_B_DIR, exist_ok=True)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str):
    text = str(text).replace("\\n", " ").replace("\n", " ")
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) > 10]


def _tokens(s):
    return set(t.lower() for t in WORD_RE.findall(str(s)))


# ── Handcrafted sentence features ───────────────────────────────────────
def sent_features(sent: str, question: str, answer: str,
                  position: float, total_sents: int):
    s_tok = _tokens(sent)
    q_tok = _tokens(question)
    a_tok = _tokens(answer)
    return [
        len(s_tok & q_tok) / max(len(q_tok), 1),         # q overlap
        len(s_tok & a_tok) / max(len(a_tok), 1),         # answer overlap
        position,                                         # normalized 0..1
        len(s_tok),                                       # length
        len(sent),                                        # char length
        sum(c.isupper() for c in sent) / max(len(sent), 1),  # proper-noun density
        sent.count(",") + sent.count(";"),
        int("?" in sent),
    ]


# ── Pseudo-gold: best sentence by TF-IDF cosine to (question+answer) ────
def _best_sentence_idx(sents, question, answer, vec):
    if not sents:
        return -1
    target = f"{question} {answer}"
    M = vec.fit_transform(sents + [target])
    sims = cosine_similarity(M[:-1], M[-1]).ravel()
    return int(np.argmax(sims))


# ── Build weak-supervision dataset ──────────────────────────────────────
def build_dataset(df: pd.DataFrame, n_articles: int = 1500, neg_per_pos: int = 3):
    rng = random.Random(42)
    rows = []
    sample = df.sample(min(n_articles, len(df)), random_state=42)
    vec = TfidfVectorizer(stop_words="english", lowercase=True,
                          token_pattern=r"(?u)\b[a-z]{2,}\b")

    for _, r in sample.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 4:
            continue
        ans_text = str(r[r["answer"]])
        try:
            best = _best_sentence_idx(sents, r["question"], ans_text, vec)
        except ValueError:
            continue
        if best < 0:
            continue
        n = len(sents)
        # Positive
        rows.append({
            "sentence": sents[best],
            "question": r["question"],
            "answer": ans_text,
            "position": best / max(n - 1, 1),
            "total_sents": n,
            "label": 1,
        })
        # Negatives
        cand = [i for i in range(n) if i != best]
        for ni in rng.sample(cand, min(neg_per_pos, len(cand))):
            rows.append({
                "sentence": sents[ni],
                "question": r["question"],
                "answer": ans_text,
                "position": ni / max(n - 1, 1),
                "total_sents": n,
                "label": 0,
            })
    return pd.DataFrame(rows)


def featurize(ds: pd.DataFrame):
    return np.array([
        sent_features(s, q, a, p, n)
        for s, q, a, p, n in zip(ds["sentence"], ds["question"],
                                  ds["answer"], ds["position"],
                                  ds["total_sents"])
    ])


# ── Train LR sentence ranker ────────────────────────────────────────────
def train_ranker():
    print("Loading splits ...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    print("Building weak-supervision dataset ...")
    tr = build_dataset(train_df, n_articles=2000)
    vl = build_dataset(val_df,   n_articles=400)
    print(f"  Train rows: {len(tr)}  |  Val rows: {len(vl)}  |  pos ratio: {tr['label'].mean():.3f}")

    Xtr, ytr = featurize(tr), tr["label"].values
    Xvl, yvl = featurize(vl), vl["label"].values

    print("Training Logistic Regression sentence ranker ...")
    lr = LogisticRegression(C=1.0, max_iter=500)
    lr.fit(Xtr, ytr)
    pred = lr.predict(Xvl)
    proba = lr.predict_proba(Xvl)[:, 1]
    print(f"  VAL accuracy : {accuracy_score(yvl, pred):.4f}")
    print(f"  VAL precision: {precision_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL recall   : {recall_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL F1       : {f1_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL ROC-AUC  : {roc_auc_score(yvl, proba):.4f}")
    print(f"  Coefs        : {dict(zip(['q_ov','a_ov','pos','len_w','len_c','propn','punct','q_mark'], np.round(lr.coef_[0], 3)))}")

    out = os.path.join(MODELS_B_DIR, "hint_ranker_lr.joblib")
    joblib.dump(lr, out)
    print(f"Saved -> {out}")
    return lr


# ── Inference: produce 3 graduated hints ────────────────────────────────
def generate_hints(article: str, question: str, answer: str, lr=None, top_k: int = 3):
    """
    Return list[str] of length up to top_k with hints ordered:
      hint[0] = most general, hint[1] = mid, hint[2] = most specific.
    """
    sents = split_sentences(article)
    if not sents:
        return []
    n = len(sents)

    if lr is None:
        lr_path = os.path.join(MODELS_B_DIR, "hint_ranker_lr.joblib")
        if os.path.exists(lr_path):
            lr = joblib.load(lr_path)

    feats = np.array([
        sent_features(s, question, answer, i / max(n - 1, 1), n)
        for i, s in enumerate(sents)
    ])

    if lr is not None:
        scores = lr.predict_proba(feats)[:, 1]
    else:
        # Fallback: pure cosine sim to (question+answer)
        vec = TfidfVectorizer(stop_words="english", lowercase=True,
                              token_pattern=r"(?u)\b[a-z]{2,}\b")
        M = vec.fit_transform(sents + [f"{question} {answer}"])
        scores = cosine_similarity(M[:-1], M[-1]).ravel()

    # Pick top-K relevant sentences, then order them general -> specific
    order = np.argsort(-scores)[:top_k]
    picked = [(sents[i], float(scores[i])) for i in order]

    # Graduated ordering: ascending score = most general first
    picked = sorted(picked, key=lambda x: x[1])
    return [s for s, _ in picked]


# ── End-to-end evaluation ───────────────────────────────────────────────
def evaluate_on_test(n_articles: int = 300):
    """Precision@1 and Precision@3 of the LR ranker against the cosine pseudo-gold sentence."""
    print(f"\n=== Hint ranker eval on {n_articles} TEST articles ===")
    lr = joblib.load(os.path.join(MODELS_B_DIR, "hint_ranker_lr.joblib"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv")).sample(
        n_articles, random_state=11
    )
    vec = TfidfVectorizer(stop_words="english", lowercase=True,
                          token_pattern=r"(?u)\b[a-z]{2,}\b")

    p1, p3, n_ok = 0, 0, 0
    for _, r in test_df.iterrows():
        sents = split_sentences(r["article"])
        if len(sents) < 4:
            continue
        ans = str(r[r["answer"]])
        try:
            gold = _best_sentence_idx(sents, r["question"], ans, vec)
        except ValueError:
            continue
        if gold < 0:
            continue
        gold_tokens = _tokens(sents[gold])
        n = len(sents)

        feats = np.array([
            sent_features(s, r["question"], ans, i / max(n - 1, 1), n)
            for i, s in enumerate(sents)
        ])
        scores = lr.predict_proba(feats)[:, 1]
        ranked = np.argsort(-scores)

        # Precision@k by token-overlap >= 0.5 with gold sentence
        def overlap(idx):
            t = _tokens(sents[idx])
            return len(t & gold_tokens) / max(len(gold_tokens), 1) >= 0.5

        if overlap(ranked[0]):
            p1 += 1
        if any(overlap(i) for i in ranked[:3]):
            p3 += 1
        n_ok += 1

    P1, P3 = p1 / max(n_ok, 1), p3 / max(n_ok, 1)
    print(f"  evaluated on {n_ok} articles")
    print(f"  Precision@1 : {P1:.4f}")
    print(f"  Precision@3 : {P3:.4f}")

    log_path = os.path.join(MODELS_B_DIR, "results_log.csv")
    extra = pd.DataFrame([{
        "model": "hint_ranker_lr",
        "test_precision_at_1": P1,
        "test_precision_at_3": P3,
        "n_evaluated": n_ok,
    }])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing["model"] != "hint_ranker_lr"]
        out_df = pd.concat([existing, extra], ignore_index=True)
    else:
        out_df = extra
    out_df.to_csv(log_path, index=False)


def demo(n: int = 3):
    print("\n=== Demo: graduated hints for TEST samples ===")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv")).sample(n, random_state=29)
    for i, (_, r) in enumerate(test_df.iterrows(), 1):
        ans = str(r[r["answer"]])
        hints = generate_hints(r["article"], r["question"], ans, top_k=3)
        print(f"\n--- Sample {i} ---")
        print(f"Question     : {r['question']}")
        print(f"Correct ans  : {ans}")
        print("Graduated hints (general -> specific):")
        for j, h in enumerate(hints, 1):
            print(f"  Hint {j}: {h[:200]}")


if __name__ == "__main__":
    train_ranker()
    evaluate_on_test(300)
    demo(3)
    print("\nDone.")
