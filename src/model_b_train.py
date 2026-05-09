"""
model_b_train.py - Distractor generation & ranking for Model B.

Pipeline (classical ML only, per the project spec):

  Step 1.  Candidate extraction:
            - tokenize the passage; collect content phrases (1-3 word n-grams)
              filtered by frequency and length.
  Step 2.  Feature engineering for each candidate:
            - OHE/TF-IDF cosine similarity to the correct answer
            - character-level overlap with the correct answer
            - passage frequency of the candidate
            - length-ratio to the correct answer
            - whether the candidate IS (a substring of) the correct answer  [excluded later]
  Step 3.  ML Ranker:
            - Random Forest, weak-supervision training:
                positives = the gold distractors B/C/D (when A is correct, etc.)
                negatives = random other passage phrases
            - At inference: pick top-3 non-answer candidates as distractors,
              with a diversity penalty so no two are too similar.

Evaluation:
  * Precision / Recall / F1 against the gold RACE distractors
    (token-overlap matching, since exact-string matching is too strict).
  * Top-1 distractor != correct answer accuracy.
  * Average pairwise cosine distance among the 3 outputs (diversity).
"""

import os
import re
import joblib
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_B_DIR = os.path.join(BASE_DIR, "models", "model_b", "traditional")
os.makedirs(MODELS_B_DIR, exist_ok=True)


STOPWORDS = set(
    """a an the is are was were be been being have has had do does did
       will would shall should can could may might must of in on at to
       for with by from as and or but if then so than that this these
       those i you he she it we they me him her us them my your his its
       our their what which who whom whose how when where why not no
       just very also there here too s t""".split()
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


# ── 1. CANDIDATE EXTRACTION ─────────────────────────────────────────────
def extract_candidates(article: str, top_k: int = 30, max_len_words: int = 4):
    """
    Extract candidate phrases from the passage:
      - Single content tokens occurring at least twice.
      - Bigrams that occur at least twice.
      - Short noun-like phrases (sequences of capitalized tokens).
    Returns up to `top_k` unique candidates ranked by frequency.
    """
    tokens = WORD_RE.findall(str(article))
    lc = [t.lower() for t in tokens]

    # Unigrams (frequency-filtered, drop stopwords)
    uni = Counter(t for t in lc if t not in STOPWORDS and len(t) > 2)
    cand = {w: c for w, c in uni.items() if c >= 2}

    # Bigrams
    for i in range(len(lc) - 1):
        a, b = lc[i], lc[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        if len(a) < 2 or len(b) < 2:
            continue
        bg = f"{a} {b}"
        cand[bg] = cand.get(bg, 0) + 1

    # Capitalized sequences (proper nouns)
    cap_phrase, run = [], []
    for t in tokens:
        if t[0].isupper() and t.lower() not in STOPWORDS:
            run.append(t)
        else:
            if 1 <= len(run) <= max_len_words:
                cap_phrase.append(" ".join(run))
            run = []
    if 1 <= len(run) <= max_len_words:
        cap_phrase.append(" ".join(run))
    for p in cap_phrase:
        cand[p.lower()] = cand.get(p.lower(), 0) + 1

    # Sort by frequency, drop very long candidates
    out = sorted(cand.items(), key=lambda kv: -kv[1])
    out = [(c, f) for c, f in out if 1 <= len(c.split()) <= max_len_words]
    return out[:top_k]


# ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────
def char_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.lower())
    return sum(c in sa for c in b.lower()) / max(len(b), 1)


def candidate_features(cand: str, freq: int, answer: str,
                       tfidf_vec: TfidfVectorizer):
    """Return a fixed-length feature vector for one candidate."""
    a_vec = tfidf_vec.transform([str(answer).lower()])
    c_vec = tfidf_vec.transform([cand])
    cos = float(cosine_similarity(a_vec, c_vec)[0, 0])
    return [
        cos,                                     # cosine sim to correct answer
        char_overlap(answer, cand),              # char overlap
        freq,                                    # passage frequency
        len(cand.split()),                       # length in words
        len(cand),                               # length in chars
        len(cand.split()) / max(len(str(answer).split()), 1),  # length ratio
        int(cand.lower() == str(answer).lower()),# is the answer itself
    ]


# ── 3. RANKER TRAINING (weak supervision) ───────────────────────────────
def _normalize_phrase(s):
    return re.sub(r"\s+", " ", str(s).lower().strip())


def build_ranker_dataset(df: pd.DataFrame, n_articles: int = 1500,
                         tfidf_vec: TfidfVectorizer = None):
    """
    Positive examples: gold distractors (the 3 wrong options for each row).
    Negative examples: random non-distractor passage phrases.
    """
    rng = random.Random(42)
    rows = []
    sample = df.sample(min(n_articles, len(df)), random_state=42)

    for _, r in sample.iterrows():
        article = str(r["article"])
        answer_label = r["answer"]
        answer_text = str(r[answer_label])
        gold_distractors = [str(r[k]) for k in "ABCD" if k != answer_label]
        gold_norm = {_normalize_phrase(d) for d in gold_distractors}
        ans_norm = _normalize_phrase(answer_text)

        cands = extract_candidates(article, top_k=40)
        if len(cands) < 8:
            continue

        # Positive samples: synthetic — use gold distractors directly with
        # a "pseudo-frequency" of 1 (they may not appear verbatim in passage).
        for d in gold_distractors:
            rows.append({
                "candidate": d, "freq": 1, "answer": answer_text,
                "article": article, "label": 1,
            })

        # Negative samples: passage phrases that are NOT (a substring of) any gold
        negatives = [(c, f) for c, f in cands
                     if c not in gold_norm and c != ans_norm]
        rng.shuffle(negatives)
        for c, f in negatives[:6]:
            rows.append({
                "candidate": c, "freq": f, "answer": answer_text,
                "article": article, "label": 0,
            })

    return pd.DataFrame(rows)


def featurize_dataset(ds: pd.DataFrame, tfidf_vec: TfidfVectorizer):
    X = np.array([
        candidate_features(c, f, a, tfidf_vec)
        for c, f, a in zip(ds["candidate"], ds["freq"], ds["answer"])
    ])
    y = ds["label"].values
    return X, y


def train_ranker():
    print("Loading splits ...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    print("Fitting TF-IDF over articles + options (Model B) ...")
    corpus = (train_df["article"].astype(str).tolist()
              + train_df["A"].astype(str).tolist()
              + train_df["B"].astype(str).tolist()
              + train_df["C"].astype(str).tolist()
              + train_df["D"].astype(str).tolist())
    tfidf = TfidfVectorizer(stop_words="english", lowercase=True,
                            sublinear_tf=True, max_features=15000,
                            token_pattern=r"(?u)\b[a-z]{2,}\b")
    tfidf.fit(corpus)

    print("Building weak-supervision ranker dataset ...")
    tr = build_ranker_dataset(train_df, n_articles=2000, tfidf_vec=tfidf)
    vl = build_ranker_dataset(val_df,   n_articles=400,  tfidf_vec=tfidf)
    print(f"  Train rows: {len(tr)}  |  Val rows: {len(vl)}  |  pos ratio: {tr['label'].mean():.3f}")

    print("Featurizing ...")
    Xtr, ytr = featurize_dataset(tr, tfidf)
    Xvl, yvl = featurize_dataset(vl, tfidf)

    print("Training RandomForest distractor ranker ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xvl)
    proba = rf.predict_proba(Xvl)[:, 1]
    print(f"  VAL accuracy : {accuracy_score(yvl, pred):.4f}")
    print(f"  VAL precision: {precision_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL recall   : {recall_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL F1       : {f1_score(yvl, pred, zero_division=0):.4f}")
    print(f"  VAL ROC-AUC  : {roc_auc_score(yvl, proba):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(yvl, pred)}")

    joblib.dump(rf,    os.path.join(MODELS_B_DIR, "model_b_ranker.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_B_DIR, "model_b_tfidf.joblib"))
    print(f"Saved ranker + tfidf -> {MODELS_B_DIR}")

    # results log
    log_path = os.path.join(MODELS_B_DIR, "results_log.csv")
    row = {
        "model": "model_b_ranker_rf",
        "val_acc": accuracy_score(yvl, pred),
        "val_prec": precision_score(yvl, pred, zero_division=0),
        "val_recall": recall_score(yvl, pred, zero_division=0),
        "val_f1": f1_score(yvl, pred, zero_division=0),
        "val_auc": roc_auc_score(yvl, proba),
    }
    pd.DataFrame([row]).to_csv(log_path, index=False)
    return rf, tfidf


# ── INFERENCE: generate top-3 distractors for a (article, answer) ───────
def _diversity_filter(scored, top_n=3, sim_threshold=0.7,
                      tfidf_vec: TfidfVectorizer = None):
    """Greedy: pick highest-scored, then skip candidates too similar to any picked."""
    chosen = []
    for cand, score in scored:
        if not chosen:
            chosen.append((cand, score))
        else:
            v = tfidf_vec.transform([cand])
            picked = tfidf_vec.transform([c for c, _ in chosen])
            sim = cosine_similarity(v, picked).max()
            if sim < sim_threshold:
                chosen.append((cand, score))
        if len(chosen) >= top_n:
            break
    return chosen


def generate_distractors(article: str, answer: str, rf, tfidf_vec, top_n: int = 3):
    cands = extract_candidates(article, top_k=40)
    cands = [(c, f) for c, f in cands
             if _normalize_phrase(c) != _normalize_phrase(answer)]
    if not cands:
        return []
    feats = np.array([candidate_features(c, f, answer, tfidf_vec) for c, f in cands])
    scores = rf.predict_proba(feats)[:, 1]
    scored = sorted(zip([c for c, _ in cands], scores), key=lambda x: -x[1])
    return _diversity_filter(scored, top_n=top_n, tfidf_vec=tfidf_vec)


def evaluate_on_test(n_articles: int = 200):
    """End-to-end MCQ-style evaluation against the gold RACE distractors."""
    print(f"\n=== End-to-end distractor evaluation on {n_articles} TEST articles ===")
    rf    = joblib.load(os.path.join(MODELS_B_DIR, "model_b_ranker.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_B_DIR, "model_b_tfidf.joblib"))

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    sample = test_df.sample(min(n_articles, len(test_df)), random_state=11)

    p_list, r_list, top1_correct_count, diversities = [], [], 0, []
    for _, row in sample.iterrows():
        gold_ans = str(row[row["answer"]])
        gold_distractors = [str(row[k]) for k in "ABCD" if k != row["answer"]]
        gold_tokens = [set(WORD_RE.findall(d.lower())) for d in gold_distractors]

        gen = generate_distractors(row["article"], gold_ans, rf, tfidf, top_n=3)
        if not gen:
            continue

        # Token-overlap matching (>=1 shared content token = match)
        gen_sets = [set(WORD_RE.findall(c.lower())) for c, _ in gen]
        tp = sum(any(len(g & d) >= 1 for d in gold_tokens) for g in gen_sets)
        prec = tp / max(len(gen_sets), 1)
        rec  = tp / 3
        p_list.append(prec)
        r_list.append(rec)

        # Top-1 must NOT equal the gold answer
        top1_correct_count += int(_normalize_phrase(gen[0][0]) != _normalize_phrase(gold_ans))

        # Diversity: 1 - mean pairwise cosine sim
        if len(gen) >= 2:
            vecs = tfidf.transform([c for c, _ in gen])
            sims = cosine_similarity(vecs)
            n = sims.shape[0]
            off = sims.sum() - sims.trace()
            diversities.append(1 - off / max(n * (n - 1), 1))

    P = float(np.mean(p_list))
    R = float(np.mean(r_list))
    F1 = 2 * P * R / max(P + R, 1e-9)
    top1_acc = top1_correct_count / max(len(p_list), 1)
    div = float(np.mean(diversities)) if diversities else 0.0

    print(f"  Precision (token-overlap) : {P:.4f}")
    print(f"  Recall    (token-overlap) : {R:.4f}")
    print(f"  F1                        : {F1:.4f}")
    print(f"  Top-1 != correct answer   : {top1_acc:.4f}")
    print(f"  Diversity (1 - avg cos)   : {div:.4f}")

    log_path = os.path.join(MODELS_B_DIR, "results_log.csv")
    extra = pd.DataFrame([{
        "model": "model_b_end_to_end",
        "test_precision_overlap": P, "test_recall_overlap": R, "test_f1_overlap": F1,
        "top1_not_answer_acc": top1_acc, "diversity": div,
    }])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing["model"] != "model_b_end_to_end"]
        out_df = pd.concat([existing, extra], ignore_index=True)
    else:
        out_df = extra
    out_df.to_csv(log_path, index=False)


def demo(n: int = 3):
    print("\n=== Demo: generated distractors for TEST samples ===")
    rf    = joblib.load(os.path.join(MODELS_B_DIR, "model_b_ranker.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_B_DIR, "model_b_tfidf.joblib"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv")).sample(n, random_state=23)
    for i, (_, r) in enumerate(test_df.iterrows(), 1):
        gold = str(r[r["answer"]])
        gen = generate_distractors(r["article"], gold, rf, tfidf, top_n=3)
        print(f"\n--- Sample {i} ---")
        print(f"Question     : {r['question']}")
        print(f"Correct ans  : {gold}")
        print("Generated distractors:")
        for c, s in gen:
            print(f"  [{s:.3f}] {c}")
        print("Gold distractors:")
        for k in "ABCD":
            if k != r["answer"]:
                print(f"  - {r[k]}")


if __name__ == "__main__":
    train_ranker()
    evaluate_on_test(n_articles=300)
    demo(3)
    print("\nDone.")
