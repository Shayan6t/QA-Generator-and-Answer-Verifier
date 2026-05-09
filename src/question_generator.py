"""
question_generator.py - Template-based question generation for Model A.

Pipeline (per the project spec, classical ML only):

  Step 1.  Extract candidate sentences from the passage by OHE/TF-IDF keyword
           overlap with the correct answer.
  Step 2.  Apply Wh-word templates (Who/What/Where/When/Why/How/Which) to
           rewrite a candidate sentence into a question.
  Step 3.  Rank generated questions with a trained ML ranker (Random Forest)
           that scores fluency + relevance features.

The ranker is trained with weak supervision:
  * Positive examples = the sentence that has the highest TF-IDF cosine
    similarity with the original RACE question  (i.e. a likely source).
  * Negative examples = three other random sentences from the same article.

Outputs
  - models/model_a/traditional/qg_ranker.joblib
  - prints a few generated samples on TEST articles
"""

import os
import re
import joblib
import random
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Sentence splitter (simple, no NLTK required) ────────────────────────
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
def split_sentences(text: str):
    text = str(text).replace("\\n", " ").replace("\n", " ")
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) > 10]


# ── Step 1: candidate sentence extraction ───────────────────────────────
def candidate_sentences(article: str, answer: str, top_k: int = 5):
    """Return top-k sentences ranked by TF-IDF cosine similarity to the answer."""
    sents = split_sentences(article)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english", lowercase=True,
                          token_pattern=r"(?u)\b[a-z]{2,}\b")
    M = vec.fit_transform(sents + [str(answer)])
    sims = cosine_similarity(M[:-1], M[-1]).ravel()
    order = np.argsort(-sims)[:top_k]
    return [(sents[i], float(sims[i])) for i in order]


# ── Step 2: Wh-word templates ───────────────────────────────────────────
WH_TEMPLATES = {
    "what":  "What does the passage say about {focus}?",
    "who":   "Who is associated with {focus}?",
    "where": "Where is {focus} located or mentioned?",
    "when":  "When does {focus} occur in the passage?",
    "why":   "Why is {focus} important according to the passage?",
    "how":   "How is {focus} described in the passage?",
    "which": "Which of the following best describes {focus}?",
}


def _pick_focus(sentence: str, answer: str) -> str:
    """Pick the focus phrase: prefer the answer if present, else the longest
    content noun-like token in the sentence."""
    if answer and answer.lower() in sentence.lower():
        return answer
    toks = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", sentence)
    toks = [t for t in toks if t.lower() not in {"the", "and", "that", "with",
                                                  "this", "from", "have", "been"}]
    if not toks:
        return answer or "the topic"
    # Heuristic: longest capitalized or longest token
    cap = [t for t in toks if t[0].isupper()]
    return max(cap, key=len) if cap else max(toks, key=len)


def generate_questions(article: str, answer: str, top_k: int = 5):
    """Return list of dicts: {question, source_sentence, focus, sim}."""
    out = []
    for sent, sim in candidate_sentences(article, answer, top_k=top_k):
        focus = _pick_focus(sent, answer)
        for wh, tpl in WH_TEMPLATES.items():
            q = tpl.format(focus=focus)
            out.append({"question": q, "wh": wh, "source": sent,
                        "focus": focus, "sim_to_answer": sim})
    return out


# ── Step 3: ML ranker training data via weak supervision ────────────────
def _ranker_features(sentence: str, question: str, article: str):
    """Compute simple fluency + relevance features for the ranker."""
    s_tokens = re.findall(r"[a-z]+", sentence.lower())
    q_tokens = re.findall(r"[a-z]+", question.lower())
    a_tokens = re.findall(r"[a-z]+", article.lower())
    set_s, set_q, set_a = set(s_tokens), set(q_tokens), set(a_tokens)

    return [
        len(s_tokens),
        len(q_tokens),
        len(set_s & set_q) / max(len(set_q), 1),         # q-coverage by sentence
        len(set_s & set_a) / max(len(set_s), 1),         # how representative the sentence is
        sum(c.isupper() for c in sentence) / max(len(sentence), 1),  # proper-noun density
        sentence.count(",") + sentence.count(";"),       # punctuation richness
    ]


def build_ranker_dataset(df: pd.DataFrame, n_articles: int = 1500, neg_per_pos: int = 3):
    """Weak-supervision dataset: positive = best-matching sentence, negatives = random."""
    rng = random.Random(42)
    rows = []
    sample = df.sample(min(n_articles, len(df)), random_state=42)

    vec = TfidfVectorizer(stop_words="english", lowercase=True,
                          token_pattern=r"(?u)\b[a-z]{2,}\b")

    for _, r in sample.iterrows():
        art, q = str(r["article"]), str(r["question"])
        sents = split_sentences(art)
        if len(sents) < 4:
            continue
        try:
            M = vec.fit_transform(sents + [q])
        except ValueError:
            continue
        sims = cosine_similarity(M[:-1], M[-1]).ravel()
        pos_idx = int(np.argmax(sims))
        rows.append({"sentence": sents[pos_idx], "question": q, "article": art, "label": 1})
        # Negatives: random other sentences
        cand = [i for i in range(len(sents)) if i != pos_idx]
        for ni in rng.sample(cand, min(neg_per_pos, len(cand))):
            rows.append({"sentence": sents[ni], "question": q, "article": art, "label": 0})
    return pd.DataFrame(rows)


def train_ranker():
    print("Building weak-supervision ranker dataset ...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    tr = build_ranker_dataset(train_df, n_articles=1500)
    vl = build_ranker_dataset(val_df,   n_articles=300)
    print(f"  Train rows: {len(tr)}  |  Val rows: {len(vl)}  |  pos ratio: {tr['label'].mean():.3f}")

    Xtr = np.array([_ranker_features(s, q, a) for s, q, a in
                    zip(tr["sentence"], tr["question"], tr["article"])])
    Xvl = np.array([_ranker_features(s, q, a) for s, q, a in
                    zip(vl["sentence"], vl["question"], vl["article"])])
    ytr, yvl = tr["label"].values, vl["label"].values

    print("Training Random Forest ranker ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xvl)
    proba = rf.predict_proba(Xvl)[:, 1]
    print(f"  VAL accuracy : {accuracy_score(yvl, pred):.4f}")
    print(f"  VAL ROC-AUC  : {roc_auc_score(yvl, proba):.4f}")

    out = os.path.join(MODELS_DIR, "qg_ranker.joblib")
    joblib.dump(rf, out)
    print(f"Saved ranker -> {out}")

    # Persist a tiny results row
    log_path = os.path.join(MODELS_DIR, "results_log.csv")
    row = {"model": "qg_ranker_rf",
           "val_acc": accuracy_score(yvl, pred),
           "val_auc": roc_auc_score(yvl, proba)}
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing["model"] != row["model"]]
        out_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out_df = pd.DataFrame([row])
    out_df.to_csv(log_path, index=False)
    return rf


# ── End-to-end demo ─────────────────────────────────────────────────────
def demo(n: int = 3, top_k_questions: int = 3):
    print("\n=== Demo: generate + rank questions on TEST articles ===")
    rf = joblib.load(os.path.join(MODELS_DIR, "qg_ranker.joblib"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv")).sample(n, random_state=7)
    for i, (_, r) in enumerate(test_df.iterrows(), 1):
        article, gold_q = str(r["article"]), str(r["question"])
        gold_answer = str(r[r["answer"]])
        candidates = generate_questions(article, gold_answer, top_k=4)
        feats = np.array([_ranker_features(c["source"], c["question"], article)
                          for c in candidates])
        scores = rf.predict_proba(feats)[:, 1]
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

        print(f"\n--- Article {i} ---")
        print(f"Gold question: {gold_q}")
        print(f"Gold answer  : {gold_answer}")
        print("Top generated questions:")
        for c, s in ranked[:top_k_questions]:
            print(f"  [{s:.3f}] {c['question']}")


if __name__ == "__main__":
    train_ranker()
    demo()
    print("\nDone.")
