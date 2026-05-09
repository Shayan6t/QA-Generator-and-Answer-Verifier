"""
inference.py - Unified prediction API for Model A and Model B.

Used by the Streamlit UI and by external test scripts.  Loads all trained
artifacts on first call and caches them for subsequent predictions.

Public API
----------
    verify_answer(article, question, option) -> float
        Return P(option is correct) using the soft-vote ensemble.

    score_mcq(article, question, options) -> dict
        Return per-option probabilities + the model's predicted label.

    generate_questions(article, answer, top_k=3) -> list[dict]
        Top-K Wh-template questions ranked by the QG ranker.

    generate_distractors(article, answer, top_n=3) -> list[str]
        Top-N distractor candidates ranked by the Model B RF ranker.

    generate_hints(article, question, answer, top_k=3) -> list[str]
        K graduated hints (general -> specific).

    full_pipeline(article, question=None, answer=None) -> dict
        End-to-end convenience wrapper used by the UI.
"""

import os
import joblib
import numpy as np
from scipy import sparse

from src.preprocessing import (
    clean_text, keyword_overlap, char_match_ratio, length_ratio,
)
from src.question_generator import (
    generate_questions as _qg_generate,
    _ranker_features as _qg_ranker_features,
)
from src.model_b_train import (
    generate_distractors as _generate_distractors,
)
from src.hint_generator import (
    generate_hints as _generate_hints,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_A_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
MODEL_B_DIR = os.path.join(BASE_DIR, "models", "model_b", "traditional")


# ── Lazy artifact loading ───────────────────────────────────────────────
class _Artifacts:
    def __init__(self):
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        # Vectorizers (built in preprocessing.py)
        self.ohe   = joblib.load(os.path.join(MODEL_A_DIR, "ohe_vectorizer.joblib"))
        self.tfidf = joblib.load(os.path.join(MODEL_A_DIR, "tfidf_vectorizer.joblib"))

        # Model A base classifiers
        self.lr  = joblib.load(os.path.join(MODEL_A_DIR, "model_a_lr.joblib"))
        self.svm = joblib.load(os.path.join(MODEL_A_DIR, "model_a_svm.joblib"))
        self.nb  = joblib.load(os.path.join(MODEL_A_DIR, "model_a_nb.joblib"))

        # Question generator ranker
        self.qg_ranker = joblib.load(os.path.join(MODEL_A_DIR, "qg_ranker.joblib"))

        # Model B
        self.b_ranker = joblib.load(os.path.join(MODEL_B_DIR, "model_b_ranker.joblib"))
        self.b_tfidf  = joblib.load(os.path.join(MODEL_B_DIR, "model_b_tfidf.joblib"))
        self.hint_ranker = joblib.load(os.path.join(MODEL_B_DIR, "hint_ranker_lr.joblib"))

        self._loaded = True


_ART = _Artifacts()


# ── Feature builder for a single (article, question, option) row ────────
def _row_cosine(a, b):
    """Per-row cosine similarity for two sparse matrices of equal shape."""
    a_norm = sparse.linalg.norm(a, axis=1)
    b_norm = sparse.linalg.norm(b, axis=1)
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return (dot / np.maximum(a_norm * b_norm, 1e-9)).astype(np.float32)


def _build_row_features(article: str, question: str, option: str):
    _ART.load()
    art_c = clean_text(article)
    q_c   = clean_text(question)
    opt_c = clean_text(option)
    combined = f"{art_c} {art_c} {q_c} {opt_c}"

    X_ohe = _ART.ohe.transform([combined])
    art_v = _ART.tfidf.transform([art_c])
    q_v   = _ART.tfidf.transform([q_c])
    opt_v = _ART.tfidf.transform([opt_c])

    cos_aq = _row_cosine(art_v, q_v)[0]
    cos_ao = _row_cosine(art_v, opt_v)[0]
    cos_qo = _row_cosine(q_v,  opt_v)[0]

    extra = np.array([[
        cos_aq, cos_ao, cos_qo,
        keyword_overlap(article, option),
        keyword_overlap(question, option),
        char_match_ratio(article, option),
        length_ratio(question, option),
        len(str(option).split()),
    ]], dtype=np.float32)

    X = sparse.hstack([X_ohe, sparse.csr_matrix(extra)]).tocsr()
    return X


def _clip_nonneg(X):
    X = X.copy()
    X.data = np.clip(X.data, 0.0, None)
    return X


# ── PUBLIC API ──────────────────────────────────────────────────────────
def verify_answer(article: str, question: str, option: str) -> float:
    """Soft-vote ensemble P(is_correct)."""
    _ART.load()
    X = _build_row_features(article, question, option)
    p = (_ART.lr.predict_proba(X)[:, 1]
         + _ART.svm.predict_proba(X)[:, 1]
         + _ART.nb.predict_proba(_clip_nonneg(X))[:, 1]) / 3
    return float(p[0])


def score_mcq(article: str, question: str, options: dict) -> dict:
    """
    Score 4 (or fewer) options.

    Parameters
    ----------
    options : dict  e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
    """
    scores = {label: verify_answer(article, question, txt)
              for label, txt in options.items()}
    pred = max(scores, key=scores.get)
    return {"scores": scores, "predicted": pred}


def generate_questions(article: str, answer: str, top_k: int = 3):
    """Top-K Wh-template questions ranked by the QG ranker."""
    _ART.load()
    cands = _qg_generate(article, answer, top_k=4)
    if not cands:
        return []
    feats = np.array([_qg_ranker_features(c["source"], c["question"], article)
                      for c in cands])
    scores = _ART.qg_ranker.predict_proba(feats)[:, 1]
    ranked = sorted(zip(cands, scores), key=lambda x: -x[1])[:top_k]
    return [{"question": c["question"], "score": float(s),
             "source_sentence": c["source"]} for c, s in ranked]


def generate_distractors(article: str, answer: str, top_n: int = 3):
    """Top-N distractor strings."""
    _ART.load()
    pairs = _generate_distractors(article, answer, _ART.b_ranker, _ART.b_tfidf, top_n=top_n)
    return [d for d, _ in pairs]


def generate_hints(article: str, question: str, answer: str, top_k: int = 3):
    """K graduated hints from general (1) to specific (k)."""
    _ART.load()
    return _generate_hints(article, question, answer, lr=_ART.hint_ranker, top_k=top_k)


def full_pipeline(article: str, question: str = None, answer: str = None,
                  options: dict = None):
    """
    Convenience wrapper for the UI.
      * If options provided -> verify with Model A.
      * Else if answer provided -> generate distractors -> 4-option MCQ.
      * Always returns hints if question + answer provided.
    """
    out = {}

    # Generate question if user only gave an answer
    if question is None and answer is not None:
        gen = generate_questions(article, answer, top_k=3)
        out["generated_questions"] = gen
        question = gen[0]["question"] if gen else None

    # Build full MCQ if no options provided
    if options is None and answer is not None:
        distr = generate_distractors(article, answer, top_n=3)
        labels = ["A", "B", "C", "D"]
        # Random-ish but deterministic placement of correct answer
        correct_pos = (sum(map(ord, str(answer))) % 4)
        opts = []
        d_iter = iter(distr)
        for i in range(4):
            opts.append(answer if i == correct_pos else next(d_iter, f"option {i}"))
        options = dict(zip(labels, opts))
        out["correct_label"] = labels[correct_pos]
        out["options"] = options

    # Verify
    if options is not None and question is not None:
        out["verification"] = score_mcq(article, question, options)

    # Hints
    if question is not None and answer is not None:
        out["hints"] = generate_hints(article, question, answer, top_k=3)

    return out


# ── Smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    article = (
        "Tom is a college student who loves cycling. Every morning he rides "
        "his bike for an hour around the lake near campus. He says cycling "
        "keeps him fit and clears his mind before classes. On weekends he "
        "joins a local club that organizes long-distance rides through the "
        "countryside. Last year, Tom completed a 200-kilometer charity ride "
        "and raised money for a children's hospital."
    )
    question = "Why does Tom cycle every morning?"
    options = {
        "A": "Because he needs to deliver newspapers.",
        "B": "Because cycling keeps him fit and clears his mind.",
        "C": "Because his university requires it.",
        "D": "Because he is training for the Olympics.",
    }

    print("\n--- score_mcq ---")
    print(score_mcq(article, question, options))

    print("\n--- generate_distractors (answer = correct option B) ---")
    print(generate_distractors(article, options["B"], top_n=3))

    print("\n--- generate_hints ---")
    for i, h in enumerate(generate_hints(article, question, options["B"], top_k=3), 1):
        print(f"  Hint {i}: {h[:200]}")

    print("\n--- generate_questions (given correct answer) ---")
    for q in generate_questions(article, options["B"], top_k=3):
        print(f"  [{q['score']:.3f}] {q['question']}")

    print("\nDone.")
