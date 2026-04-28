"""
Clout or Cancel — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os
from scipy.sparse import hstack, csr_matrix
from typing import Optional
import time

# ── Load artifacts ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

tfidf          = load("tfidf_vectorizer.pkl")
lr             = load("logistic_regression.pkl")
EXTRA_FEATURES = load("extra_features.pkl")   # list of feature names

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clout or Cancel API",
    description="X.com tweet sentiment analysis — Positive / Negative",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_features(text: str) -> list:
    """
    Compute the same extra features used during training.
    Order must match EXTRA_FEATURES list:
    ['text_length', 'word_count', 'uppercase_ratio',
     'has_url', 'has_positive_emoji', 'has_negative_emoji']
    """
    url_pattern      = re.compile(r'https?://\S+')
    pos_emoji_pattern = re.compile(r'[:;]-?\)+|<3|:\*|;-?\)', re.IGNORECASE)
    neg_emoji_pattern = re.compile(r'[:;]-?\(+|>:|:\/|:\|', re.IGNORECASE)

    text_length     = len(text)
    word_count      = len(text.split())
    letters         = [c for c in text if c.isalpha()]
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    has_url         = int(bool(url_pattern.search(text)))
    has_positive_emoji = int(bool(pos_emoji_pattern.search(text)))
    has_negative_emoji = int(bool(neg_emoji_pattern.search(text)))

    return [text_length, word_count, uppercase_ratio,
            has_url, has_positive_emoji, has_negative_emoji]


def clean_text(text: str) -> str:
    """Light cleaning — mirrors preprocessing pipeline."""
    text = re.sub(r'https?://\S+', '', text)         # remove URLs
    text = re.sub(r'@\w+', '', text)                  # remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # keep only alpha
    text = re.sub(r'\s+', ' ', text).strip().lower()  # normalise whitespace
    return text


def classify(text: str) -> dict:
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    extra   = csr_matrix([extract_features(text)])    # raw text for feature extraction
    X       = hstack([vec, extra])

    pred    = int(lr.predict(X)[0])
    probs   = lr.predict_proba(X)[0]
    neg_p   = round(float(probs[0]), 4)
    pos_p   = round(float(probs[1]), 4)

    # Clout / Cancel risk score (0–100)
    cancel_risk = round(neg_p * 100, 1)
    clout_score = round(pos_p * 100, 1)

    label = "positive" if pred == 1 else "negative"
    verdict = "🔥 Clout" if pred == 1 else "❌ Cancel Risk"

    return {
        "label":        label,
        "verdict":      verdict,
        "confidence":   round(max(neg_p, pos_p) * 100, 1),
        "clout_score":  clout_score,
        "cancel_risk":  cancel_risk,
        "probabilities": {"negative": neg_p, "positive": pos_p},
    }

# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    tweet: str

    class Config:
        json_schema_extra = {
            "example": {"tweet": "This is the best update they've ever shipped!"}
        }


class BatchRequest(BaseModel):
    tweets: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Clout or Cancel API is live 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "logistic_regression", "features": EXTRA_FEATURES}


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    tweet = req.tweet.strip()
    if not tweet:
        raise HTTPException(status_code=400, detail="Tweet text cannot be empty.")
    if len(tweet) > 500:
        raise HTTPException(status_code=400, detail="Tweet too long (max 500 chars).")

    t0     = time.perf_counter()
    result = classify(tweet)
    ms     = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "tweet":      tweet,
        "latency_ms": ms,
        **result,
    }


@app.post("/batch")
def batch_analyse(req: BatchRequest):
    if not req.tweets:
        raise HTTPException(status_code=400, detail="Provide at least one tweet.")
    if len(req.tweets) > 100:
        raise HTTPException(status_code=400, detail="Max 100 tweets per batch.")

    results = []
    for tweet in req.tweets:
        tweet = tweet.strip()
        if not tweet:
            continue
        results.append({"tweet": tweet, **classify(tweet)})

    # Aggregate stats
    labels     = [r["label"] for r in results]
    pos_count  = labels.count("positive")
    neg_count  = labels.count("negative")
    avg_conf   = round(sum(r["confidence"] for r in results) / len(results), 1) if results else 0

    return {
        "count":     len(results),
        "positive":  pos_count,
        "negative":  neg_count,
        "avg_confidence": avg_conf,
        "results":   results,
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cloutorcancel.live",
        "https://www.cloutorcancel.live",
        "https://clout-or-cancel.vercel.app"   # ← no trailing slash
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
