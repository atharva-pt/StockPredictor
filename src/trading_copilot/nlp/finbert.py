"""FinBERT sentiment analyzer — primary engine.

Uses ProsusAI/finbert from Hugging Face. Runs on CPU (MPS if available).
Model is downloaded on first use and cached by transformers.
"""

from __future__ import annotations

from trading_copilot.logging_setup import get_logger
from trading_copilot.nlp.models import SentimentResult

log = get_logger("nlp.finbert")

_pipeline = None

_LABEL_MAP = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}

_SCORE_MAP = {
    "bullish": 1.0,
    "bearish": -1.0,
    "neutral": 0.0,
}


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        log.info("loading_finbert")
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        log.info("finbert_loaded")
    return _pipeline


def analyze(text: str) -> SentimentResult | None:
    """Analyze a single text with FinBERT. Returns None if model fails."""
    try:
        pipe = _get_pipeline()
        results = pipe(text[:512])[0]  # list of {label, score} dicts

        best = max(results, key=lambda r: r["score"])
        sentiment = _LABEL_MAP.get(best["label"].lower(), "neutral")

        return SentimentResult(
            text=text[:200],
            sentiment=sentiment,
            score=_SCORE_MAP[sentiment] * best["score"],
            confidence=best["score"],
            method="finbert",
        )
    except Exception as exc:
        log.warning("finbert_failed", error=str(exc))
        return None


def analyze_batch(texts: list[str]) -> list[SentimentResult | None]:
    """Batch analyze for efficiency. FinBERT handles batching internally."""
    try:
        pipe = _get_pipeline()
        truncated = [t[:512] for t in texts]
        all_results = pipe(truncated)

        output: list[SentimentResult | None] = []
        for text, results in zip(texts, all_results, strict=False):
            best = max(results, key=lambda r: r["score"])
            sentiment = _LABEL_MAP.get(best["label"].lower(), "neutral")
            output.append(
                SentimentResult(
                    text=text[:200],
                    sentiment=sentiment,
                    score=_SCORE_MAP[sentiment] * best["score"],
                    confidence=best["score"],
                    method="finbert",
                )
            )
        return output
    except Exception as exc:
        log.warning("finbert_batch_failed", error=str(exc))
        return [None] * len(texts)
