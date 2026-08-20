"""
train_model.py
----------------
Standalone script to (re)train the intent classifier from intents.json
and persist the artifacts to trained_model/.

Usage:
    python train_model.py
"""

from __future__ import annotations

import logging

from config import configure_logging
from services.intent_classifier import IntentClassifier

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Train and persist the intent classifier, then run a quick smoke test."""
    logger.info("Starting intent classifier training...")
    classifier = IntentClassifier()
    classifier.train()  # force retrain even if a persisted model exists
    logger.info("Training complete. Running smoke test predictions:")

    samples = [
        "hello there",
        "what is machine learning",
        "thanks a lot",
        "tell me about the college fees",
        "bye for now",
        "asdkjaslkdjaskld",
    ]
    for sample in samples:
        prediction = classifier.predict(sample)
        print(f"  {sample!r:45s} -> intent={prediction.intent:20s} confidence={prediction.confidence:.3f}")


if __name__ == "__main__":
    main()
