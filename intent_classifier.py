"""
services/intent_classifier.py
-------------------------------
TF-IDF + Logistic Regression intent classifier.

The classifier is trained on the patterns defined in intents.json and
persisted to disk with joblib. At runtime, this module loads the
persisted artifacts (training automatically on first run if none
exist) and exposes a simple `predict` API used by the response
generator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from config import (
    INTENTS_FILE,
    LABEL_ENCODER_FILE,
    MODEL_DIR,
    MODEL_FILE,
    UNKNOWN_INTENT,
    VECTORIZER_FILE,
)
from utils.preprocessing import preprocess

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Result of an intent classification call."""

    intent: str
    confidence: float


class IntentClassifier:
    """Wraps a TF-IDF vectorizer + Logistic Regression classifier.

    Handles both training (from intents.json) and inference (loading
    persisted model artifacts and predicting on new text).
    """

    def __init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._load_or_train()

    # ------------------------------------------------------------------
    # Loading / training
    # ------------------------------------------------------------------
    def _load_or_train(self) -> None:
        """Load persisted model artifacts, training fresh ones if absent."""
        if MODEL_FILE.exists() and VECTORIZER_FILE.exists() and LABEL_ENCODER_FILE.exists():
            try:
                self.model = joblib.load(MODEL_FILE)
                self.vectorizer = joblib.load(VECTORIZER_FILE)
                self.label_encoder = joblib.load(LABEL_ENCODER_FILE)
                logger.info("Loaded persisted intent classifier artifacts.")
                return
            except Exception as exc:  # pragma: no cover - corrupted artifacts, etc.
                logger.warning("Failed to load persisted model (%s). Retraining.", exc)

        logger.info("No valid persisted model found. Training a new one.")
        self.train()

    @staticmethod
    def _load_training_data() -> Tuple[List[str], List[str]]:
        """Read intents.json and flatten it into (text, label) pairs."""
        with open(INTENTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        texts: List[str] = []
        labels: List[str] = []
        for intent in data["intents"]:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                texts.append(pattern)
                labels.append(tag)
        return texts, labels

    def train(self) -> None:
        """Train the TF-IDF + Logistic Regression pipeline from intents.json."""
        texts, labels = self._load_training_data()

        # preprocess() is applied as the TF-IDF preprocessor so both
        # training and inference share the exact same cleaning pipeline.
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(texts)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        self.model = LogisticRegression(max_iter=1000, C=10.0)
        self.model.fit(X, y)

        self._persist()
        logger.info("Intent classifier trained on %d examples across %d intents.", len(texts), len(set(labels)))

    def _persist(self) -> None:
        """Save trained artifacts to disk so future runs skip retraining."""
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.vectorizer, VECTORIZER_FILE)
        joblib.dump(self.label_encoder, LABEL_ENCODER_FILE)
        logger.info("Persisted model artifacts to %s", MODEL_DIR)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, text: str) -> IntentPrediction:
        """Predict the intent of a piece of user text.

        Args:
            text: Raw user message.

        Returns:
            IntentPrediction with the predicted intent tag and the
            model's confidence (max class probability). If the model
            isn't ready or input is empty, returns the UNKNOWN_INTENT
            with zero confidence.
        """
        if not text or not text.strip() or self.model is None:
            return IntentPrediction(intent=UNKNOWN_INTENT, confidence=0.0)

        X = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(X)[0]
        best_idx = probabilities.argmax()
        confidence = float(probabilities[best_idx])
        intent = self.label_encoder.inverse_transform([best_idx])[0]

        return IntentPrediction(intent=intent, confidence=confidence)


# Module-level singleton so the (potentially expensive) model is
# loaded/trained only once per process.
_classifier_instance: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """Return a lazily-initialized, process-wide IntentClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance
