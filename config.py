"""
config.py
---------
Centralized configuration for the chatbot application.

All tunable constants (paths, thresholds, database URL, logging config)
live here so the rest of the codebase never hard-codes them.
"""

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

INTENTS_FILE: Path = BASE_DIR / "intents.json"
KNOWLEDGE_BASE_FILE: Path = BASE_DIR / "knowledge_base.json"
MODEL_DIR: Path = BASE_DIR / "trained_model"
MODEL_FILE: Path = MODEL_DIR / "intent_model.joblib"
VECTORIZER_FILE: Path = MODEL_DIR / "vectorizer.joblib"
LABEL_ENCODER_FILE: Path = MODEL_DIR / "label_encoder.joblib"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv("CHATBOT_DB_URL", f"sqlite:///{BASE_DIR / 'chatbot.db'}")

# ---------------------------------------------------------------------------
# NLP / Classification
# ---------------------------------------------------------------------------
# Below this confidence, the response generator falls back to
# knowledge-base search and finally the default fallback message.
CONFIDENCE_THRESHOLD: float = 0.35

# Intent that is returned when confidence is too low or the model
# has not been trained yet.
UNKNOWN_INTENT: str = "unknown"

DEFAULT_FALLBACK_MESSAGE: str = (
    "I'm sorry, I couldn't understand your question. Could you please rephrase it?"
)

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "Intermediate AI Chatbot"
APP_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "chatbot.log"
LOG_DIR.mkdir(exist_ok=True)


def configure_logging() -> None:
    """Configure root logging for the whole application.

    Logs to both a rotating file and stdout so the app is easy to
    debug locally as well as in a container (stdout is what Docker
    captures by default).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
