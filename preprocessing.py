"""
utils/preprocessing.py
-----------------------
Text preprocessing pipeline used by the intent classifier.

Steps performed:
    1. Lowercasing
    2. Tokenization
    3. Stopword removal
    4. Lemmatization

The module relies on NLTK for tokenization / stopwords / lemmatization.
Required NLTK corpora are downloaded lazily (once) on first import. If,
for some reason, the machine running this code has no internet access
and the corpora cannot be fetched, the module transparently falls back
to a lightweight regex tokenizer and a built-in stopword list so the
application keeps working (with slightly reduced NLP quality).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to set up NLTK. Fall back gracefully if data / network is unavailable.
# ---------------------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def _ensure_nltk_data() -> bool:
        """Download required NLTK corpora if missing. Returns success flag."""
        required = [
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        try:
            for path, pkg in required:
                try:
                    nltk.data.find(path)
                except LookupError:
                    nltk.download(pkg, quiet=True)
            return True
        except Exception as exc:  # pragma: no cover - network issues, etc.
            logger.warning("Could not fetch NLTK data (%s). Using fallback preprocessing.", exc)
            return False

    _NLTK_READY = _ensure_nltk_data()
except ImportError:  # pragma: no cover
    logger.warning("NLTK not installed. Using fallback preprocessing.")

# A small built-in stopword list used only if NLTK data is unavailable.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "to", "from", "in", "on", "do", "does", "did", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must",
    "not", "no", "so", "as", "than", "then", "there", "here", "what",
    "which", "who", "whom", "how", "why", "when", "where",
}


@lru_cache(maxsize=1)
def _get_stopwords() -> set:
    if _NLTK_READY:
        try:
            return set(nltk_stopwords.words("english"))
        except Exception:  # pragma: no cover
            return _FALLBACK_STOPWORDS
    return _FALLBACK_STOPWORDS


@lru_cache(maxsize=1)
def _get_lemmatizer():
    if _NLTK_READY:
        try:
            return WordNetLemmatizer()
        except Exception:  # pragma: no cover
            return None
    return None


def _fallback_tokenize(text: str) -> List[str]:
    """Simple regex-based word tokenizer used when NLTK is unavailable."""
    return re.findall(r"[a-zA-Z']+", text)


def tokenize(text: str) -> List[str]:
    """Tokenize raw text into a list of word tokens."""
    if _NLTK_READY:
        try:
            return word_tokenize(text)
        except Exception:  # pragma: no cover
            return _fallback_tokenize(text)
    return _fallback_tokenize(text)


def preprocess(text: str) -> str:
    """Run the full preprocessing pipeline on a piece of text.

    Pipeline: lowercase -> tokenize -> remove stopwords/punctuation ->
    lemmatize -> rejoin into a cleaned string.

    Args:
        text: Raw user input.

    Returns:
        A cleaned, space-joined string ready for vectorization.
    """
    if not text:
        return ""

    lowered = text.lower().strip()
    tokens = tokenize(lowered)

    stop_words = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    cleaned_tokens: List[str] = []
    for token in tokens:
        if not token.isalpha():
            continue
        if token in stop_words:
            continue
        if lemmatizer is not None:
            try:
                token = lemmatizer.lemmatize(token, pos="v")
                token = lemmatizer.lemmatize(token, pos="n")
            except Exception:  # pragma: no cover
                pass
        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)
