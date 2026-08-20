"""
services/knowledge_service.py
-------------------------------
Loads knowledge_base.json and provides simple keyword-based FAQ lookup.

This is intentionally lightweight (keyword overlap scoring) rather than
a full semantic search, keeping the project dependency-light while
still being effective for a small, curated FAQ set.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config import KNOWLEDGE_BASE_FILE
from utils.preprocessing import preprocess

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Provides lookup over a small JSON-based FAQ knowledge base."""

    def __init__(self) -> None:
        self.topics: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load and validate the knowledge base file.

        Raises:
            FileNotFoundError: if the knowledge base file is missing.
            ValueError: if the file content is malformed.
        """
        if not KNOWLEDGE_BASE_FILE.exists():
            logger.error("Knowledge base file missing at %s", KNOWLEDGE_BASE_FILE)
            raise FileNotFoundError(f"Knowledge base file not found: {KNOWLEDGE_BASE_FILE}")

        try:
            with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.topics = data.get("topics", [])
            if not self.topics:
                raise ValueError("Knowledge base contains no topics.")
            logger.info("Loaded %d knowledge base topics.", len(self.topics))
        except json.JSONDecodeError as exc:
            logger.error("Knowledge base file is not valid JSON: %s", exc)
            raise ValueError("Knowledge base file is not valid JSON.") from exc

    def search(self, query: str, min_matches: int = 1) -> Optional[str]:
        """Search the knowledge base for the best matching answer.

        Uses simple keyword-overlap scoring between the preprocessed
        query and each topic's keyword list plus its stored question.

        Args:
            query: Raw user text.
            min_matches: Minimum overlapping keywords required for a match.

        Returns:
            The best-matching answer string, or None if nothing matches
            well enough.
        """
        cleaned_query_tokens = set(preprocess(query).split())
        if not cleaned_query_tokens:
            return None

        best_topic: Optional[dict] = None
        best_score = 0

        for topic in self.topics:
            keyword_tokens = set()
            for kw in topic.get("keywords", []):
                keyword_tokens.update(preprocess(kw).split())
            # also consider tokens from the stored question itself
            keyword_tokens.update(preprocess(topic.get("question", "")).split())

            overlap = len(cleaned_query_tokens & keyword_tokens)
            if overlap > best_score:
                best_score = overlap
                best_topic = topic

        if best_topic is not None and best_score >= min_matches:
            return best_topic["answer"]
        return None


_kb_instance: Optional[KnowledgeBaseService] = None


def get_knowledge_base() -> KnowledgeBaseService:
    """Return a lazily-initialized, process-wide KnowledgeBaseService instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBaseService()
    return _kb_instance
