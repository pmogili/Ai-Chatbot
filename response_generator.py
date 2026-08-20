"""
services/response_generator.py
--------------------------------
Combines intent classification, knowledge base lookup, and conversation
memory into a single response-generation pipeline.

Priority order (as per spec):
    1. Greeting (and other high-confidence "small talk" intents)
    2. Intent match (general intents.json responses)
    3. Knowledge base
    4. Previous conversation (memory / history questions)
    5. Default fallback
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

from sqlalchemy.orm import Session

from config import CONFIDENCE_THRESHOLD, DEFAULT_FALLBACK_MESSAGE, INTENTS_FILE, UNKNOWN_INTENT
from services.intent_classifier import get_classifier
from services.knowledge_service import get_knowledge_base
from services.memory import MemoryService

logger = logging.getLogger(__name__)

# Intents whose canned responses in intents.json are conversational
# "small talk" and should be returned directly without a KB lookup.
_SMALL_TALK_INTENTS = {"greeting", "goodbye", "thanks", "help", "about", "weather"}

# Intents that should be answered from the knowledge base rather than
# a canned response, because they require factual lookup.
_KB_INTENTS = {"college_information", "ai_questions"}


@dataclass
class BotReply:
    """Structured result returned by the response generator."""

    response: str
    intent: str
    confidence: float


def _load_intent_responses() -> dict[str, list[str]]:
    with open(INTENTS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {intent["tag"]: intent.get("responses", []) for intent in data["intents"]}


_INTENT_RESPONSES = _load_intent_responses()


class ResponseGenerator:
    """Orchestrates classification, KB search, and memory to build a reply."""

    def __init__(self) -> None:
        self.classifier = get_classifier()
        self.knowledge_base = get_knowledge_base()

    def generate(self, db: Session, session_id: str, user_message: str) -> BotReply:
        """Generate a bot reply for a user message.

        Args:
            db: Active SQLAlchemy session (for memory lookups).
            session_id: The chat session identifier.
            user_message: Raw text typed by the user.

        Returns:
            A BotReply with the response text, detected intent, and
            classifier confidence.
        """
        text = (user_message or "").strip()
        if not text:
            return BotReply(response="Please type something so I can help you!", intent=UNKNOWN_INTENT, confidence=0.0)

        # --- Priority 4 shortcut: explicit memory/history questions -----
        # These are checked early because they are meta-questions about
        # the conversation itself, not something the intent classifier
        # or knowledge base is trained to answer.
        if MemoryService.is_history_query(text):
            answer = MemoryService.answer_history_query(db, session_id, text)
            return BotReply(response=answer, intent="memory", confidence=1.0)

        # --- Priority 1 & 2: intent classification -----------------------
        prediction = self.classifier.predict(text)

        if prediction.confidence >= CONFIDENCE_THRESHOLD and prediction.intent != UNKNOWN_INTENT:
            if prediction.intent in _SMALL_TALK_INTENTS:
                responses = _INTENT_RESPONSES.get(prediction.intent, [])
                if responses:
                    return BotReply(
                        response=random.choice(responses),
                        intent=prediction.intent,
                        confidence=prediction.confidence,
                    )

            if prediction.intent in _KB_INTENTS:
                kb_answer = self.knowledge_base.search(text)
                if kb_answer:
                    return BotReply(response=kb_answer, intent=prediction.intent, confidence=prediction.confidence)

        # --- Priority 3: knowledge base fallback search -------------------
        kb_answer = self.knowledge_base.search(text)
        if kb_answer:
            return BotReply(response=kb_answer, intent=prediction.intent, confidence=prediction.confidence)

        # --- Priority 5: default fallback ---------------------------------
        return BotReply(response=DEFAULT_FALLBACK_MESSAGE, intent=UNKNOWN_INTENT, confidence=prediction.confidence)


_generator_instance: ResponseGenerator | None = None


def get_response_generator() -> ResponseGenerator:
    """Return a lazily-initialized, process-wide ResponseGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResponseGenerator()
    return _generator_instance
