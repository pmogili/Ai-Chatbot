"""
services/memory.py
--------------------
Conversation memory: persists turns to the database and provides
helpers to answer meta-questions like "what did I ask before?" or
"summarize our conversation."
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

import models

logger = logging.getLogger(__name__)

# Phrases that indicate the user is asking about conversation history
# rather than asking a new domain question.
_HISTORY_TRIGGERS = (
    "what did i ask",
    "what did i say",
    "what was my last question",
    "previous question",
    "summarize our conversation",
    "summarize the conversation",
    "summarise our conversation",
    "recap our conversation",
    "what have we talked about",
)


class MemoryService:
    """Reads/writes conversation turns and answers memory-related queries."""

    @staticmethod
    def ensure_session(db: Session, session_id: str) -> models.UserSession:
        """Get or create a UserSession row for the given session_id."""
        session_obj = db.get(models.UserSession, session_id)
        if session_obj is None:
            session_obj = models.UserSession(session_id=session_id)
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)
            logger.info("Created new session: %s", session_id)
        return session_obj

    @staticmethod
    def save_turn(
        db: Session,
        session_id: str,
        user_message: str,
        bot_response: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> models.Conversation:
        """Persist one (user_message, bot_response) turn to the database."""
        MemoryService.ensure_session(db, session_id)
        turn = models.Conversation(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            confidence=confidence,
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return turn

    @staticmethod
    def get_history(db: Session, session_id: str) -> List[models.Conversation]:
        """Return all conversation turns for a session, oldest first."""
        return (
            db.query(models.Conversation)
            .filter(models.Conversation.session_id == session_id)
            .order_by(models.Conversation.id.asc())
            .all()
        )

    @staticmethod
    def delete_history(db: Session, session_id: str) -> int:
        """Delete all conversation turns for a session. Returns rows deleted."""
        deleted = (
            db.query(models.Conversation)
            .filter(models.Conversation.session_id == session_id)
            .delete()
        )
        db.commit()
        return deleted

    @staticmethod
    def is_history_query(text: str) -> bool:
        """Check whether the user's message is asking about past conversation."""
        lowered = text.lower().strip()
        return any(trigger in lowered for trigger in _HISTORY_TRIGGERS)

    @staticmethod
    def answer_history_query(db: Session, session_id: str, text: str) -> str:
        """Produce a response for a memory/history-related question.

        Handles two cases:
            * "what did I ask before?" -> returns the previous user message
            * "summarize our conversation" -> returns a short summary
        """
        history = MemoryService.get_history(db, session_id)

        # Exclude the current in-flight turn if it was already saved
        # (it isn't at the point this is called, so history here is
        # everything strictly before the current message).
        if not history:
            return "We haven't talked about anything yet in this session!"

        lowered = text.lower()
        if "summarize" in lowered or "summarise" in lowered or "recap" in lowered:
            topics = [turn.intent or "general" for turn in history]
            unique_topics = list(dict.fromkeys(topics))
            count = len(history)
            return (
                f"So far we've exchanged {count} message{'s' if count != 1 else ''}, "
                f"covering topics like: {', '.join(unique_topics)}."
            )

        # Default: "what did I ask before?"
        last_turn = history[-1]
        return f"You previously asked: \"{last_turn.user_message}\""
