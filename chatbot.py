"""
chatbot.py
-----------
Top-level ChatBot facade that ties together the response generator and
conversation memory behind a single, simple interface. This is the
class the FastAPI routes (and tests) interact with.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from services.memory import MemoryService
from services.response_generator import BotReply, get_response_generator

logger = logging.getLogger(__name__)


class ChatBot:
    """High-level chatbot interface: classify, respond, remember."""

    def __init__(self) -> None:
        self.response_generator = get_response_generator()

    def handle_message(self, db: Session, session_id: str, user_message: str) -> BotReply:
        """Process one user message end-to-end: generate + persist a reply.

        Args:
            db: Active SQLAlchemy session.
            session_id: Identifier for the current chat session.
            user_message: Raw text from the user.

        Returns:
            The BotReply that was generated (and has now been saved to
            the conversation history).
        """
        reply = self.response_generator.generate(db, session_id, user_message)

        MemoryService.save_turn(
            db=db,
            session_id=session_id,
            user_message=user_message,
            bot_response=reply.response,
            intent=reply.intent,
            confidence=reply.confidence,
        )

        logger.info(
            "session=%s intent=%s confidence=%.2f message=%r",
            session_id, reply.intent, reply.confidence, user_message,
        )
        return reply


_chatbot_instance: ChatBot | None = None


def get_chatbot() -> ChatBot:
    """Return a lazily-initialized, process-wide ChatBot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatBot()
    return _chatbot_instance
