"""
models.py
----------
SQLAlchemy ORM models: Conversation and UserSession.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class UserSession(Base):
    """Represents a single chat session (one browser tab / user visit)."""

    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="session", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """A single turn of conversation: one user message + one bot response."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_sessions.session_id"))
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(nullable=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="conversations")

    def to_dict(self) -> dict:
        """Serialize this conversation turn to a plain dict for JSON responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "intent": self.intent,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
