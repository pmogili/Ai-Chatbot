"""
tests/test_response_generator.py
-----------------------------------
Unit tests for the response generation pipeline and its priority order.
"""

from services.memory import MemoryService
from services.response_generator import get_response_generator


def test_greeting_returns_small_talk_response(db_session):
    generator = get_response_generator()
    reply = generator.generate(db_session, "test-session-1", "hi there")
    assert reply.intent == "greeting"
    assert len(reply.response) > 0


def test_knowledge_base_question_is_answered(db_session):
    generator = get_response_generator()
    reply = generator.generate(db_session, "test-session-2", "what is python")
    assert "python" in reply.response.lower() or "programming" in reply.response.lower()


def test_unknown_input_triggers_fallback(db_session):
    generator = get_response_generator()
    reply = generator.generate(db_session, "test-session-3", "qzxjkw plmn asdgh")
    assert "rephrase" in reply.response.lower() or reply.intent == "unknown"


def test_empty_message_handled_gracefully(db_session):
    generator = get_response_generator()
    reply = generator.generate(db_session, "test-session-4", "")
    assert reply.intent == "unknown"
    assert "type something" in reply.response.lower()


def test_memory_query_recalls_previous_message(db_session):
    session_id = "test-session-memory"
    MemoryService.save_turn(db_session, session_id, "what is AI", "AI is...", intent="ai_questions", confidence=0.9)

    generator = get_response_generator()
    reply = generator.generate(db_session, session_id, "what did I ask before?")
    assert "what is ai" in reply.response.lower()
    assert reply.intent == "memory"
