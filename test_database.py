"""
tests/test_database.py
------------------------
Unit tests for the database models and MemoryService persistence layer.
"""

from services.memory import MemoryService


def test_ensure_session_creates_new_session(db_session):
    session_obj = MemoryService.ensure_session(db_session, "unit-test-session")
    assert session_obj.session_id == "unit-test-session"


def test_save_turn_persists_conversation(db_session):
    MemoryService.save_turn(
        db_session,
        session_id="unit-test-session-2",
        user_message="hello",
        bot_response="hi!",
        intent="greeting",
        confidence=0.9,
    )
    history = MemoryService.get_history(db_session, "unit-test-session-2")
    assert len(history) == 1
    assert history[0].user_message == "hello"
    assert history[0].bot_response == "hi!"


def test_get_history_returns_turns_in_order(db_session):
    session_id = "unit-test-session-3"
    MemoryService.save_turn(db_session, session_id, "first", "reply1")
    MemoryService.save_turn(db_session, session_id, "second", "reply2")

    history = MemoryService.get_history(db_session, session_id)
    assert [turn.user_message for turn in history] == ["first", "second"]


def test_delete_history_removes_all_turns(db_session):
    session_id = "unit-test-session-4"
    MemoryService.save_turn(db_session, session_id, "msg", "reply")
    deleted_count = MemoryService.delete_history(db_session, session_id)
    assert deleted_count == 1
    assert MemoryService.get_history(db_session, session_id) == []


def test_is_history_query_detects_memory_questions():
    assert MemoryService.is_history_query("what did I ask before?")
    assert MemoryService.is_history_query("Summarize our conversation please")
    assert not MemoryService.is_history_query("what is machine learning")
