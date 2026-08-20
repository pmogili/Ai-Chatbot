"""
tests/test_api.py
--------------------
Integration tests for the FastAPI REST endpoints.
"""


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_returns_valid_response(client):
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert "intent" in body
    assert "confidence" in body
    assert "session_id" in body


def test_chat_rejects_empty_message(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_chat_persists_and_returns_same_session_id(client):
    first = client.post("/chat", json={"message": "hi"}).json()
    session_id = first["session_id"]

    second = client.post("/chat", json={"message": "thanks", "session_id": session_id})
    assert second.json()["session_id"] == session_id


def test_history_endpoint_returns_conversation(client):
    first = client.post("/chat", json={"message": "hello"}).json()
    session_id = first["session_id"]
    client.post("/chat", json={"message": "thank you", "session_id": session_id})

    history_response = client.get(f"/history/{session_id}")
    assert history_response.status_code == 200
    turns = history_response.json()["turns"]
    assert len(turns) == 2


def test_delete_history_clears_conversation(client):
    first = client.post("/chat", json={"message": "hello"}).json()
    session_id = first["session_id"]

    delete_response = client.delete(f"/history/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_count"] == 1

    history_response = client.get(f"/history/{session_id}")
    assert history_response.json()["turns"] == []


def test_chat_missing_message_field_returns_422(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422
