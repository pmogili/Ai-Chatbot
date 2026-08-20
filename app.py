"""
app.py
-------
FastAPI application entry point. Exposes the chat UI and REST API.

Run locally with:
    uvicorn app:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from chatbot import get_chatbot
from config import APP_NAME, APP_VERSION, BASE_DIR, configure_logging
from database import get_db, init_db
from services.memory import MemoryService

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown hooks.

    On startup: initialize the database and warm up the chatbot
    (loads/trains the intent classifier once so the first real
    request isn't slow).
    """
    logger.info("Starting %s v%s ...", APP_NAME, APP_VERSION)
    init_db()
    get_chatbot()  # warm up: loads classifier + knowledge base
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down %s.", APP_NAME)


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Payload for POST /chat."""

    message: str = Field(..., min_length=1, description="The user's message text.")
    session_id: str | None = Field(default=None, description="Existing session ID, if any.")


class ChatResponse(BaseModel):
    """Response schema for POST /chat."""

    response: str
    intent: str
    confidence: float
    session_id: str


class ConversationTurn(BaseModel):
    """A single stored conversation turn."""

    id: int
    session_id: str
    user_message: str
    bot_response: str
    intent: str | None
    confidence: float | None
    timestamp: str | None


class HistoryResponse(BaseModel):
    """Response schema for GET /history/{session_id}."""

    session_id: str
    turns: list[ConversationTurn]


class DeleteHistoryResponse(BaseModel):
    """Response schema for DELETE /history/{session_id}."""

    session_id: str
    deleted_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request) -> HTMLResponse:
    """Serve the chatbot's web UI."""
    return templates.TemplateResponse(request, "index.html", {"app_name": APP_NAME})


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Accept a user message and return the chatbot's reply.

    Creates a new session_id if one wasn't supplied, so the frontend
    can start a conversation without any prior setup.
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = payload.session_id or str(uuid.uuid4())
    MemoryService.ensure_session(db, session_id)

    try:
        bot = get_chatbot()
        reply = bot.handle_message(db, session_id, message)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Error while generating chat response: %s", exc)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your message.")

    return ChatResponse(
        response=reply.response,
        intent=reply.intent,
        confidence=round(reply.confidence, 4),
        session_id=session_id,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["History"])
async def get_history(session_id: str, db: Session = Depends(get_db)) -> HistoryResponse:
    """Return the full conversation history for a session."""
    turns = MemoryService.get_history(db, session_id)
    return HistoryResponse(
        session_id=session_id,
        turns=[ConversationTurn(**turn.to_dict()) for turn in turns],
    )


@app.delete("/history/{session_id}", response_model=DeleteHistoryResponse, tags=["History"])
async def delete_history(session_id: str, db: Session = Depends(get_db)) -> DeleteHistoryResponse:
    """Delete all conversation history for a session."""
    deleted = MemoryService.delete_history(db, session_id)
    return DeleteHistoryResponse(session_id=session_id, deleted_count=deleted)


@app.get("/health", tags=["Meta"])
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}
