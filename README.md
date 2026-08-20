# Intermediate AI Chatbot

An intermediate-level AI chatbot built with **Python, FastAPI, and scikit-learn**.
It understands user intent with an NLP pipeline (NLTK preprocessing + TF-IDF +
Logistic Regression), answers factual questions from a small JSON knowledge
base, remembers conversation history in SQLite, and ships with a modern,
responsive chat UI (with dark mode).

---

## Features

- **Intent classification** — TF-IDF + Logistic Regression trained on `intents.json`, with lowercasing, tokenization, stopword removal, and lemmatization via NLTK.
- **Knowledge base search** — keyword-overlap lookup over `knowledge_base.json` (college info, AI/ML/DL, Python, admissions, fees, placements).
- **Conversation memory** — every turn is stored in SQLite; the bot can answer "what did I ask before?" and "summarize our conversation."
- **REST API** — `/chat`, `/history/{session_id}` (GET & DELETE), `/health`, with interactive Swagger docs at `/docs`.
- **Modern web UI** — chat bubbles, auto-scroll, typing indicator, timestamps, loading spinner, and a dark-mode toggle. No frontend framework — vanilla HTML/CSS/JS.
- **Robust error handling** — empty input, invalid requests, missing knowledge base, and database errors are all handled gracefully.
- **Logging** — structured logs (console + file) of requests, predictions, and errors.
- **Tests** — pytest suite covering the classifier, response generator, database layer, and API endpoints (24 tests).
- **Docker-ready** — `Dockerfile` + `docker-compose.yml` for one-command deployment.

---

## Project Structure

```
chatbot/
├── app.py                     # FastAPI app & routes
├── config.py                  # Central configuration
├── database.py                # SQLAlchemy engine/session setup
├── models.py                  # ORM models: Conversation, UserSession
├── chatbot.py                 # High-level ChatBot facade
├── train_model.py             # Standalone training script
├── intents.json                # Intent training data
├── knowledge_base.json         # FAQ knowledge base
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── static/
│   ├── style.css               # Chat UI styling (incl. dark mode)
│   └── script.js               # Chat UI logic (fetch, rendering, state)
│
├── templates/
│   └── index.html              # Chat UI markup
│
├── services/
│   ├── intent_classifier.py    # TF-IDF + Logistic Regression classifier
│   ├── response_generator.py   # Priority-based response pipeline
│   ├── memory.py                # Conversation history + memory Q&A
│   └── knowledge_service.py    # Knowledge base loader & search
│
├── utils/
│   └── preprocessing.py        # NLP preprocessing pipeline
│
└── tests/
    ├── conftest.py
    ├── test_intent_classifier.py
    ├── test_response_generator.py
    ├── test_database.py
    └── test_api.py
```

---

## Installation

### 1. Clone / unzip the project and enter the directory

```bash
cd chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first time the app runs, it will automatically download the small
NLTK corpora it needs (`punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`).
If you're on a machine without internet access at runtime, pre-download
them once with:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Train the intent classifier (optional — auto-trains on first run)

```bash
python train_model.py
```

This creates `trained_model/intent_model.joblib`, `vectorizer.joblib`,
and `label_encoder.joblib`. Re-run this script any time you edit
`intents.json`.

### 5. Run the app

```bash
uvicorn app:app --reload
```

Open your browser at **http://127.0.0.1:8000** for the chat UI, or
**http://127.0.0.1:8000/docs** for the interactive Swagger API docs.

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Chat web UI |
| POST | `/chat` | Send a message, get a bot reply |
| GET | `/history/{session_id}` | Retrieve conversation history |
| DELETE | `/history/{session_id}` | Delete conversation history |
| GET | `/health` | Health check |

### POST `/chat`

**Request body:**
```json
{
  "message": "what is machine learning",
  "session_id": "optional-existing-session-id"
}
```

**Response:**
```json
{
  "response": "Machine Learning is a subset of AI where systems learn patterns from data...",
  "intent": "ai_questions",
  "confidence": 0.8174,
  "session_id": "922c906d-2cab-46eb-ac8a-5d1e289e92cf"
}
```

If `session_id` is omitted, a new one is generated and returned — store
it client-side (the bundled UI does this in `localStorage`) and pass it
on subsequent requests to maintain conversation memory.

### GET `/history/{session_id}`

Returns every stored turn (`user_message`, `bot_response`, `intent`,
`confidence`, `timestamp`) for that session, oldest first.

### DELETE `/history/{session_id}`

Deletes all stored turns for that session and returns how many rows
were removed.

Full interactive documentation (with "Try it out") is auto-generated
by FastAPI at **`/docs`** (Swagger UI) and **`/redoc`**.

---

## Response Generation Priority

For every incoming message, the bot resolves a reply in this order:

1. **Memory/history questions** (e.g. "what did I ask before?") are detected first and answered from stored conversation turns.
2. **Greeting/small-talk intents** (greeting, goodbye, thanks, help, about, weather) return a canned response if classified with sufficient confidence.
3. **Knowledge-base-backed intents** (college info, AI questions) search `knowledge_base.json` for the best keyword match.
4. **Generic knowledge base search** — if the above steps didn't produce an answer, the raw message is searched against the knowledge base directly.
5. **Default fallback** — "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

---

## Architecture Diagram

```
┌─────────────┐      HTTP       ┌──────────────┐
│   Browser   │ ───────────────▶│   FastAPI    │
│ (HTML/CSS/  │◀─────────────── │   app.py     │
│   JS UI)    │      JSON       └──────┬───────┘
└─────────────┘                        │
                                        ▼
                              ┌───────────────────┐
                              │    chatbot.py      │
                              │  (ChatBot facade)  │
                              └─────────┬──────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
      ┌────────────────────┐ ┌──────────────────────┐ ┌──────────────────┐
      │ response_generator │ │   memory.py           │ │  database.py /   │
      │  .py (priority      │ │ (history + Q&A)       │ │  models.py       │
      │   pipeline)         │ └──────────┬───────────┘ │  (SQLite via     │
      └─────────┬───────────┘            │              │  SQLAlchemy)     │
                │                        └─────────────▶└──────────────────┘
     ┌──────────┴───────────┐
     ▼                      ▼
┌───────────────────┐ ┌─────────────────────┐
│ intent_classifier   │ │ knowledge_service    │
│ .py (TF-IDF +       │ │ .py (FAQ keyword     │
│ Logistic Regression)│ │  search)             │
└─────────┬───────────┘ └──────────────────────┘
          ▼
┌───────────────────────┐
│ utils/preprocessing.py │
│ (lowercase, tokenize,  │
│ stopwords, lemmatize)  │
└───────────────────────┘
```

---

## Testing

Run the full test suite with:

```bash
pytest tests/ -v
```

The suite covers:
- Intent classifier training and predictions (`test_intent_classifier.py`)
- Response generation priority logic (`test_response_generator.py`)
- Database models & memory persistence (`test_database.py`)
- FastAPI endpoints, including error cases (`test_api.py`)

---

## Deployment

### Local

```bash
uvicorn app:app --reload
```

### Docker

Build and run with Docker Compose (recommended — persists the database
and trained model in named volumes):

```bash
docker-compose up --build
```

The app will be available at **http://localhost:8000**.

Or with plain Docker:

```bash
docker build -t intermediate-ai-chatbot .
docker run -p 8000:8000 intermediate-ai-chatbot
```

---

## Extending the Chatbot

- **Add a new intent:** add a new object to `intents.json` with a `tag`, `patterns`, and `responses`, then run `python train_model.py` to retrain.
- **Add FAQ content:** add a new object to `knowledge_base.json` with `topic`, `keywords`, `question`, and `answer` — no retraining needed.
- **Swap the database:** set the `CHATBOT_DB_URL` environment variable to any SQLAlchemy-compatible connection string (e.g. PostgreSQL) — the ORM models work unchanged.
- **Improve NLU:** swap `services/intent_classifier.py`'s TF-IDF + Logistic Regression for a transformer-based classifier without touching `response_generator.py`, since it only depends on the `IntentPrediction` interface.

---

## Screenshots

> Placeholder — add your own screenshots here after running the app locally:
>
> `docs/screenshot-light-mode.png`
> `docs/screenshot-dark-mode.png`

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| NLP | NLTK (tokenization, stopwords, lemmatization), scikit-learn (TF-IDF, Logistic Regression) |
| Database | SQLite via SQLAlchemy ORM |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Testing | pytest |
| Deployment | Docker, docker-compose |

---

## License

This project is provided as a learning/reference implementation. Use and
modify it freely for educational or personal projects.
