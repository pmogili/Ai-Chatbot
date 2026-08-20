# 🤖 AI Chatbot

An intermediate-level AI chatbot built using **Python, Natural Language Processing (NLP), machine learning, FastAPI, and SQLite**. The chatbot uses intent classification, a curated knowledge base, and conversation memory to understand user queries and generate relevant responses.

The project includes a web-based chat interface, REST API, persistent conversation history, trained machine learning models, automated tests, and Docker support.

## 🚀 Project Overview

Traditional rule-based chatbots often depend on manually defined responses for every possible question. This project combines **machine learning-based intent classification** with a knowledge base and conversation memory to provide a more flexible chatbot experience.

The chatbot can:

* 💬 Understand user messages
* 🧠 Classify user intent using machine learning
* 📚 Search a knowledge base for factual questions
* 🧠 Remember previous conversation turns
* 🔍 Answer questions about previous conversations
* 🌐 Provide a FastAPI REST API
* 💻 Provide an interactive web interface
* 💾 Store conversations using SQLite
* 🧪 Run automated unit and API tests
* 🐳 Run using Docker

## 🛠️ Technologies Used

### Machine Learning & NLP

* Python
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Label Encoding
* Natural Language Processing (NLP)
* NLTK
* spaCy
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy
* SQLite

### Frontend

* HTML
* CSS
* JavaScript
* Jinja2 Templates

### Testing & Deployment

* Pytest
* HTTPX
* Docker
* Docker Compose

## 📂 Project Structure

```text
intermediate-ai-chatbot/
│
├── chatbot/
│   ├── app.py
│   ├── chatbot.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── train_model.py
│   │
│   ├── intents.json
│   ├── knowledge_base.json
│   ├── requirements.txt
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py
│   │   ├── knowledge_service.py
│   │   ├── memory.py
│   │   └── response_generator.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── trained_model/
│   │   ├── intent_model.joblib
│   │   ├── vectorizer.joblib
│   │   └── label_encoder.joblib
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   │
│   └── tests/
│       ├── conftest.py
│       ├── test_api.py
│       ├── test_database.py
│       ├── test_intent_classifier.py
│       └── test_response_generator.py
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔄 Workflow

The overall chatbot workflow is:

```text
                User Message
                     │
                     ▼
             Text Preprocessing
                     │
                     ▼
             Intent Classification
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        High Confidence   Low Confidence
              │             │
              ▼             ▼
        Intent Handling   Knowledge Base
              │             │
              └──────┬──────┘
                     ▼
              Response Generation
                     │
                     ▼
              Save Conversation
                     │
                     ▼
              Return Response
                     │
                     ▼
              Web Interface
```

## 🧠 Intent Classification

The chatbot uses a supervised machine learning pipeline to classify user messages into predefined intents.

The training data is stored in:

```text
intents.json
```

Example intents include:

* `greeting`
* `goodbye`
* `thanks`
* `help`
* `about`
* `weather`
* `college_information`
* `ai_questions`
* `unknown`

### Machine Learning Pipeline

The intent classifier uses:

```text
User Text
    │
    ▼
Text Preprocessing
    │
    ▼
TF-IDF Vectorization
    │
    ▼
Logistic Regression
    │
    ▼
Intent Prediction
    │
    ▼
Confidence Score
```

The TF-IDF vectorizer uses both unigram and bigram features.

The trained model artifacts are stored in:

```text
trained_model/
```

including:

* `intent_model.joblib`
* `vectorizer.joblib`
* `label_encoder.joblib`

## 📚 Knowledge Base

The chatbot contains a curated knowledge base stored in:

```text
knowledge_base.json
```

The knowledge base provides information for topics such as:

* College information
* Admissions
* Fees
* Placements
* Facilities
* Artificial Intelligence
* Machine Learning
* Deep Learning
* NLP
* Neural Networks

The chatbot uses **keyword-overlap matching** to identify the most relevant knowledge-base entry.

## 🧠 Conversation Memory

The chatbot supports persistent conversation memory.

Conversation data is stored using:

```text
SQLite
```

Each conversation session contains:

* User message
* Bot response
* Detected intent
* Confidence score
* Timestamp

The chatbot can also answer memory-related questions such as:

```text
What did I ask before?
```

or:

```text
Summarize our conversation
```

## 📊 Features

* 🤖 Machine learning-based intent classification
* 🧠 TF-IDF + Logistic Regression
* 📚 JSON-based knowledge base
* 💬 Conversational responses
* 🧠 Conversation memory
* 💾 Persistent SQLite storage
* 📈 Intent confidence scores
* 🌐 FastAPI REST API
* 💻 Interactive web chat interface
* 🔄 Conversation history retrieval
* 🗑️ Conversation history deletion
* ❤️ Health-check endpoint
* 🧪 Automated unit and API tests
* 🐳 Docker support
* 📝 Configurable application settings
* 📋 Structured API responses using Pydantic

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/intermediate-ai-chatbot.git
cd intermediate-ai-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r chatbot/requirements.txt
```

## ▶️ Running the Project

Navigate to the chatbot directory:

```bash
cd chatbot
```

Start the FastAPI application:

```bash
uvicorn app:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

Open the URL in your browser to access the chatbot.

## 🌐 API

The application provides a REST API through FastAPI.

### Chat

```text
POST /chat
```

Example request:

```json
{
  "message": "What is machine learning?",
  "session_id": null
}
```

The API returns:

```json
{
  "response": "...",
  "intent": "ai_questions",
  "confidence": 0.95,
  "session_id": "..."
}
```

### Conversation History

```text
GET /history/{session_id}
```

Returns the conversation history associated with a session.

### Delete History

```text
DELETE /history/{session_id}
```

Deletes all stored conversation turns for a session.

### Health Check

```text
GET /health
```

Returns the application status and version.

## 📖 API Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

You can use the Swagger interface to test the chatbot API directly from the browser.

## 📓 Training the Model

The intent classifier can be trained using:

```bash
python train_model.py
```

The training process:

1. Loads training patterns from `intents.json`
2. Preprocesses the text
3. Generates TF-IDF features
4. Encodes intent labels
5. Trains a Logistic Regression classifier
6. Saves the trained model
7. Saves the TF-IDF vectorizer
8. Saves the label encoder

The resulting artifacts are stored in:

```text
trained_model/
```

## 🧪 Testing

The project includes automated tests using **Pytest**.

Run all tests from the `chatbot` directory:

```bash
pytest
```

The test suite covers:

* API functionality
* Database operations
* Intent classification
* Response generation
* Conversation handling

## 🐳 Docker

The project includes:

```text
Dockerfile
docker-compose.yml
```

Build and run the application using Docker Compose:

```bash
docker-compose up --build
```

The application can then be accessed through the exposed application port.

## 🔮 Future Improvements

Possible improvements include:

* Transformer-based intent classification
* BERT-based NLP models
* Semantic/vector-based knowledge retrieval
* Retrieval-Augmented Generation (RAG)
* Large Language Model integration
* Multi-language support
* Voice input and speech recognition
* Text-to-speech responses
* User authentication
* Cloud deployment
* Redis-based session management
* PostgreSQL database support
* Advanced conversation summarization
* Real-time external API integration
* Improved contextual understanding

## ⚠️ Limitations

The current chatbot is designed as an **intermediate educational AI project**.

Its responses are primarily based on:

* Predefined intents
* A curated knowledge base
* Keyword matching
* Stored conversation history

It does not currently use a large language model or live external knowledge retrieval.

