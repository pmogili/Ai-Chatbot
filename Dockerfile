FROM python:3.12-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout (for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed to build some Python packages (e.g. scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NLTK corpora used by the preprocessing pipeline so the
# container doesn't need network access to them at runtime.
RUN python -c "import nltk; \
    nltk.download('punkt_tab'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('omw-1.4')"

# Copy the application source
COPY . .

# Directory for the persistent SQLite database (mounted as a volume in
# docker-compose.yml)
RUN mkdir -p /app/data

# Train the intent classifier at build time so the container starts
# instantly (no first-request training delay).
RUN python train_model.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
