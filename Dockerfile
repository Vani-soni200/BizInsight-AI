# BizInsight AI - Dockerfile
# Builds a container image that runs the Streamlit app with all
# required dependencies (incl. NLTK/TextBlob corpora and ML models) baked in.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies:
#  - build-essential: needed to build some ML packages (hdbscan, etc.) from source
#  - curl: used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached
# unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK_DATA must be set BEFORE the downloader runs below, otherwise the
# corpora land in the default location (e.g. /root/nltk_data) instead of
# the path we declare here, and the app fails to find them at runtime,
# falling back to a slow/flaky download on first request.
ENV NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/usr/local/share/huggingface

# Pre-download everything the app needs to run inference offline:
#  - vader_lexicon / TextBlob corpora: used by sentiment.py / app.py
#  - SentenceTransformer models: used by clustering/vectorize.py and
#    sync_vectors.py / rag_api/config.py for embeddings
#  - CrossEncoder: used by rag_api/chains.py to rerank RAG results
# Baking these into the image avoids pulling hundreds of MB on first
# request and keeps startup fast and reliable.
RUN python -m nltk.downloader vader_lexicon -d "$NLTK_DATA" \
    && python -m textblob.download_corpora \
    && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-mpnet-base-v2'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Now copy the rest of the application code.
COPY . .

# Streamlit's default port
EXPOSE 8501

# Container healthcheck against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
