# BizInsight AI - Dockerfile
# Builds a container image that runs the Streamlit app with all
# required dependencies (incl. NLTK/TextBlob corpora) baked in.

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

# torch is pulled from PyPI's default index by `pip install -r requirements.txt`,
# which serves CUDA-enabled wheels on Linux (2GB+) -- unnecessary for this
# CPU-only Streamlit app and a major contributor to image size/build time.
# Installing the official CPU-only build first means the later
# `pip install -r requirements.txt` sees torch>=2.2.0 already satisfied
# (pip's default "only-if-needed" upgrade strategy) and leaves it alone.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies first so this layer is cached
# unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK_DATA must be set BEFORE the downloader runs, otherwise the
# corpora land in the default location (e.g. /root/nltk_data) instead
# of the path below, and the app fails to find them at runtime and
# falls back to a slow/flaky download on first request.
# HF_HOME is set explicitly (rather than relying on the implicit
# ~/.cache/huggingface default) so the build-time download and the
# runtime lookup are guaranteed to use the same path.
ENV NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/usr/local/share/huggingface

# Pre-download everything the app needs to do inference offline:
#  - vader_lexicon / TextBlob corpora: used by sentiment.py / app.py
#  - SentenceTransformer models: used by clustering/vectorize.py and
#    sync_vectors.py / rag_api/config.py for embeddings
#  - CrossEncoder: used by rag_api/chains.py to rerank RAG results
# Baking these into the image means the first request doesn't have to
# pull ~hundreds of MB from the internet, and the app still works in
# network-restricted environments.
RUN python -m nltk.downloader vader_lexicon -d "$NLTK_DATA" \
    && python -m textblob.download_corpora \
    && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-mpnet-base-v2'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Now copy the rest of the application code.
COPY . .

# Create a data directory and symlink the database file into it.
# database.py does sqlite3.connect("bizinsight.db"), a path relative to
# /app, so /app/bizinsight.db now resolves through this symlink into
# /data/bizinsight.db. /data is meant to be mounted as a volume in
# docker-compose.yml.
#
# This avoids bind-mounting the .db file directly: a single-file bind
# mount doesn't give SQLite a real directory to create its sibling
# "-journal"/"-wal"/"-shm" files in, which can cause locking issues, and
# it requires the file to already exist on the host before the first
# `docker compose up` (otherwise Docker creates a directory in its place
# and the app crashes trying to open a directory as a database).
RUN mkdir -p /data && ln -sf /data/bizinsight.db /app/bizinsight.db

# Pre-create /app/chroma_db so it exists in the image *before* the
# corresponding volume is mounted onto it. Docker initializes a fresh
# named volume's ownership from whatever already exists at that path in
# the image -- if nothing exists there, the volume mounts as root-owned
# and the non-root appuser below can't write to it (ChromaDB would fail
# to persist embeddings at runtime).
RUN mkdir -p /app/chroma_db

# Run as a non-root user. By default containers run as root, which is
# unnecessary privilege for a Streamlit app and a real risk if any
# dependency has an exploitable vulnerability.
#
# chown covers:
#  - /app, /data: app code + writable data dirs (DB, chroma_db)
#  - $NLTK_DATA, $HF_HOME: the pre-downloaded caches are root-owned from
#    the build steps above. Reading cached files works fine under the
#    default 755 permissions, but huggingface_hub also writes a small
#    per-file *lock* into a ".locks/" subfolder of the cache root on
#    every load (even when the file is already cached, as part of its
#    normal concurrent-access safety mechanism) -- that's a write to the
#    cache root itself, which a non-owner can't do without this chown.
RUN useradd -u 10001 -m appuser \
    && chown -R appuser:appuser /app /data "$NLTK_DATA" "$HF_HOME"
USER appuser

# Streamlit's default port
EXPOSE 8501

# Container healthcheck against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
