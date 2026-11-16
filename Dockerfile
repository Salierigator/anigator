# Render.com (Docker, free tier): the backend in real mode. Build context = repo root.
#
# The model bundle (export/twotower/) is gitignored, so it is not in the build context: it is downloaded
# from a Hugging Face model repo at BUILD time and baked into the image (no download on cold start).
FROM python:3.11-slim

# libgomp1 = LightGBM's OpenMP runtime. python:*-slim lacks it, so `import lightgbm` dies at
# boot: "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Run as non-root uid 1000 (good practice; Render does not require it).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# 1) deps: the most expensive layer, first for the cache. torch comes separately from the CPU-only index.
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-cache-dir -r requirements-deploy.txt && \
    pip install --no-cache-dir huggingface_hub==0.36.0

# 2) the bundle from the HF model repo (layout artifacts/ + data/cleaned/), all moved into
#    ./export/twotower/.
#    Another repo = change the default ARG here.
#    ARTIFACTS_REPO="" skips this step and expects export/twotower/ as a bind mount at run time
#    (local tests).
ARG ARTIFACTS_REPO=Salierigator/anime-recommender-artifacts
ENV ARTIFACTS_REPO=${ARTIFACTS_REPO}
RUN if [ -n "$ARTIFACTS_REPO" ]; then \
      python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id=os.environ['ARTIFACTS_REPO'], local_dir='.', \
allow_patterns=['artifacts/*', 'data/cleaned/*'])" && rm -rf .cache && \
      mkdir -p export/twotower && mv artifacts/* data/cleaned/* export/twotower/ && \
      rmdir artifacts data/cleaned data ; \
    else echo '>> skip the bundle download: expect a bind mount of export/twotower/ at run time' ; fi

# 3) code: changes most often, so it comes last and only this layer rebuilds on code edits.
COPY --chown=user . .

# MAL_CLIENT_ID + CORS_ORIGINS are set in the Render environment, never hard-coded here.
ENV MOCK_MODE=0

WORKDIR $HOME/app/service/backend
EXPOSE 7860
# Render passes $PORT at run time; a local docker run has none, so it falls back to 7860.
# Shell-form CMD is required to expand the variable.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
