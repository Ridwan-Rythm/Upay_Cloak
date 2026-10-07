# syntax=docker/dockerfile:1
# UpayShield: ONE container. FastAPI serves the API and the static dashboard in Frontend/ on port 8000.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---- base: OS library LightGBM needs (libgomp1) -----------------------------------------------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- deps: Python packages in a venv. Only requirements.txt is copied, so this layer is cached
#      until requirements.txt changes (source edits do not re-install anything). ------------------
FROM base AS deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- artifacts: decide where models/risk_engine.joblib comes from ------------------------------
# 1. If models/risk_engine.joblib is in the build context AND loads with the library versions
#    installed above, it is used as-is.
# 2. Otherwise (file missing, e.g. a fresh git clone, or it cannot be unpickled) `python -m ml.train`
#    runs here, at build time. It also rewrites data/cache/scored_cache.parquet and reports/metrics.json.
FROM base AS artifacts
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . /app
RUN mkdir -p models data/cache reports \
    && if [ -f models/risk_engine.joblib ] \
          && python -c "from ml.score import RiskEngine; RiskEngine('models/risk_engine.joblib')"; then \
         echo ">> using the model artifact from the build context"; \
       else \
         echo ">> no usable models/risk_engine.joblib -> training at build time"; \
         python -m ml.train; \
       fi \
    && test -f models/risk_engine.joblib && test -f reports/metrics.json && test -f data/cache/scored_cache.parquet

# ---- runtime ---------------------------------------------------------------------------------
FROM base AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    UPAY_DB_PATH=/app/dbdata/upayshield.db
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --chown=app:app . /app
COPY --from=artifacts --chown=app:app /app/models /app/models
COPY --from=artifacts --chown=app:app /app/data/cache /app/data/cache
COPY --from=artifacts --chown=app:app /app/reports /app/reports
# SQLite lives alone in /app/dbdata so a volume can be mounted there without hiding the CSVs/cache.
RUN mkdir -p /app/dbdata && chown -R app:app /app/dbdata /app/data /app/reports /app/models
USER app

EXPOSE 8000

# status must be "ok" (model loaded). Start-up replays the held-out period (~10-15 s), so allow 90 s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); sys.exit(0 if r.get('status')=='ok' else 1)"]

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
