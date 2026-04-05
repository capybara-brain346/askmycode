FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

COPY src/ ./src/
COPY config.json .

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/logs \
    && chown -R appuser /app
USER appuser

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
