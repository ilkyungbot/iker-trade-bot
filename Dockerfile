FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -r -s /bin/false botuser \
    && mkdir -p /app/data \
    && chown botuser:botuser /app/data

COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

USER botuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sqlite3; sqlite3.connect('/app/data/signal_bot.db').execute('SELECT 1')" || exit 1

CMD ["python", "src/main.py"]
