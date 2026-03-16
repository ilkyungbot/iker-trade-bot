FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "print('ok')"

CMD ["python", "src/main.py"]
