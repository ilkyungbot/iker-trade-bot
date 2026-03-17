FROM python:3.13-slim

WORKDIR /app

# 의존성만 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사 (site-packages에 설치하지 않음)
COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "print('ok')"

CMD ["python", "src/main.py"]
