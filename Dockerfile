FROM python:3.13-slim

WORKDIR /app

# System deps for TA-Lib C library + Python package compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make wget && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir . && \
    apt-get purge -y gcc g++ make wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["python", "src/main.py"]
