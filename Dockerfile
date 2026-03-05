FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "memu-py[postgres]>=1.4.0" \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.23.0" \
    "python-dotenv>=1.0.0"

COPY server.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
