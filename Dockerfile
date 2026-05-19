FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by tree-sitter builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Chroma persist dir inside container (mount a volume in production)
ENV CHROMA_PERSIST_DIR=/data/chroma
RUN mkdir -p /data/chroma

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
