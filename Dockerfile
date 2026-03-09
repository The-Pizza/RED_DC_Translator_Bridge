FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for fasttext
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]