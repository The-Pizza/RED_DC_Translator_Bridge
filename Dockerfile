FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for fasttext
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

RUN mkdir -p /data /app/models \
    && if [ -f /data/lid.176.bin ]; then mv /data/lid.176.bin /app/models/lid.176.bin; fi

ENV PYTHONUNBUFFERED=1
ENV FASTTEXT_MODEL_PATH=/app/models/lid.176.bin

CMD ["python", "bot.py"]