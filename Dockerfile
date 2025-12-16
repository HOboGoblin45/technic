FROM python:3.10.11-slim

WORKDIR /app

# Layer 1: System dependencies (rarely changes)
RUN apt-get update && apt-get install -y \
    build-essential \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python dependencies (changes only when requirements.txt changes)
# This layer will be CACHED on code changes, saving 10-12 minutes!
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Application code (changes frequently, but doesn't invalidate pip cache)
COPY technic_v4 ./technic_v4
COPY models ./models
COPY start.sh ./start.sh
COPY api.py ./api.py

# Make start.sh executable
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8502

CMD ["bash", "start.sh"]

