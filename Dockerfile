FROM python:3.10.11-slim

WORKDIR /app

# System dependencies (if you need any compiled stuff)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions if present
COPY requirements.txt ./ 

# Install Python dependencies
RUN if [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the application code
COPY technic_v4 ./technic_v4

# Environment
ENV PYTHONUNBUFFERED=1

# 8502 is fine locally; on Render we'll use $PORT
EXPOSE 8502

CMD ["sh", "-c", "uvicorn technic_v4.api_server:app --host 0.0.0.0 --port ${PORT:-8502}"]
