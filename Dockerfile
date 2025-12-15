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

COPY technic_v4 ./technic_v4
COPY models ./models
COPY start.sh ./start.sh

# Make start.sh executable
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8502

CMD ["bash", "start.sh"]

