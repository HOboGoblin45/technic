FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install runtime dependencies (FastAPI/uvicorn included in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit and API ports
EXPOSE 8501 8000

# Default to serving the FastAPI app; override CMD for other entrypoints as needed
CMD ["uvicorn", "technic_v4.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
