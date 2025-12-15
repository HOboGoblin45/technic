#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlink to training data on persistent disk
if [ -f "/opt/render/project/data/training_data_v2.parquet" ]; then
    ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet
    echo "✅ Symlink created for training_data_v2.parquet"
else
    echo "⚠️  Warning: training_data_v2.parquet not found on persistent disk"
fi

# Start the API server
exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT"
