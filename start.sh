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

# Start the hybrid API server (with Lambda support)
exec python -m uvicorn api_hybrid:app --host 0.0.0.0 --port "$PORT"
