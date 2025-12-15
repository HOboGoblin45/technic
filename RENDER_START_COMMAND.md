# Render Start Command (copy/paste ready)

One-liner (raw text, no code fences):
mkdir -p data && ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet && python -m uvicorn api:app --host 0.0.0.0 --port $PORT

Where to paste:
1) Render dashboard -> your service -> Settings -> Start Command  
2) Replace the existing value with the line above, then Save (Render redeploys automatically)

What it does:
- mkdir -p data: ensure local data folder exists
- ln -sf /opt/render/project/data/training_data_v2.parquet data/training_data_v2.parquet: symlink to persistent disk
- python -m uvicorn api:app --host 0.0.0.0 --port $PORT: start the API

Quick check after deploy:
[META] loaded meta experience from data/training_data_v2.parquet
