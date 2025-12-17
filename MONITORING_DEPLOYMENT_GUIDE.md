# Monitoring System Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the ML Monitoring System to production.

## System Architecture

```
┌─────────────────────┐
│   ML API :8002      │
│  (Monitored)        │
└──────────┬──────────┘
           │ metrics
           ▼
┌─────────────────────┐
│ Monitoring API      │
│     :8003           │
└──────────┬──────────┘
           │ data
           ▼
┌─────────────────────┐
│   Dashboard         │
│     :8502/8504      │
└─────────────────────┘
```

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Optional
- Docker & Docker Compose (for containerized deployment)
- Redis (for enhanced caching - optional)
- Nginx (for reverse proxy - recommended for production)

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Start Services

**Terminal 1: Monitoring API**
```bash
python monitoring_api.py
```

**Terminal 2: Dashboard**
```bash
streamlit run monitoring_dashboard_enhanced.py --server.port 8504
```

**Terminal 3: ML API (Optional)**
```bash
python api_ml_monitored.py
```

### 3. Verify Deployment

- Monitoring API: http://localhost:8003/health
- API Documentation: http://localhost:8003/docs
- Dashboard: http://localhost:8504

## Production Deployment

### Option 1: Docker Deployment (Recommended)

#### 1. Create Docker Compose Configuration

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  monitoring-api:
    build:
      context: .
      dockerfile: Dockerfile.monitoring
    ports:
      - "8003:8003"
    environment:
      - PYTHON_ENV=production
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8504:8504"
    environment:
      - MONITORING_API_URL=http://monitoring-api:8003
    depends_on:
      - monitoring-api
    restart: unless-stopped

  ml-api:
    build:
      context: .
      dockerfile: Dockerfile.ml
    ports:
      - "8002:8002"
    environment:
      - MONITORING_API_URL=http://monitoring-api:8003
    depends_on:
      - monitoring-api
    restart: unless-stopped

networks:
  default:
    name: monitoring-network
```

#### 2. Create Dockerfiles

**Dockerfile.monitoring:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring_api.py .
COPY technic_v4/ ./technic_v4/

EXPOSE 8003

CMD ["python", "monitoring_api.py"]
```

**Dockerfile.dashboard:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

COPY monitoring_dashboard_enhanced.py .

EXPOSE 8504

CMD ["streamlit", "run", "monitoring_dashboard_enhanced.py", "--server.port=8504", "--server.address=0.0.0.0"]
```

**Dockerfile.ml:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_ml_monitored.py .
COPY technic_v4/ ./technic_v4/

EXPOSE 8002

CMD ["python", "api_ml_monitored.py"]
```

#### 3. Deploy with Docker Compose

```bash
# Build and start services
docker-compose -f docker-compose.monitoring.yml up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Stop services
docker-compose -f docker-compose.monitoring.yml down
```

### Option 2: Systemd Service (Linux)

#### 1. Create Service Files

**monitoring-api.service:**
```ini
[Unit]
Description=ML Monitoring API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/project/.venv/bin"
ExecStart=/path/to/project/.venv/bin/python monitoring_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**monitoring-dashboard.service:**
```ini
[Unit]
Description=ML Monitoring Dashboard
After=network.target monitoring-api.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/project/.venv/bin"
ExecStart=/path/to/project/.venv/bin/streamlit run monitoring_dashboard_enhanced.py --server.port=8504
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. Enable and Start Services

```bash
# Copy service files
sudo cp monitoring-api.service /etc/systemd/system/
sudo cp monitoring-dashboard.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable monitoring-api
sudo systemctl enable monitoring-dashboard

# Start services
sudo systemctl start monitoring-api
sudo systemctl start monitoring-dashboard

# Check status
sudo systemctl status monitoring-api
sudo systemctl status monitoring-dashboard
```

### Option 3: Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [
    {
      name: 'monitoring-api',
      script: 'python',
      args: 'monitoring_api.py',
      cwd: '/path/to/project',
      interpreter: '/path/to/project/.venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G'
    },
    {
      name: 'monitoring-dashboard',
      script: 'streamlit',
      args: 'run monitoring_dashboard_enhanced.py --server.port=8504',
      cwd: '/path/to/project',
      interpreter: '/path/to/project/.venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G'
    }
  ]
};
EOF

# Start services
pm2 start ecosystem.config.js

# Save configuration
pm2 save

# Setup startup script
pm2 startup
```

## Environment Variables

### Monitoring API

```bash
# Optional: Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password

# Logging
LOG_LEVEL=info  # debug, info, warning, error

# API Configuration
API_HOST=0.0.0.0
API_PORT=8003
```

### Dashboard

```bash
# Monitoring API URL
MONITORING_API_URL=http://localhost:8003

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8504
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### ML API

```bash
# Monitoring integration
MONITORING_API_URL=http://localhost:8003
ENABLE_MONITORING=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8002
```

## Nginx Reverse Proxy (Recommended)

### Configuration

Create `/etc/nginx/sites-available/monitoring`:

```nginx
# Monitoring API
server {
    listen 80;
    server_name monitoring-api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8003;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://localhost:8003;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Dashboard
server {
    listen 80;
    server_name dashboard.yourdomain.com;

    location / {
        proxy_pass http://localhost:8504;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Streamlit WebSocket
    location /_stcore/stream {
        proxy_pass http://localhost:8504/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/monitoring /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## SSL/TLS Configuration (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d monitoring-api.yourdomain.com
sudo certbot --nginx -d dashboard.yourdomain.com

# Auto-renewal is configured automatically
```

## Monitoring & Maintenance

### Health Checks

```bash
# Check Monitoring API
curl http://localhost:8003/health

# Check metrics
curl http://localhost:8003/metrics/current

# Check alerts
curl http://localhost:8003/alerts/active
```

### Log Management

```bash
# View logs (systemd)
sudo journalctl -u monitoring-api -f
sudo journalctl -u monitoring-dashboard -f

# View logs (Docker)
docker-compose -f docker-compose.monitoring.yml logs -f monitoring-api
docker-compose -f docker-compose.monitoring.yml logs -f dashboard

# View logs (PM2)
pm2 logs monitoring-api
pm2 logs monitoring-dashboard
```

### Backup & Recovery

```bash
# Backup configuration
tar -czf monitoring-backup-$(date +%Y%m%d).tar.gz \
    monitoring_api.py \
    monitoring_dashboard_enhanced.py \
    api_ml_monitored.py \
    technic_v4/monitoring/ \
    requirements.txt

# Restore from backup
tar -xzf monitoring-backup-YYYYMMDD.tar.gz
```

## Performance Tuning

### Monitoring API

```python
# In monitoring_api.py, adjust worker count
if __name__ == "__main__":
    uvicorn.run(
        "monitoring_api:app",
        host="0.0.0.0",
        port=8003,
        workers=4,  # Adjust based on CPU cores
        log_level="info"
    )
```

### Dashboard

```bash
# Increase Streamlit performance
streamlit run monitoring_dashboard_enhanced.py \
    --server.port=8504 \
    --server.maxUploadSize=200 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true
```

## Troubleshooting

### Issue: Services won't start

**Solution:**
```bash
# Check if ports are in use
netstat -tulpn | grep :8003
netstat -tulpn | grep :8504

# Kill processes using the ports
kill -9 $(lsof -t -i:8003)
kill -9 $(lsof -t -i:8504)
```

### Issue: Dashboard not connecting to API

**Solution:**
1. Verify Monitoring API is running: `curl http://localhost:8003/health`
2. Check firewall rules
3. Verify MONITORING_API_URL environment variable
4. Check network connectivity between services

### Issue: High memory usage

**Solution:**
```bash
# Limit metrics history
# In technic_v4/monitoring/metrics_collector.py
# Reduce maxlen in deque initialization

# Restart services to apply changes
```

### Issue: Slow dashboard loading

**Solution:**
1. Reduce time range in historical queries
2. Enable Redis caching
3. Increase dashboard refresh interval
4. Optimize database queries

## Security Checklist

- [ ] Change default ports if exposed to internet
- [ ] Enable SSL/TLS for all endpoints
- [ ] Set up firewall rules
- [ ] Use strong passwords for Redis
- [ ] Enable authentication for APIs
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Implement rate limiting
- [ ] Use environment variables for secrets
- [ ] Regular backups

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Firewall rules configured
- [ ] SSL certificates obtained (if needed)
- [ ] Backup strategy in place

### Deployment
- [ ] Services started successfully
- [ ] Health checks passing
- [ ] Monitoring API accessible
- [ ] Dashboard loading correctly
- [ ] Metrics being collected
- [ ] Alerts configured

### Post-Deployment
- [ ] Monitor logs for errors
- [ ] Verify metrics accuracy
- [ ] Test alert notifications
- [ ] Document any issues
- [ ] Update documentation
- [ ] Notify team of deployment

## Support & Resources

### Documentation
- API Documentation: http://localhost:8003/docs
- Streamlit Docs: https://docs.streamlit.io
- FastAPI Docs: https://fastapi.tiangolo.com

### Monitoring Endpoints
- Health: `/health`
- Current Metrics: `/metrics/current`
- Historical Metrics: `/metrics/history`
- Alerts: `/alerts/active`
- API Docs: `/docs`

### Contact
For issues or questions, refer to:
- Project README.md
- PHASE2_DAY3_PROGRESS_SUMMARY.md
- Technical documentation in docs/

## Conclusion

This monitoring system is production-ready and can be deployed using any of the methods described above. Choose the deployment method that best fits your infrastructure and requirements.

For additional features or customization, refer to the remaining tasks in PHASE2_DAY3_PLAN.md.
