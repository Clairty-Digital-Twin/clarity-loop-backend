# 🚀 CLARITY Demo Troubleshooting Guide

## Quick Demo Script Issues

If the `quick_demo.sh` script fails, here are the most common issues and solutions:

### 1. Script Exits Immediately with Error
**Problem**: The script uses strict error handling and exits on any failure.

**Solution**: 
```bash
# Check if Docker is running
docker version

# Check if ports are available
netstat -tuln | grep ':8000\|:3000\|:9090'

# Create .env file if missing
cp .env.example .env

# Try the demo again
bash quick_demo.sh
```

### 2. Missing Prerequisites

**Docker not installed:**
```bash
# Install Docker Desktop
# Visit: https://docs.docker.com/get-docker/
```

**Docker not running:**
```bash
# Start Docker Desktop or
sudo service docker start
```

**Port conflicts:**
```bash
# Find what's using the ports
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
lsof -ti:3000 | xargs kill -9  # Kill process on port 3000
```

### 3. Environment File Issues

**Missing .env file:**
```bash
# Copy the example file
cp .env.example .env

# Or create a minimal one
cat > .env << 'EOF'
ENVIRONMENT=development
DEBUG=true
GOOGLE_CLOUD_PROJECT=clarity-demo
FIREBASE_PROJECT_ID=clarity-demo
SKIP_EXTERNAL_SERVICES=true
DEMO_MODE=true
EOF
```

### 4. Docker Compose Version Issues

**Using older Docker Compose:**
```bash
# Check version
docker-compose version

# Or use newer syntax
docker compose version
```

### 5. Permission Issues (Linux/Mac)

```bash
# Make script executable
chmod +x quick_demo.sh
chmod +x scripts/demo_deployment.sh

# Run Docker without sudo (Linux)
sudo usermod -aG docker $USER
# Then logout and login again
```

### 6. Network Issues

**Services not responding:**
```bash
# Check Docker networks
docker network ls

# Restart Docker
sudo service docker restart  # Linux
# Or restart Docker Desktop

# Check if containers are running
docker ps
```

### 7. Manual Demo Startup

If the script fails, you can start manually:

```bash
# Create .env file
cp .env.example .env

# Start services one by one
docker-compose up redis -d
docker-compose up firestore -d
docker-compose up clarity-backend -d
docker-compose up prometheus -d
docker-compose up grafana -d

# Check status
docker-compose ps
```

### 8. Testing the Demo

Once running, test these URLs:

- ✅ **Main API**: http://localhost:8000
- ✅ **API Docs**: http://localhost:8000/docs  
- ✅ **Health Check**: http://localhost:8000/health
- ✅ **Grafana**: http://localhost:3000 (admin/admin)
- ✅ **Prometheus**: http://localhost:9090

### 9. Getting Help

**Check logs:**
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs clarity-backend

# Follow logs
docker-compose logs -f
```

**Clean slate restart:**
```bash
# Nuclear option - completely reset
docker-compose down --volumes --remove-orphans
docker system prune -f
bash quick_demo.sh
```

### 10. Platform-Specific Issues

**macOS:**
- Ensure Docker Desktop is running
- Check if ports are blocked by firewall

**Linux:**
- User permissions for Docker
- Firewall/iptables blocking ports

**Windows:**
- WSL2 backend enabled in Docker Desktop
- Windows Defender allowing Docker
- Hyper-V enabled

## Success Indicators

✅ **Script completes without errors**  
✅ **All services show as "Up" in `docker-compose ps`**  
✅ **HTTP requests to localhost:8000 return responses**  
✅ **Grafana dashboard loads at localhost:3000**

## Still Having Issues?

1. **Check the specific error message** in the script output
2. **Verify all prerequisites** are installed and running
3. **Try the manual startup** process above
4. **Check Docker resources** (CPU/Memory limits)
5. **Restart your machine** (sometimes helps with Docker issues)

The demo is designed to be resilient, but different machine configurations can cause unexpected issues. The improved script should handle most common problems automatically.