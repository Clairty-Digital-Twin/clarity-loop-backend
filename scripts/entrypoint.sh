#!/bin/bash
# Entrypoint script that downloads models before starting the application
set -e

echo "Starting CLARITY Backend..."

# Download models if not already present
if [ ! -f "/app/models/pat/.models_downloaded" ]; then
    echo "Downloading ML models from S3..."
    if /app/scripts/download_models.sh; then
        echo "✅ Models downloaded successfully"
    else
        echo "⚠️ Model download failed, but continuing to start the app..."
        echo "📝 ML endpoints may not be available until models are present"
    fi
else
    echo "Models already downloaded, skipping..."
fi

# Start the application
echo "Starting Gunicorn server..."
# Unified main.py automatically detects production environment and uses bulletproof mode
exec gunicorn -c gunicorn.aws.conf.py clarity.main:app