#!/bin/bash
# Entrypoint script that downloads models before starting the application
set -e

echo "Starting CLARITY Backend..."

# Download models if not already present
if [ ! -f "/app/models/pat/.models_downloaded" ]; then
    echo "Downloading ML models from S3..."
    /app/scripts/download_models.sh
else
    echo "Models already downloaded, skipping..."
fi

# Start the application
echo "Starting Gunicorn server..."
exec gunicorn -c gunicorn.aws.conf.py clarity.main:app