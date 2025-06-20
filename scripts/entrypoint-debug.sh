#!/bin/bash
# Debug entrypoint script to verify environment variables
set -e

echo "Starting CLARITY Backend (DEBUG MODE)..."
echo "============================================"
echo "Environment Variables Check:"
echo "ENVIRONMENT: ${ENVIRONMENT}"
echo "SECRET_KEY exists: $(if [ -n "$SECRET_KEY" ]; then echo "YES (length: ${#SECRET_KEY})"; else echo "NO"; fi)"
echo "GEMINI_API_KEY exists: $(if [ -n "$GEMINI_API_KEY" ]; then echo "YES (length: ${#GEMINI_API_KEY})"; else echo "NO"; fi)"
echo "============================================"

# Add a small delay to ensure secrets are injected
echo "Waiting 5 seconds for secret injection..."
sleep 5

echo "After wait:"
echo "SECRET_KEY exists: $(if [ -n "$SECRET_KEY" ]; then echo "YES (length: ${#SECRET_KEY})"; else echo "NO"; fi)"
echo "GEMINI_API_KEY exists: $(if [ -n "$GEMINI_API_KEY" ]; then echo "YES (length: ${#GEMINI_API_KEY})"; else echo "NO"; fi)"
echo "============================================"

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
exec gunicorn -c gunicorn.aws.conf.py clarity.main:app