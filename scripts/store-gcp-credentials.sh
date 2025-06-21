#!/bin/bash

# Script to store Google service account JSON in AWS Secrets Manager
# and update ECS task definition

set -e

echo "=== Storing GCP Service Account in AWS Secrets Manager ==="

# Path to the service account JSON file
SERVICE_ACCOUNT_FILE="$HOME/.clarity-secrets/clarity-loop-backend-f770782498c7.json"
SECRET_NAME="clarity/gcp-service-account"
AWS_REGION="us-east-1"

# Check if the file exists
if [ ! -f "$SERVICE_ACCOUNT_FILE" ]; then
    echo "Error: Service account file not found: $SERVICE_ACCOUNT_FILE"
    exit 1
fi

# Read and minify the JSON content
JSON_CONTENT=$(cat "$SERVICE_ACCOUNT_FILE" | jq -c .)

# Check if secret already exists
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" 2>/dev/null; then
    echo "Secret already exists. Updating..."
    aws secretsmanager update-secret \
        --secret-id "$SECRET_NAME" \
        --secret-string "$JSON_CONTENT" \
        --region "$AWS_REGION"
else
    echo "Creating new secret..."
    aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --description "Google Cloud service account credentials for Vertex AI" \
        --secret-string "$JSON_CONTENT" \
        --region "$AWS_REGION"
fi

# Get the secret ARN
SECRET_ARN=$(aws secretsmanager describe-secret \
    --secret-id "$SECRET_NAME" \
    --region "$AWS_REGION" \
    --query 'ARN' \
    --output text)

echo "Secret stored successfully. ARN: $SECRET_ARN"

# Move the service account file to a secure location outside git repo
SECURE_DIR="$HOME/.clarity-secrets"
mkdir -p "$SECURE_DIR"
mv "$SERVICE_ACCOUNT_FILE" "$SECURE_DIR/"
echo "Service account file moved to: $SECURE_DIR/$(basename $SERVICE_ACCOUNT_FILE)"

echo ""
echo "=== Next Steps ==="
echo "1. Update the ECS task definition to include the secret:"
echo "   Add this to the 'secrets' array in ops/ecs-task-definition.json:"
echo ""
echo '        {'
echo '          "name": "GOOGLE_APPLICATION_CREDENTIALS_JSON",'
echo "          \"valueFrom\": \"$SECRET_ARN\""
echo '        }'
echo ""
echo "2. Update your application code to use the environment variable:"
echo "   - Read the JSON from environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON"
echo "   - Parse it and use it to initialize Google Cloud clients"
echo ""
echo "3. Ensure the ECS task execution role has permission to access this secret"