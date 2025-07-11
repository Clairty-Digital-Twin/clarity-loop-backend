#!/bin/bash
# Create secure pre-signed URLs for Matt to download PAT models
# This is MUCH safer than making the bucket public!

set -e

BUCKET_NAME="clarity-ml-models-124355672559"
REGION="us-east-1"
EXPIRATION_HOURS="${1:-24}"  # Default 24 hours, can override

echo "🔐 Creating secure pre-signed URLs for Matt (expires in ${EXPIRATION_HOURS} hours)..."
echo ""

# Calculate expiration in seconds
EXPIRATION_SECONDS=$((EXPIRATION_HOURS * 3600))

# Generate pre-signed URLs for each model
echo "📧 Send these URLs to Matt (matthewgorbett@gmail.com):"
echo "=================================================="
echo ""

# PAT-S Model
echo "PAT-S Model (Small):"
aws s3 presign "s3://${BUCKET_NAME}/pat/PAT-S_29k_weights.h5" \
    --expires-in ${EXPIRATION_SECONDS} \
    --region ${REGION}
echo ""

# PAT-M Model
echo "PAT-M Model (Medium):"
aws s3 presign "s3://${BUCKET_NAME}/pat/PAT-M_29k_weights.h5" \
    --expires-in ${EXPIRATION_SECONDS} \
    --region ${REGION}
echo ""

# PAT-L Model
echo "PAT-L Model (Large):"
aws s3 presign "s3://${BUCKET_NAME}/pat/PAT-L_29k_weights.h5" \
    --expires-in ${EXPIRATION_SECONDS} \
    --region ${REGION}
echo ""

echo "=================================================="
echo ""
echo "✅ Pre-signed URLs generated successfully!"
echo "⏰ These URLs will expire in ${EXPIRATION_HOURS} hours"
echo ""
echo "📝 Instructions for Matt:"
echo "1. Download models using wget or curl:"
echo "   wget -O PAT-S_29k_weights.h5 '<URL>'"
echo "2. Place them in: ./models/pat/"
echo "3. Run the code as normal"
echo ""
echo "🔒 Your S3 bucket remains PRIVATE and SECURE!"