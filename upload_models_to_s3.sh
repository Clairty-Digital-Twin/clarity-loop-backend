#!/bin/bash
# Upload PAT models to S3 with both original and expected names

set -e

echo "🚀 UPLOADING PAT MODELS TO S3..."

BUCKET="s3://clarity-ml-models-124355672559"
LOCAL_DIR="/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/clarity-loop-backend/models/pat"
REGION="us-east-1"

# Upload with original names
echo "📤 Uploading PAT-S_29k_weights.h5..."
aws s3 cp "$LOCAL_DIR/PAT-S_29k_weights.h5" "$BUCKET/pat/PAT-S_29k_weights.h5" --region $REGION

echo "📤 Uploading PAT-M_29k_weights.h5..."
aws s3 cp "$LOCAL_DIR/PAT-M_29k_weights.h5" "$BUCKET/pat/PAT-M_29k_weights.h5" --region $REGION

echo "📤 Uploading PAT-L_29k_weights.h5..."
aws s3 cp "$LOCAL_DIR/PAT-L_29k_weights.h5" "$BUCKET/pat/PAT-L_29k_weights.h5" --region $REGION

# Also upload with the transformer names our code expects
echo "📤 Uploading as transformer variants..."
aws s3 cp "$LOCAL_DIR/PAT-S_29k_weights.h5" "$BUCKET/pat/PAT-S_29k_weight_transformer.h5" --region $REGION
aws s3 cp "$LOCAL_DIR/PAT-M_29k_weights.h5" "$BUCKET/pat/PAT-M_29k_weight_transformer.h5" --region $REGION
aws s3 cp "$LOCAL_DIR/PAT-L_29k_weights.h5" "$BUCKET/pat/PAT-L_91k_weight_transformer.h5" --region $REGION

echo "✅ All models uploaded!"
echo ""
echo "📊 Verifying uploads..."
aws s3 ls "$BUCKET/pat/" --region $REGION || echo "⚠️  Can't list bucket but uploads should be there"

echo ""
echo "🎯 Next step: Restart ECS service to download fresh models"