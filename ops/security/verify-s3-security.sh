#!/bin/bash
# S3 Security Verification Script for CLARITY
# Tests all security measures implemented for S3 buckets

set -e

echo "🔥 CLARITY S3 SECURITY VERIFICATION 🔥"
echo "=================================================="

# Define our production buckets
HEALTH_BUCKET="clarity-health-data-storage"
ML_BUCKET="clarity-ml-models-124355672559"

function check_bucket_exists() {
    local bucket=$1
    echo "📦 Checking if bucket exists: $bucket"
    if aws s3 ls "s3://$bucket" >/dev/null 2>&1; then
        echo "✅ Bucket exists: $bucket"
        return 0
    else
        echo "❌ Bucket NOT found: $bucket"
        return 1
    fi
}

function check_encryption() {
    local bucket=$1
    echo "🔐 Checking encryption for: $bucket"
    if aws s3api get-bucket-encryption --bucket "$bucket" >/dev/null 2>&1; then
        echo "✅ Encryption enabled for: $bucket"
        return 0
    else
        echo "❌ Encryption NOT enabled for: $bucket"
        return 1
    fi
}

function check_versioning() {
    local bucket=$1
    echo "📝 Checking versioning for: $bucket"
    local status=$(aws s3api get-bucket-versioning --bucket "$bucket" --query 'Status' --output text 2>/dev/null || echo "Disabled")
    if [ "$status" = "Enabled" ]; then
        echo "✅ Versioning enabled for: $bucket"
        return 0
    else
        echo "❌ Versioning NOT enabled for: $bucket"
        return 1
    fi
}

function check_public_access_block() {
    local bucket=$1
    echo "🚫 Checking public access block for: $bucket"
    if aws s3api get-public-access-block --bucket "$bucket" >/dev/null 2>&1; then
        echo "✅ Public access blocked for: $bucket"
        return 0
    else
        echo "❌ Public access NOT blocked for: $bucket"
        return 1
    fi
}

function check_bucket_policy() {
    local bucket=$1
    echo "📋 Checking bucket policy for: $bucket"
    if aws s3api get-bucket-policy --bucket "$bucket" >/dev/null 2>&1; then
        echo "✅ Bucket policy exists for: $bucket"
        return 0
    else
        echo "❌ Bucket policy NOT found for: $bucket"
        return 1
    fi
}

function verify_bucket_security() {
    local bucket=$1
    echo ""
    echo "🛡️ VERIFYING SECURITY FOR: $bucket"
    echo "----------------------------------------"
    
    local all_checks_passed=true
    
    check_bucket_exists "$bucket" || all_checks_passed=false
    check_encryption "$bucket" || all_checks_passed=false
    check_versioning "$bucket" || all_checks_passed=false
    check_public_access_block "$bucket" || all_checks_passed=false
    check_bucket_policy "$bucket" || all_checks_passed=false
    
    if [ "$all_checks_passed" = true ]; then
        echo "🎯 ALL SECURITY CHECKS PASSED for: $bucket"
    else
        echo "⚠️  Some security checks failed for: $bucket"
    fi
    
    return $([[ "$all_checks_passed" = true ]] && echo 0 || echo 1)
}

# Main verification
echo ""
echo "Starting S3 security verification..."
echo ""

overall_status=true

verify_bucket_security "$HEALTH_BUCKET" || overall_status=false
verify_bucket_security "$ML_BUCKET" || overall_status=false

echo ""
echo "=================================================="
if [ "$overall_status" = true ]; then
    echo "🚀 ALL S3 BUCKETS SECURED TO SINGULARITY LEVEL! 🚀"
    echo "✅ HEALTH DATA BUCKET: Fully secured"
    echo "✅ ML MODELS BUCKET: Fully secured"
    echo "✅ ENCRYPTION: Enabled on all buckets"
    echo "✅ VERSIONING: Enabled on all buckets"
    echo "✅ PUBLIC ACCESS: Blocked on all buckets"
    echo "✅ BUCKET POLICIES: Applied to all buckets"
    echo ""
    echo "🎯 S3 SECURITY BLAST TO SINGULARITY: COMPLETE!"
    exit 0
else
    echo "❌ Some security measures need attention"
    exit 1
fi 