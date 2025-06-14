#!/bin/bash
# Monitor ACM certificate validation status

CERT_ARN="arn:aws:acm:us-east-1:124355672559:certificate/183ffae7-82d7-4259-a773-f52bb05c46d8"
REGION="us-east-1"

echo "Monitoring certificate validation status..."
echo "Certificate ARN: $CERT_ARN"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    STATUS=$(aws acm describe-certificate \
        --certificate-arn "$CERT_ARN" \
        --region "$REGION" \
        --query 'Certificate.Status' \
        --output text)
    
    DOMAIN_STATUSES=$(aws acm describe-certificate \
        --certificate-arn "$CERT_ARN" \
        --region "$REGION" \
        --query 'Certificate.DomainValidationOptions[*].[DomainName,ValidationStatus]' \
        --output text)
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Certificate Status: $STATUS"
    echo "$DOMAIN_STATUSES" | while IFS=$'\t' read -r domain status; do
        echo "  - $domain: $status"
    done
    echo ""
    
    if [ "$STATUS" = "ISSUED" ]; then
        echo "✅ Certificate has been issued successfully!"
        break
    fi
    
    sleep 30
done