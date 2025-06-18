#!/bin/bash
# Quick deployment status checker

echo "🔍 DEPLOYMENT STATUS CHECK"
echo "========================="
echo ""

# Check GitHub Actions
echo "📊 GitHub Actions Status:"
gh run list --workflow "Deploy to AWS ECS" --limit 1

echo ""
echo "🌐 Production Health Check:"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://clarity.novamindnyc.com/health)
if [ "$STATUS" = "200" ]; then
    echo "✅ Production is UP! (Status: $STATUS)"
    echo ""
    echo "📋 Health Response:"
    curl -s https://clarity.novamindnyc.com/health | jq .
else
    echo "❌ Production is DOWN (Status: $STATUS)"
fi

echo ""
echo "🎯 Next Steps:"
if [ "$STATUS" = "200" ]; then
    echo "1. Production is working! 🎉"
    echo "2. Ready to merge additional features"
    echo "3. Consider merging PR #16 (Observability)"
else
    echo "1. Deployment still in progress..."
    echo "2. Wait a few more minutes"
    echo "3. Check CloudWatch logs if it fails"
fi