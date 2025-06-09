#!/bin/bash
# Deploy to Modal production environment

echo "🚀 Deploying to Modal production environment..."
echo "📦 Using optimized deployment with global app instance fix"

# Deploy to production
modal deploy --env prod modal_deploy_optimized.py

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Check logs: modal logs --env prod"
echo "2. Test auth: curl -H 'Authorization: Bearer YOUR_TOKEN' https://your-app.modal.run/api/v1/insights/status"