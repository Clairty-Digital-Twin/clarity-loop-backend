#!/bin/bash
# 🚀 CLARITY DIGITAL TWIN - INSTANT DEMO LAUNCHER
# One command to rule them all for your 6 PM technical co-founder meeting

set -e

echo "🚀 STARTING CLARITY DIGITAL TWIN DEMO..."
echo "⏰ Perfect for your 6 PM technical co-founder meeting!"
echo ""

# Launch the full demo
bash scripts/demo_deployment.sh

echo ""
echo "🔥 DEMO IS LIVE! Key URLs:"
echo "📱 Main App: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
echo ""
echo "💡 DEMO COMMANDS TO RUN LIVE:"
echo "  curl http://localhost:8000/health"
echo "  python scripts/api_test_suite.py"
echo "  docker compose ps"
echo ""
echo "🏆 GO SHOCK THAT TECHNICAL CO-FOUNDER!" 