#!/bin/bash
# Check redirect source: ALB listener vs WAF

echo "🔍 Investigating 301 Redirect Source"
echo "===================================="

# Test with curl verbose to see headers
echo "Testing HTTP request with verbose output:"
curl -v "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health" 2>&1 | grep -E "(> |< |Location:|Server:)"

echo ""
echo "Checking ALB listeners for redirect rules..."

# Save listener info to file to avoid shell parsing issues
aws elbv2 describe-listeners \
    --load-balancer-arn "arn:aws:elasticloadbalancing:us-east-1:124355672559:loadbalancer/app/clarity-alb/fe7fa83dabc9ef21" \
    --region us-east-1 > /tmp/listeners.json

echo "Listeners found:"
grep -o '"Port":[0-9]*' /tmp/listeners.json || echo "No port info found"

# Check for redirect actions
if grep -q '"Type":"redirect"' /tmp/listeners.json; then
    echo "✅ FOUND: Redirect action in ALB listener configuration"
    echo "   This means the 301 is from ALB HTTP→HTTPS redirect (EXPECTED)"
else
    echo "❓ No explicit redirect action found in listeners"
    echo "   The 301 might be from target groups or WAF default action"
fi

# Check WAF default action
echo ""
echo "Checking WAF default action..."
aws wafv2 get-web-acl \
    --id "c690043c-9688-44fe-adb2-9883ccd8776b" \
    --scope REGIONAL \
    --region us-east-1 > /tmp/waf.json

if grep -q '"Action":"ALLOW"' /tmp/waf.json; then
    echo "✅ WAF default action is ALLOW (not causing redirects)"
elif grep -q '"Action":"BLOCK"' /tmp/waf.json; then
    echo "⚠️  WAF default action is BLOCK"
else
    echo "❓ WAF default action unclear"
fi

echo ""
echo "🔍 CONCLUSION:"
echo "If ALB has redirect rules → 301 is expected security hardening"
echo "If no ALB redirect found → investigate target group health/config"

# Cleanup
rm -f /tmp/listeners.json /tmp/waf.json 