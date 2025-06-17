# Admin Quick Reference - Security Operations

## 🚨 EMERGENCY CONTACTS
- **Security Team**: [Contact Info]
- **DevOps Team**: [Contact Info] 
- **On-Call Engineer**: [Contact Info]

## 🔐 DAILY SECURITY CHECKS

### Morning Checklist (5 minutes)
```bash
# 1. Check WAF status
aws wafv2 get-web-acl --scope=REGIONAL --id=clarity-backend-rate-limiting --region=us-east-1

# 2. Verify ALB health
aws elbv2 describe-load-balancers --names clarity-alb --region=us-east-1

# 3. Check security headers
curl -I https://$(aws elbv2 describe-load-balancers --names clarity-alb --query 'LoadBalancers[0].DNSName' --output text)/health
```

## 👤 USER ACCOUNT MANAGEMENT

### Create New User (2 minutes)
```bash
# Method 1: AWS CLI (Recommended)
aws cognito-idp admin-create-user \
  --user-pool-id us-east-1_XXXXXXXXX \
  --username user@example.com \
  --user-attributes Name=email,Value=user@example.com \
  --temporary-password TempPass123! \
  --message-action SUPPRESS \
  --region us-east-1
```

### Reset User Password
```bash
aws cognito-idp admin-set-user-password \
  --user-pool-id us-east-1_XXXXXXXXX \
  --username user@example.com \
  --password NewPassword123! \
  --permanent \
  --region us-east-1
```

## 🛡️ SECURITY MONITORING

### Check Rate Limiting (Real-time)
```bash
# View recent WAF logs
aws logs tail /aws/wafv2/clarity-backend-rate-limiting --follow --region us-east-1

# Count blocked requests in last hour
aws logs filter-log-events \
  --log-group-name /aws/wafv2/clarity-backend-rate-limiting \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --filter-pattern "BLOCK" \
  --region us-east-1
```

### Security Headers Verification
```bash
# Quick header check
curl -s -I https://your-alb-dns/health | grep -E "(X-Content-Type|X-Frame|X-XSS|Strict-Transport|Content-Security|Referrer-Policy)"
```

## 🚨 INCIDENT RESPONSE

### High Traffic Alert Response (< 2 minutes)
1. **Assess**: `aws logs tail /aws/wafv2/clarity-backend-rate-limiting --follow`
2. **Check**: Source IPs and patterns
3. **Decide**: Legitimate traffic spike or attack?
4. **Act**: Lower rate limit if attack detected
5. **Document**: Log incident details

### Rate Limit Adjustment (Emergency)
```bash
# Temporarily lower to 50 req/5min
sed -i 's/"Limit": 100/"Limit": 50/' ops/aws-waf-rate-limiting.json
bash ops/deploy-waf-rate-limiting.sh
```

## 📊 KEY METRICS TO WATCH

- **403/429 Responses**: > 100/hour = investigate
- **Failed Logins**: > 50/hour = potential attack  
- **WAF Blocks**: > 500/hour = likely DDoS
- **ALB 5xx Errors**: > 10/hour = backend issues

## 🔧 TROUBLESHOOTING

### "Users can't access the app"
1. Check ALB status
2. Verify WAF isn't over-blocking
3. Test auth endpoint
4. Check Cognito user pool

### "Rate limiting seems broken"
1. Verify WAF association: `bash ops/test-waf-final.sh`
2. Check CloudWatch logs
3. Test with burst requests
4. Validate ALB DNS resolution

## 📱 MOBILE/API QUICK TESTS

```bash
# Test health endpoint
curl -v https://your-alb-dns/health

# Test rate limiting (should get 429 after 100 requests)
for i in {1..105}; do curl -s -o /dev/null -w "%{http_code}\n" https://your-alb-dns/health; done

# Test security headers
curl -I https://your-alb-dns/health | head -20
```

---
*Keep this handy for daily operations!* 🚀