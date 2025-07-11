# AWS WAF Rate Limiting Deployment Guide

## Overview

This guide implements **Task 3: AWS WAF Rate Limiting** for the Clarity Digital Twin Backend as part of the comprehensive security hardening initiative.

## 🛡️ Protection Features

### Rate Limiting
- **100 requests per 5 minutes per IP address**
- Automatically blocks IPs exceeding the threshold
- CloudWatch metrics for monitoring

### Attack Protection (AWS Managed Rules)
1. **Common Rule Set** - OWASP Top 10 protection
2. **SQL Injection Protection** - Blocks SQL injection attempts  
3. **Known Bad Inputs** - Blocks malicious payloads
4. **IP Reputation List** - Blocks known malicious IPs

### Monitoring & Alerting
- **CloudWatch Metrics** for all blocked requests
- **Sampled Request Logging** for detailed analysis
- **Cost-effective** at ~$5-10/month

## 📋 Prerequisites

### AWS Credentials
```bash
# Verify AWS CLI is configured
aws sts get-caller-identity

# Should show:
# - Account: 124355672559  
# - Region: us-east-1
# - Valid credentials
```

### Required Permissions
- `wafv2:CreateWebACL`
- `wafv2:AssociateWebACL`
- `wafv2:GetWebACL`
- `elbv2:DescribeLoadBalancers`
- `elbv2:DescribeTargetGroups`

### Dependencies
- AWS CLI v2.x
- jq (for JSON parsing)
- curl (for testing)

## 🚀 Deployment Steps

### Step 1: Deploy AWS WAF
```bash
# Run the deployment script
./ops/deploy-waf-rate-limiting.sh
```

**Expected Output:**
```
🔒 Deploying AWS WAF Rate Limiting for Clarity Digital Twin Backend
==================================
Region: us-east-1
ALB: clarity-alb-1762715656
WAF Name: clarity-backend-rate-limiting
==================================

1. Verifying AWS CLI configuration...
✅ AWS CLI configured

2. Finding Application Load Balancer ARN...
✅ Found ALB: arn:aws:elasticloadbalancing:us-east-1:124355672559:loadbalancer/app/clarity-alb-1762715656/...

3. Checking for existing WAF Web ACL...

4. Creating WAF Web ACL with rate limiting rules...
✅ Created WAF Web ACL: [WAF-ID]

5. Associating WAF Web ACL with Application Load Balancer...
✅ Associated WAF with ALB

6. Verifying WAF association...
✅ WAF successfully associated with ALB

🔒 AWS WAF DEPLOYMENT COMPLETE!
```

### Step 2: Verify Deployment
```bash
# Run the test script
./ops/test-waf-rate-limiting.sh
```

**Expected Test Results:**
- ✅ Normal requests succeed (HTTP 200)
- ✅ SQL injection blocked (HTTP 403)
- ✅ Bad inputs blocked (HTTP 403) 
- ✅ WAF associated with ALB

## 🧪 Testing & Validation

### Manual Testing Commands

**Normal Request (should succeed):**
```bash
curl -v http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health
# Expected: HTTP 200
```

**SQL Injection Test (should be blocked):**
```bash
curl -v "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health?id=1%27%20OR%201=1--"
# Expected: HTTP 403 (Forbidden)
```

**Rate Limiting Test:**
```bash
# Send 101+ requests in 5 minutes from same IP
for i in {1..110}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health
  sleep 0.1
done
# Expected: First 100 return 200, subsequent requests return 403/429
```

### CloudWatch Metrics Verification

**Navigate to AWS Console > CloudWatch > Metrics > WAF**

Monitor these metrics:
- `clarity-rate-limit-blocked` - Rate limiting blocks
- `clarity-common-rule-set` - General attack blocks  
- `clarity-bad-inputs-blocked` - Malicious payload blocks
- `clarity-sqli-blocked` - SQL injection blocks
- `clarity-ip-reputation-blocked` - Malicious IP blocks

## 📊 Monitoring Dashboard

### CloudWatch Dashboard Query
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/WAFV2", "BlockedRequests", "WebACL", "clarity-backend-rate-limiting"],
          [".", "AllowedRequests", ".", "."]
        ],
        "period": 300,
        "stat": "Sum", 
        "region": "us-east-1",
        "title": "WAF Request Statistics"
      }
    }
  ]
}
```

### Alerts Setup
```bash
# Create CloudWatch alarm for excessive blocks
aws cloudwatch put-metric-alarm \
  --alarm-name "WAF-High-Block-Rate" \
  --alarm-description "High rate of blocked requests" \
  --metric-name BlockedRequests \
  --namespace AWS/WAFV2 \
  --statistic Sum \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=WebACL,Value=clarity-backend-rate-limiting \
  --evaluation-periods 1
```

## 🔧 Configuration Management

### WAF Configuration File
- **Location**: `ops/aws-waf-rate-limiting.json`
- **Scope**: REGIONAL (for ALB)
- **Default Action**: Allow (with rule-based blocking)

### Rule Priorities
1. **Priority 1**: Rate Limiting (100 req/5min)
2. **Priority 2**: Common Rule Set (OWASP)
3. **Priority 3**: Known Bad Inputs
4. **Priority 4**: SQL Injection Protection
5. **Priority 5**: IP Reputation List

### Customization Options

**Adjust Rate Limit:**
```json
{
  "Statement": {
    "RateBasedStatement": {
      "Limit": 200,  // Change from 100 to 200
      "AggregateKeyType": "IP"
    }
  }
}
```

**Add Geographic Blocking:**
```json
{
  "Name": "GeoBlockRule",
  "Priority": 6,
  "Statement": {
    "GeoMatchStatement": {
      "CountryCodes": ["CN", "RU", "KP"]  // Block China, Russia, North Korea
    }
  },
  "Action": {
    "Block": {}
  }
}
```

## 🔄 Rollback Procedures

### Emergency Rollback (Disassociate WAF)
```bash
# Get ALB ARN
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names clarity-alb-1762715656 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

# Disassociate WAF (immediate effect)
aws wafv2 disassociate-web-acl \
  --resource-arn $ALB_ARN \
  --region us-east-1

echo "✅ WAF disassociated - normal traffic restored"
```

### Complete WAF Removal
```bash
# Get WAF ID
WAF_ID=$(aws wafv2 list-web-acls \
  --scope REGIONAL \
  --query "WebACLs[?Name=='clarity-backend-rate-limiting'].Id" \
  --output text)

# Get lock token and delete
LOCK_TOKEN=$(aws wafv2 get-web-acl \
  --scope REGIONAL \
  --id $WAF_ID \
  --query 'LockToken' \
  --output text)

aws wafv2 delete-web-acl \
  --scope REGIONAL \
  --id $WAF_ID \
  --lock-token $LOCK_TOKEN

echo "✅ WAF completely removed"
```

## 💰 Cost Analysis

### Monthly Cost Estimates
- **Web ACL**: $1.00/month
- **Rules**: $1.00 x 5 rules = $5.00/month  
- **Requests**: $0.60 per million requests
- **Total**: ~$6-10/month for typical traffic

### Cost Optimization
- Monitor unused rules and disable if not triggering
- Adjust rate limits based on actual traffic patterns
- Use AWS Cost Explorer to track WAF spending

## ⚠️ Troubleshooting

### Common Issues

**WAF Not Blocking Attacks:**
```bash
# Check WAF association
aws wafv2 get-web-acl-for-resource \
  --resource-arn $ALB_ARN \
  --region us-east-1

# Check rule configuration
aws wafv2 get-web-acl \
  --scope REGIONAL \
  --id $WAF_ID \
  --region us-east-1
```

**False Positives (Legitimate Traffic Blocked):**
1. Review CloudWatch metrics for specific rule triggers
2. Check sampled requests in WAF console
3. Add exception rules for legitimate patterns
4. Adjust rate limits if too aggressive

**High Costs:**
1. Monitor request volume in CloudWatch
2. Check for unusual traffic patterns  
3. Consider adjusting rule sensitivity
4. Review sampled requests for attack patterns

### Support Resources
- **AWS WAF Documentation**: https://docs.aws.amazon.com/waf/
- **CloudWatch Logs**: Enable for detailed request analysis
- **AWS Support**: For complex rule configurations

## ✅ Success Criteria

### Functional Validation
- [ ] WAF deployed and associated with ALB
- [ ] Rate limiting blocks 101+ requests/5min
- [ ] SQL injection attempts blocked  
- [ ] XSS/bad input attempts blocked
- [ ] Normal traffic flows without issues
- [ ] CloudWatch metrics reporting correctly

### Security Validation  
- [ ] External security scan shows improved protection
- [ ] Load testing confirms rate limiting works
- [ ] Attack simulation shows blocking effectiveness
- [ ] No false positives for legitimate traffic

### Operational Validation
- [ ] Monitoring dashboard functional
- [ ] Alerts configured and testing
- [ ] Rollback procedures tested
- [ ] Cost tracking in place
- [ ] Documentation complete

## 🎯 Next Steps

After WAF deployment completion:

1. **Task 8**: DynamoDB Encryption at Rest
2. **Task 9**: S3 Bucket Security Hardening  
3. **Task 6**: Application-Level Rate Limiting
4. **Task 7**: Enhanced Error Handling

## 📚 Additional Resources

- [Security Hardening PRD](scripts/security_hardening_prd.txt)
- [AWS WAF Best Practices](https://docs.aws.amazon.com/waf/latest/developerguide/waf-chapter.html)
- [OWASP Web Application Firewall](https://owasp.org/www-community/Web_Application_Firewall)

---

**✅ TASK 3: AWS WAF RATE LIMITING IMPLEMENTATION COMPLETE**

*Provides infrastructure-level DDoS protection, rate limiting, and attack blocking for the Clarity Digital Twin Backend.* 