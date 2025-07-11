# IAM Permissions Required for GitHub Actions

## Current Issue
The `GitHubActionsDeploy` IAM role is missing permissions required for the Security Smoke Test workflow.

## Required Permissions to Add

Add the following permissions to the `GitHubActionsDeploy` IAM role:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "LoadBalancerReadAccess",
            "Effect": "Allow",
            "Action": [
                "elasticloadbalancing:DescribeLoadBalancers"
            ],
            "Resource": "*"
        },
        {
            "Sid": "WAFReadAccess",
            "Effect": "Allow",
            "Action": [
                "wafv2:GetWebACLForResource"
            ],
            "Resource": "*"
        }
    ]
}
```

## How to Fix

1. Go to AWS Console → IAM → Roles
2. Search for `GitHubActionsDeploy`
3. Add an inline policy with the above permissions
4. Or attach the AWS managed policy `ElasticLoadBalancingReadOnly` and create a custom policy for WAF

## Security Best Practices (2025)

Following the principle of least privilege:
- These are read-only permissions
- Consider restricting resources to specific ARNs if possible:
  - Load Balancer ARN: `arn:aws:elasticloadbalancing:us-east-1:{ACCOUNT_ID}:loadbalancer/app/clarity-alb/*`
  - WAF ACL ARN pattern: `arn:aws:wafv2:us-east-1:{ACCOUNT_ID}:*/webacl/*`

## Verification

After adding permissions, the Security Smoke Test workflow should:
1. Successfully describe the `clarity-alb` load balancer
2. Retrieve the associated WAF ACL
3. Verify it's named `clarity-backend-rate-limiting`
4. Test SQL injection blocking (expecting 403 response)