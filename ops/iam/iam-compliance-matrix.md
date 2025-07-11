# IAM Compliance Matrix - CLARITY Platform

**Last Updated:** December 17, 2024  
**Review Frequency:** Quarterly  
**Next Review:** March 17, 2025

## Permission Matrix

### ecsTaskExecutionRole

| Service | Action | Resource | Business Justification | Risk Level |
|---------|--------|----------|----------------------|------------|
| ECR | GetAuthorizationToken | * (Regional) | Pull container images | LOW |
| ECR | BatchCheckLayerAvailability, GetDownloadUrlForLayer, BatchGetImage | clarity-backend repo | Access application image | LOW |
| Logs | CreateLogStream, PutLogEvents | /ecs/clarity-backend/* | Application logging | LOW |
| Secrets | GetSecretValue | clarity/gemini-api-key, clarity/cognito-config | Access API credentials | MEDIUM |

### clarity-backend-task-role

| Service | Action | Resource | Business Justification | Risk Level |
|---------|--------|----------|----------------------|------------|
| DynamoDB | Read/Write operations | clarity-health-data table | Store/retrieve health data | HIGH |
| S3 | GetObject, PutObject, DeleteObject | clarity-health-data-storage/* | Health data file storage | HIGH |
| S3 | GetObject | clarity-ml-models-124355672559/* | Load ML models | MEDIUM |
| Cognito | User management | us-east-1_efXaR5EcP pool | User authentication | HIGH |
| CloudWatch | PutMetricData | Clarity/* namespaces | Application metrics | LOW |
| KMS | Encrypt/Decrypt | DynamoDB encryption key | Data encryption | HIGH |

## Compliance Mappings

### HIPAA Requirements

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Minimum Necessary | Resource-specific permissions | No wildcards, specific ARNs |
| Access Controls | Role-based access | Separate execution/task roles |
| Encryption | KMS integration | Conditional encryption checks |
| Audit Logging | CloudTrail enabled | All IAM actions logged |

### SOC2 Controls

| Control | Implementation | Testing Frequency |
|---------|----------------|-------------------|
| CC6.1 - Logical Access | Least-privilege IAM | Quarterly |
| CC6.2 - New Access | Documented approval process | Per change |
| CC6.3 - Access Modification | Change management process | Per change |
| CC7.2 - Monitoring | CloudWatch alarms | Continuous |

### ISO 27001 Compliance

| Control | Status | Evidence |
|---------|--------|----------|
| A.9.1.2 - Access to networks | ✅ Compliant | VPC/Security groups |
| A.9.2.3 - Management of privileged access | ✅ Compliant | Least-privilege roles |
| A.9.4.1 - Information access restriction | ✅ Compliant | Resource-level permissions |
| A.12.4.1 - Event logging | ✅ Compliant | CloudTrail/CloudWatch |

## Monitoring & Alerting

### CloudWatch Alarms

| Alarm Name | Metric | Threshold | Action |
|------------|--------|-----------|--------|
| IAM-AccessDenied | AccessDenied events | > 5 in 5 min | Page on-call |
| IAM-RoleAssumption | AssumeRole calls | > 10 in 1 min | Email security |
| IAM-PolicyChange | IAM policy modifications | Any | Audit log + email |

### Regular Reviews

| Review Type | Frequency | Owner | Last Completed |
|-------------|-----------|-------|----------------|
| Permission audit | Quarterly | Security Team | Dec 17, 2024 |
| Unused permissions | Monthly | DevOps | Pending |
| External access | Weekly | Security Team | Automated |
| Compliance check | Quarterly | Compliance Officer | Dec 17, 2024 |

## Emergency Procedures

### Break-Glass Access
1. **Scenario:** Application completely locked out
2. **Procedure:** 
   - Use root account (MFA required)
   - Apply emergency-iam-restore.json
   - Investigate root cause
   - Revert to least-privilege within 24 hours
3. **Approval:** CTO or Security Lead required

### Incident Response
1. **Detection:** CloudWatch alarm or user report
2. **Triage:** Determine permission gap
3. **Temporary Fix:** Grant minimal additional permission
4. **Root Cause:** Analyze why permission was needed
5. **Permanent Fix:** Update least-privilege policy
6. **Documentation:** Update this matrix

## Approval Matrix

| Change Type | Approver | SLA |
|-------------|----------|-----|
| Add new permission | Security Team | 1 business day |
| Remove permission | DevOps Lead | 2 hours |
| Emergency access | CTO/Security Lead | 30 minutes |
| New role creation | Security Team + App Owner | 2 business days |

## Audit Trail

| Date | Change | Approver | Ticket |
|------|--------|----------|--------|
| 2024-12-17 | Initial least-privilege implementation | Security Team | SEC-001 |
| - | - | - | - |

## Appendix: Quick Reference

### Test Commands
```bash
# Verify role permissions
./ops/test-iam-permissions.sh

# Check current role configuration
aws iam get-role --role-name [ROLE_NAME]

# List policies
aws iam list-role-policies --role-name [ROLE_NAME]
aws iam list-attached-role-policies --role-name [ROLE_NAME]
```

### Rollback Commands
```bash
# Emergency rollback
./ops/emergency-iam-restore.sh

# Revert specific role
aws iam put-role-policy --role-name [ROLE] --policy-name [POLICY] --policy-document file://backup/[POLICY].json
```