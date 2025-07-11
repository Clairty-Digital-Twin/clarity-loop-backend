# IAM Least-Privilege Implementation Runbook

**Version:** 1.0  
**Last Updated:** December 17, 2024  
**Risk Level:** HIGH - Requires careful execution

## Pre-Implementation Checklist

- [ ] Backup current IAM policies
- [ ] Verify staging environment is ready
- [ ] CloudTrail logging is enabled
- [ ] Monitoring dashboards are ready
- [ ] Rollback scripts are prepared
- [ ] Team is on standby

## Phase 1: Backup Current Configuration

### Step 1.1: Export Current Policies
```bash
# Create backup directory
mkdir -p ~/clarity-iam-backup-$(date +%Y%m%d)
cd ~/clarity-iam-backup-$(date +%Y%m%d)

# Backup ecsTaskExecutionRole
aws iam get-role --role-name ecsTaskExecutionRole > ecsTaskExecutionRole-backup.json
aws iam list-attached-role-policies --role-name ecsTaskExecutionRole > ecsTaskExecutionRole-policies.json
aws iam get-role-policy --role-name ecsTaskExecutionRole --policy-name ClaritySecretsAccess > ClaritySecretsAccess-backup.json
aws iam get-role-policy --role-name ecsTaskExecutionRole --policy-name SecretsManagerAccess > SecretsManagerAccess-backup.json

# Backup clarity-backend-task-role
aws iam get-role --role-name clarity-backend-task-role > clarity-backend-task-role-backup.json
aws iam list-attached-role-policies --role-name clarity-backend-task-role > clarity-backend-task-role-policies.json
```

### Step 1.2: Document Current State
```bash
# Generate report
echo "IAM Backup Report - $(date)" > backup-report.txt
echo "=========================" >> backup-report.txt
echo "" >> backup-report.txt
echo "Roles Backed Up:" >> backup-report.txt
ls -la *.json >> backup-report.txt
```

## Phase 2: Staging Environment Testing

### Step 2.1: Create Test Roles
```bash
# Create test execution role
aws iam create-role \
  --role-name ecsTaskExecutionRole-staging \
  --assume-role-policy-document file://trust-policy-ecs.json

# Create test task role
aws iam create-role \
  --role-name clarity-backend-task-role-staging \
  --assume-role-policy-document file://trust-policy-task.json
```

### Step 2.2: Apply Least-Privilege Policies
```bash
# Apply to staging execution role
aws iam put-role-policy \
  --role-name ecsTaskExecutionRole-staging \
  --policy-name ClarityECSExecutionPolicy \
  --policy-document file://ecs-execution-policy.json

# Apply to staging task role
aws iam put-role-policy \
  --role-name clarity-backend-task-role-staging \
  --policy-name ClarityApplicationPolicy \
  --policy-document file://application-policy.json
```

### Step 2.3: Deploy to Staging
```bash
# Update staging task definition
aws ecs register-task-definition \
  --family clarity-backend-staging \
  --execution-role-arn arn:aws:iam::124355672559:role/ecsTaskExecutionRole-staging \
  --task-role-arn arn:aws:iam::124355672559:role/clarity-backend-task-role-staging \
  --container-definitions file://container-def.json

# Update staging service
aws ecs update-service \
  --cluster clarity-staging-cluster \
  --service clarity-backend-staging \
  --task-definition clarity-backend-staging:latest
```

## Phase 3: Testing Protocol

### Step 3.1: Functional Tests
```bash
# Run automated test suite
cd /path/to/clarity-backend
npm run test:integration

# Specific permission tests
npm run test:iam-permissions
```

### Step 3.2: Manual Verification Checklist
- [ ] User can register/login
- [ ] Health data upload works
- [ ] ML models load correctly
- [ ] Data retrieval functions
- [ ] Logs appear in CloudWatch
- [ ] Metrics are published

### Step 3.3: Monitor for Permission Errors
```bash
# Watch CloudTrail for access denied
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=AccessDenied \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --region us-east-1
```

## Phase 4: Production Implementation

### Step 4.1: Create New Inline Policies
```bash
# Remove old policies from ecsTaskExecutionRole
aws iam delete-role-policy --role-name ecsTaskExecutionRole --policy-name ClaritySecretsAccess
aws iam delete-role-policy --role-name ecsTaskExecutionRole --policy-name SecretsManagerAccess

# Add new consolidated policy
aws iam put-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name ClarityECSExecutionPolicy \
  --policy-document file://ecs-execution-policy.json

# Detach AWS managed policies
aws iam detach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### Step 4.2: Update Task Role
```bash
# Detach overly permissive policies
aws iam detach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonCognitoPowerUser
aws iam detach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam detach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Add least-privilege policy
aws iam put-role-policy \
  --role-name clarity-backend-task-role \
  --policy-name ClarityApplicationPolicy \
  --policy-document file://application-policy.json
```

### Step 4.3: Deploy Changes
```bash
# Force new deployment to pick up IAM changes
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --force-new-deployment
```

## Phase 5: Validation & Monitoring

### Step 5.1: Real-time Monitoring
```bash
# Watch service events
aws ecs describe-services \
  --cluster clarity-backend-cluster \
  --services clarity-backend-service \
  --query 'services[0].events[0:10]'

# Monitor task health
watch -n 5 'aws ecs list-tasks --cluster clarity-backend-cluster --service-name clarity-backend-service'
```

### Step 5.2: CloudWatch Dashboard
Create alarms for:
- ECS task failures
- Permission denied errors
- Application error rates
- Response time degradation

### Step 5.3: User Acceptance Testing
- [ ] Test user registration flow
- [ ] Test data upload/download
- [ ] Verify AI insights generation
- [ ] Check admin functions

## Phase 6: Rollback Procedure

### If Issues Occur:
```bash
# Revert execution role
aws iam delete-role-policy --role-name ecsTaskExecutionRole --policy-name ClarityECSExecutionPolicy
aws iam put-role-policy --role-name ecsTaskExecutionRole --policy-name ClaritySecretsAccess --policy-document file://ClaritySecretsAccess-backup.json
aws iam put-role-policy --role-name ecsTaskExecutionRole --policy-name SecretsManagerAccess --policy-document file://SecretsManagerAccess-backup.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Revert task role
aws iam delete-role-policy --role-name clarity-backend-task-role --policy-name ClarityApplicationPolicy
aws iam attach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonCognitoPowerUser
aws iam attach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam attach-role-policy --role-name clarity-backend-task-role --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

# Force service update
aws ecs update-service --cluster clarity-backend-cluster --service clarity-backend-service --force-new-deployment
```

## Post-Implementation

### Documentation Updates
- [ ] Update architecture diagrams
- [ ] Update security documentation
- [ ] Update runbooks
- [ ] Schedule quarterly reviews

### Monitoring Setup
- [ ] Create IAM usage dashboard
- [ ] Set up weekly permission reports
- [ ] Configure anomaly detection
- [ ] Schedule access reviews

## Success Criteria

- [ ] All functional tests pass
- [ ] No increase in error rates
- [ ] No performance degradation
- [ ] CloudTrail shows no unauthorized access attempts
- [ ] All compliance requirements met

## Emergency Contacts

- **DevOps Lead:** [Contact]
- **Security Team:** [Contact]
- **Application Owner:** [Contact]
- **AWS Support:** [Case Number]