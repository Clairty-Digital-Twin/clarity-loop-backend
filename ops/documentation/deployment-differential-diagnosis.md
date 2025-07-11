# Deployment Failure Differential Diagnosis

## Chief Complaint
ECS deployments consistently fail with "Waiter ServicesStable failed: Max attempts exceeded" after 10 minutes.

## History of Present Illness
- Last successful deployment: June 21, 2025
- All subsequent deployments timeout at `aws ecs wait services-stable` (line 215 of deploy.sh)
- Service shows 0 running tasks despite healthy container logs
- Multiple overlapping deployments remain in IN_PROGRESS state

## Differential Diagnosis (in order of likelihood)

### 1. **ALB Health Check Misconfiguration** (Most Likely)
**Pathophysiology**: ECS waits for ALB to report healthy targets before considering deployment stable
**Evidence**: 
- Container logs show healthy responses (200 OK on /health)
- But ALB shows targets as "draining" or "unhealthy"
- Service can't stabilize without healthy ALB targets

### 2. **Task Role Permission Insufficiency**
**Pathophysiology**: Tasks start but can't pull secrets or access required AWS resources
**Evidence**:
- S3 bucket has explicit deny for non-task-role access
- Tasks might be failing to access Cognito, DynamoDB, or S3

### 3. **Network Configuration Issue**
**Pathophysiology**: Tasks can't communicate with ALB or AWS services
**Evidence**:
- Public IP assignment enabled but might have routing issues
- Security group allows inbound 8000 but might block outbound

### 4. **Resource Constraints**
**Pathophysiology**: Insufficient CPU/memory causing container crashes
**Evidence**:
- Task definition: 1024 CPU, 3072 memory
- No explicit container-level limits set

### 5. **Deployment Configuration Bug**
**Pathophysiology**: minimumHealthyPercent was 0, causing cascading failures
**Evidence**:
- Already fixed to 50% but damage may persist
- Multiple stuck deployments competing for resources

## Diagnostic Plan

1. **Check ALB Target Health**
   ```bash
   aws elbv2 describe-target-health --target-group-arn <ARN>
   ```

2. **Examine Task Stop Reasons**
   ```bash
   aws ecs describe-tasks --cluster clarity-backend-cluster --tasks <stopped-tasks>
   ```

3. **Review CloudWatch Logs**
   ```bash
   aws logs tail /ecs/clarity-backend --follow
   ```

4. **Verify Task Role Permissions**
   ```bash
   aws iam simulate-principal-policy --policy-source-arn <task-role-arn>
   ```

5. **Test Network Connectivity**
   - Check security group rules
   - Verify subnet routing tables
   - Test NAT gateway if private subnets

## Treatment Plan

### Immediate Interventions
1. Stop all existing deployments
2. Reset service to known good state
3. Enable ECS deployment circuit breaker
4. Deploy with comprehensive monitoring

### Long-term Management
1. Implement deployment circuit breaker
2. Add CloudWatch alarms for deployment failures
3. Use blue/green deployments
4. Add comprehensive health checks beyond /health endpoint

## Prognosis
Good - with proper diagnosis and intervention, deployments should stabilize within 30 minutes.