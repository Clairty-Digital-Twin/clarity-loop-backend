# 🚀 AGENT 3: AWS Cleanup & TaskMaster

## PRIMARY MISSION
Clean AWS resources, update TaskMaster, prepare for merge

## TASKS

### 1. Clean AWS Resources
```bash
# Delete old ECR images
aws ecr batch-delete-image \
  --repository-name clarity-backend \
  --region us-east-1 \
  --image-ids file://ecr-images-to-delete.json

# Stop ECS service (save money!)
aws ecs update-service \
  --cluster ***REMOVED*** \
  --service clarity-backend-service \
  --desired-count 0 \
  --region us-east-1

# List and clean old task definitions
aws ecs list-task-definitions --region us-east-1 | grep clarity
```

### 2. Update TaskMaster Status
```bash
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/clarity-loop-backend

# Mark completed tasks
tm set-status 3 done  # Cognito migration
tm set-status 4 done  # DynamoDB migration  
tm set-status 5 done  # WebSocket support
tm set-status 6 done  # All endpoints working

# Check remaining
tm get-tasks --status pending
```

### 3. Document Architecture Decisions
- Update ARCHITECTURE_DEBT.md with Vertex AI decision
- Document why we use Gemini API in AWS deployment
- Create migration completion report

### 4. Prepare Final Commit
```bash
# After other agents complete their work
git add -A
git commit -m "feat: Complete Firebase to AWS migration - all systems operational

- Migrated auth: Firebase → AWS Cognito
- Migrated database: Firestore → DynamoDB  
- Fixed all 60+ API endpoints
- Cleaned linting errors
- Updated all tests
- Removed Firebase references

Deployment: http://***REMOVED***"
```

## SUCCESS CRITERIA
- AWS resources cleaned (no ongoing charges)
- TaskMaster shows accurate completion status
- Ready to merge all agent branches