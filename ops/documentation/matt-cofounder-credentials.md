# AWS Access for Matt Gorbett - Co-founder Evaluation

## Instructions
**IMPORTANT**: The actual credentials have been sent to you separately via secure channel.

## User Setup
- **Username**: matt-gorbett-cofounder
- **AWS Account**: 124355672559
- **Region**: us-east-1

## Permissions Granted
1. **ReadOnlyAccess** - Full read access to all AWS services
2. **ClarityS3ReadAccess** - Custom S3 access for Clarity buckets

## Configuration Steps
1. Install AWS CLI if not already installed
2. Run `aws configure` and enter the credentials sent separately
3. Set region to `us-east-1`
4. Set output format to `json`

## Test Commands
```bash
aws sts get-caller-identity
aws s3 ls
aws s3 ls s3://clarity-health-data-storage/
```

## Available Resources
- S3 Buckets: clarity-health-data-storage, clarity-ml-models-124355672559
- Full read-only access to evaluate infrastructure

## Security Note
These are temporary evaluation credentials. Access can be revoked at any time.