# GCP Credentials Setup for CLARITY Backend

This document explains how Google Cloud Platform (GCP) credentials are managed in the CLARITY backend, particularly for containerized deployments.

## Overview

The CLARITY backend uses Google Cloud services (e.g., Vertex AI) that require authentication via service account credentials. In production environments like AWS ECS, these credentials are stored securely in AWS Secrets Manager and injected as environment variables.

## Credential Flow

1. **Local Development**: The application looks for a service account JSON file in the project root
2. **Production (ECS)**: Credentials are stored in AWS Secrets Manager and injected as an environment variable
3. **Runtime**: The `GCPCredentialsManager` handles both scenarios transparently

## Setup Instructions

### 1. Store Credentials in AWS Secrets Manager

Run the provided script to store your service account JSON in AWS Secrets Manager:

```bash
./scripts/store-gcp-credentials.sh
```

This script will:
- Store the JSON in AWS Secrets Manager with the name `clarity/gcp-service-account`
- Move the local file to a secure location outside the git repository
- Provide the secret ARN for the ECS task definition

### 2. Update ECS Task Definition

The ECS task definition has been updated to include:

```json
{
  "name": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
  "valueFrom": "arn:aws:secretsmanager:us-east-1:124355672559:secret:clarity/gcp-service-account"
}
```

### 3. Application Integration

The application automatically handles GCP credentials through the `GCPCredentialsManager`:

```python
from clarity.services.gcp_credentials import initialize_gcp_credentials

# Called during application startup
initialize_gcp_credentials()
```

## How It Works

1. **At Startup**: The `GCPCredentialsManager` checks for credentials in this order:
   - Existing `GOOGLE_APPLICATION_CREDENTIALS` environment variable
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable (from AWS Secrets Manager)
   - Local service account files (for development)

2. **For ECS Deployments**: 
   - AWS Secrets Manager provides the JSON as an environment variable
   - The manager creates a temporary file with the credentials
   - Sets `GOOGLE_APPLICATION_CREDENTIALS` to point to this file
   - Google Cloud SDKs automatically use this file for authentication

3. **Cleanup**: Temporary credential files are cleaned up on application shutdown

## Security Notes

- Never commit service account JSON files to version control
- Service account files matching these patterns are gitignored:
  - `service-account*.json`
  - `firebase-admin*.json`
  - `gcp-credentials*.json`
- In production, credentials are only accessible to the ECS task via IAM roles
- The ECS task execution role must have permission to read the secret from Secrets Manager

## Troubleshooting

### Check if credentials are loaded:
```python
from clarity.services.gcp_credentials import get_gcp_credentials_manager

manager = get_gcp_credentials_manager()
print(f"Credentials path: {manager.get_credentials_path()}")
print(f"Project ID: {manager.get_project_id()}")
```

### Common Issues:

1. **"No GCP credentials found" warning**: 
   - Ensure the secret is properly configured in ECS task definition
   - Check that the ECS execution role has SecretManager read permissions

2. **"Invalid JSON" error**:
   - Verify the secret contains valid JSON
   - Check for any encoding issues

3. **Permission denied errors**:
   - Ensure the service account has necessary permissions in GCP
   - Verify the project ID matches your GCP project

## Required IAM Permissions

### AWS (for ECS Task Execution Role):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:124355672559:secret:clarity/gcp-service-account*"
    }
  ]
}
```

### GCP (for Service Account):
- Vertex AI User
- Storage Object Viewer (if accessing GCS buckets)
- Any other service-specific roles required by your application