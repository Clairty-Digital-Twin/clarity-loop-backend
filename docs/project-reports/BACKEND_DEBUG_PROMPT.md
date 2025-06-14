# 🚨 BACKEND DEBUGGING REQUEST

## Context
We've verified AWS ALB correctly routes to your FastAPI backend (IP: 172.31.10.190:8000). The frontend is receiving a BadRequest error that appears to be from AWS Cognito with message: **"The server did not understand the operation that was requested."**

AWS infrastructure has been ruled out - ALB is routing correctly to the healthy backend instance.

## Frontend Request Details

### Endpoint
```
POST http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/api/v1/auth/login
```

### Headers
```
Content-Type: application/json
```

### JSON Payload Structure
```json
{
  "email": "user@example.com",
  "password": "UserPassword123!",
  "remember_me": true,
  "device_info": {
    "device_id": "iPhone-123",
    "os_version": "iOS 18.0",
    "app_version": "1.0.0"
  }
}
```

### Frontend Code Reference
The iOS app uses:
- `BackendContractAdapter.adaptLoginRequest()` to convert frontend DTO to backend format
- JSON encoder with `keyEncodingStrategy = .convertToSnakeCase`
- All fields are properly snake_cased in the final JSON

## What We Need From Backend

1. **FastAPI Server Logs**
   - Do you see the login request arriving at FastAPI?
   - What's the exact request body FastAPI receives?
   - Any validation errors from Pydantic models?

2. **Cognito Integration Check**
   ```python
   # Check these in your FastAPI auth router:
   - Is COGNITO_CLIENT_ID properly set in environment?
   - Is the Cognito client configured with or without secret?
   - Are you using the correct Cognito pool ID?
   ```

3. **Error Source Identification**
   - Is the error coming from FastAPI validation?
   - Is it from boto3 Cognito client?
   - Check if this error appears in your logs:
   ```
   ClientError: An error occurred (InvalidParameterException) when calling the InitiateAuth operation
   ```

4. **Backend Auth Flow**
   Please trace through:
   ```python
   # In your auth router
   @router.post("/login")
   async def login(request: UserLoginRequest):
       # What's the exact structure of UserLoginRequest?
       # How are you calling Cognito's initiate_auth?
       # Are you handling device_info correctly?
   ```

5. **Quick Test**
   Can you try this direct boto3 test:
   ```python
   import boto3
   
   client = boto3.client('cognito-idp', region_name='your-region')
   
   response = client.initiate_auth(
       ClientId='your-client-id',
       AuthFlow='USER_PASSWORD_AUTH',
       AuthParameters={
           'USERNAME': 'test@example.com',
           'PASSWORD': 'TestPassword123!'
       }
   )
   ```

## Specific Questions

1. Does your Cognito app client have "Generate client secret" enabled? (This would require SECRET_HASH)
2. Is USER_PASSWORD_AUTH flow enabled in your Cognito app client?
3. Are you transforming the frontend's snake_case JSON back to match your Pydantic models?
4. What's the exact Pydantic model definition for your login endpoint?

## Error Pattern
The specific error "The server did not understand the operation that was requested" suggests:
- Either FastAPI is passing malformed data to Cognito
- Or Cognito client configuration mismatch (secret hash, auth flows)
- Or the request isn't reaching FastAPI at all (but we ruled out ALB issues)

Please investigate and provide backend logs + Cognito configuration details!