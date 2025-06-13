# Frontend Findings Summary for Backend Investigation

## What We've Confirmed

### ✅ Frontend is sending valid JSON
```swift
// From BackendAPIClient.swift
encoder.keyEncodingStrategy = .convertToSnakeCase

// Actual JSON being sent:
{
  "email": "user@example.com",
  "password": "password",
  "remember_me": true,
  "device_info": {
    "device_id": "...",
    "os_version": "...",
    "app_version": "..."
  }
}
```

### ✅ Frontend contract adapter is correct
```swift
// BackendContractAdapter.swift
func adaptLoginRequest(_ frontendRequest: UserLoginRequestDTO) -> BackendUserLogin {
    return BackendUserLogin(
        email: frontendRequest.email,
        password: frontendRequest.password,
        rememberMe: frontendRequest.rememberMe,  // Encoded as "remember_me"
        deviceInfo: frontendRequest.deviceInfo    // Encoded as "device_info"
    )
}
```

### ✅ Correct endpoint and method
- URL: `/api/v1/auth/login`
- Method: POST
- Content-Type: application/json

### ✅ AWS ALB Configuration Verified
- ALB is correctly routing `/api/*` to backend target group
- Backend instance (172.31.10.190:8000) is healthy
- No Cognito authentication at ALB level for API paths

## The Error

### Error Response
```json
{
  "code": "BadRequest",
  "message": "The server did not understand the operation that was requested.",
  "type": "client"
}
```

### Error Characteristics
1. This is AWS Cognito's standard error format
2. HTTP 400 Bad Request
3. Suggests Cognito received malformed/unexpected input

## Likely Backend Issues

### 1. Cognito Client Secret Mismatch
If your Cognito app client has "Generate client secret" enabled, you need to calculate and send SECRET_HASH:
```python
def get_secret_hash(username, client_id, client_secret):
    message = username + client_id
    dig = hmac.new(client_secret.encode('UTF-8'), 
                   message.encode('UTF-8'),
                   hashlib.sha256).digest()
    return base64.b64encode(dig).decode()
```

### 2. Auth Flow Not Enabled
Check if USER_PASSWORD_AUTH is enabled in your Cognito app client settings.

### 3. Parameter Mismatch
Cognito expects specific parameter names:
- USERNAME (not email)
- PASSWORD
- Maybe the backend is passing "email" directly instead of mapping it to "USERNAME"

### 4. Missing Required Fields
Some Cognito configurations require additional fields like:
- CLIENT_METADATA
- DEVICE_KEY
- SECRET_HASH

## Recommended Backend Debug Steps

1. **Log the exact boto3 call**:
   ```python
   print(f"Calling initiate_auth with: {auth_parameters}")
   ```

2. **Check Cognito client configuration**:
   ```bash
   aws cognito-idp describe-user-pool-client \
     --user-pool-id YOUR_POOL_ID \
     --client-id YOUR_CLIENT_ID
   ```

3. **Try a minimal test**:
   ```python
   # Direct Cognito test bypassing FastAPI
   response = cognito_client.initiate_auth(
       ClientId=CLIENT_ID,
       AuthFlow='USER_PASSWORD_AUTH',
       AuthParameters={
           'USERNAME': 'test@example.com',
           'PASSWORD': 'TestPass123!'
       }
   )
   ```

This should help isolate whether the issue is in FastAPI request handling or Cognito configuration.