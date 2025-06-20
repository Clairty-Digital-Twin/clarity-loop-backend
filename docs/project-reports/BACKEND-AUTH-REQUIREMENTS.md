# 🎯 BACKEND AUTH REQUIREMENTS GUIDE

## Goal: Ensure backend is the single source of truth for auth

## ✅ TASKS CHECKLIST

### 1. Confirm `/api/v1/auth/login` Implementation

```python
# In src/clarity/api/v1/auth.py
@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    credentials: UserLoginRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """
    Validates user with Cognito via USER_PASSWORD_AUTH + SECRET_HASH.
    Returns standardized token response for mobile clients.
    """
    # Log for debugging
    logger.info(f"Login attempt for email: {credentials.email}")
    
    try:
        # Authenticate with Cognito (handles SECRET_HASH internally)
        tokens = await auth_provider.authenticate(
            email=credentials.email,
            password=credentials.password,
        )
        
        # Return standardized response
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="Bearer",
            expires_in=tokens.get("expires_in", 3600),
            scope="openid email profile"
        )
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise
```

### 2. Implement `/api/v1/auth/refresh` Endpoint

```python
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> TokenResponse:
    """
    Accepts refresh token and returns new access/refresh tokens.
    """
    try:
        # Use Cognito's refresh token flow
        client = auth_provider.cognito_client
        response = client.initiate_auth(
            ClientId=auth_provider.client_id,
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={
                "REFRESH_TOKEN": request.refresh_token,
            },
        )
        
        result = response["AuthenticationResult"]
        return TokenResponse(
            access_token=result["AccessToken"],
            refresh_token=request.refresh_token,  # Cognito doesn't rotate
            token_type="Bearer",
            expires_in=result.get("ExpiresIn", 3600),
            scope="openid email profile"
        )
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid refresh token")
```

### 3. Create Frontend Integration Documentation

```markdown
# docs/FRONTEND_INTEGRATION.md

## Authentication API Reference

### Login Endpoint
**POST** `/api/v1/auth/login`

Request:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "remember_me": true,
  "device_info": {
    "device_id": "unique-device-id",
    "os_version": "iOS 18.0",
    "app_version": "1.0.0",
    "platform": "iOS",
    "model": "iPhone 15",
    "name": "User's iPhone"
  }
}
```

Response:

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiw...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "openid email profile"
}
```

### Refresh Token Endpoint

**POST** `/api/v1/auth/refresh`

Request:

```json
{
  "refresh_token": "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiw..."
}
```

Response: Same as login endpoint

### Example cURL Commands

Login:

```bash
curl -X POST "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123!",
    "remember_me": true,
    "device_info": {
      "device_id": "test-device",
      "os_version": "iOS 18.0",
      "app_version": "1.0.0"
    }
  }'
```

Refresh:

```bash
curl -X POST "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your-refresh-token-here"
  }'
```

```

### 4. Optional: Implement Logout Endpoint
```python
@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    current_user: dict = Depends(get_current_user),
    auth_provider: IAuthProvider = Depends(get_auth_provider),
) -> MessageResponse:
    """
    Revokes refresh token to prevent further use.
    """
    try:
        # Get refresh token from request
        body = await request.json()
        refresh_token = body.get("refresh_token")
        
        if refresh_token and hasattr(auth_provider, 'revoke_token'):
            # Revoke the refresh token in Cognito
            await auth_provider.revoke_token(refresh_token)
        
        return MessageResponse(message="Successfully logged out")
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        # Still return success - client should discard tokens
        return MessageResponse(message="Logout processed")
```

### 5. Add Integration Tests

```python
# tests/test_auth_endpoints.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_login_success(async_client: AsyncClient):
    """Test successful login returns tokens."""
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "ValidPass123!",
            "remember_me": True,
            "device_info": {
                "device_id": "test-device",
                "os_version": "iOS 18.0",
                "app_version": "1.0.0"
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] > 0

@pytest.mark.asyncio
async def test_refresh_token_success(async_client: AsyncClient):
    """Test token refresh returns new tokens."""
    # First login to get refresh token
    login_response = await async_client.post("/api/v1/auth/login", json={...})
    refresh_token = login_response.json()["refresh_token"]
    
    # Test refresh
    response = await async_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
```

## 📋 VERIFICATION STEPS

1. **Test login endpoint**

   ```bash
   # Should return tokens
   curl -X POST http://localhost:8000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"TestPass123!","remember_me":true}'
   ```

2. **Verify Cognito SECRET_HASH handling**
   - Check logs show "Login attempt for email: xxx"
   - Confirm no SECRET_HASH errors in logs
   - Verify tokens are valid JWTs

3. **Test refresh flow**
   - Use refresh token from login
   - Verify new access token returned
   - Check expiry is reset

## ✅ DEFINITION OF DONE

- [ ] `/api/v1/auth/login` validates with Cognito + SECRET_HASH
- [ ] Returns standardized token response
- [ ] `/api/v1/auth/refresh` endpoint implemented
- [ ] Frontend integration docs created
- [ ] Example cURL commands provided
- [ ] Integration tests pass
- [ ] No Cognito secrets exposed to frontend

---

**BACKEND IS READY TO SUPPORT MOBILE AUTH! 🚀**
