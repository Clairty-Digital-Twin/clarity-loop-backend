# Authentication API

**UPDATED:** December 6, 2025 - Based on actual implementation in `src/clarity/api/v1/auth.py`

## Overview

CLARITY uses Firebase Authentication for secure user management. The API provides endpoints for user registration and login.

## Endpoints

### Register User

```http
POST /api/v1/auth/register
Content-Type: application/json
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (201 Created):**

```json
{
  "message": "User registered successfully",
  "user_id": "firebase-uid-abc123"
}
```

**Error Responses:**

- `400` - Invalid email format or weak password
- `409` - Email already exists
- `500` - Internal server error

### Login User

```http
POST /api/v1/auth/login
Content-Type: application/json
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": "firebase-uid-abc123",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

**Error Responses:**

- `401` - Invalid email/password
- `400` - Missing required fields
- `500` - Internal server error

## Using Authentication Tokens

Include the access token in protected endpoint requests:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Security Features

- **Firebase Integration**: Leverages Google's enterprise-grade authentication
- **JWT Tokens**: Stateless authentication with configurable expiration
- **Password Requirements**: Enforced complexity and length requirements
- **Rate Limiting**: Built-in protection against brute force attacks

## Example Usage

```python
import httpx

# Register new user
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/auth/register",
        json={
            "email": "user@example.com",
            "password": "securepassword123"
        }
    )
    print(f"Registration: {response.json()}")

# Login
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/auth/login",
        json={
            "email": "user@example.com",
            "password": "securepassword123"
        }
    )
    token = response.json()["access_token"]
    print(f"Token: {token}")
```

## Implementation Details

- **Location**: `src/clarity/api/v1/auth.py`
- **Firebase Admin**: Uses Firebase Admin SDK for user management
- **Authentication**: Firebase Auth with custom token generation
- **Dependencies**: Requires Firebase project configuration
