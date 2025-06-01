# API Endpoints

## Authentication

All endpoints require Firebase Authentication. Include the ID token in the Authorization header:

```
Authorization: Bearer <firebase_id_token>
```

## Health Data Upload

### POST /healthkit/upload

Uploads HealthKit data from iOS devices.

**Request:**
- Content-Type: application/json
- Body: JSON payload containing HealthKit samples

**Response:**
- 200 OK: Upload successful
- 401 Unauthorized: Invalid authentication
- 400 Bad Request: Invalid data format

## Analysis Results

### GET /analysis/result

Retrieves the latest analysis results for the authenticated user.

**Request:**
- No body required, user identified via authentication token

**Response:**
- 200 OK: Results found
- 404 Not Found: No results available
- 401 Unauthorized: Invalid authentication
