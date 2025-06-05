# User Management API

**UPDATED:** December 6, 2025 - Based on actual implementation and Firebase integration

## Overview

User management is handled through Firebase Authentication integration. The system provides secure user registration, authentication, and profile management for the CLARITY Digital Twin Platform.

## Authentication Flow

CLARITY uses Firebase Authentication for all user management operations:

1. **Registration/Login** → Firebase handles identity verification
2. **JWT Token** → Firebase issues secure JWT tokens
3. **API Access** → All API endpoints validate Firebase tokens
4. **User Context** → Firebase UID used for data isolation

## User Lifecycle

### Registration Process

Users register through the authentication endpoint:

```http
POST /api/v1/auth/register
```

**Automatic Setup:**

- Firebase user account created
- Unique user ID (Firebase UID) assigned
- User profile initialized in Firestore
- Health data storage configured
- Default preferences set

### Profile Management

User profiles are automatically managed by the system:

**Profile Data Stored:**

```json
{
  "user_id": "firebase-uid-123",
  "email": "user@example.com",
  "created_at": "2025-01-15T10:30:00Z",
  "last_login": "2025-01-15T10:30:00Z",
  "preferences": {
    "timezone": "America/New_York",
    "units": "metric",
    "notifications": true,
    "data_sharing": false
  },
  "health_profile": {
    "age_range": "25-34",
    "activity_level": "moderate",
    "goals": ["improve_sleep", "increase_activity"],
    "medical_conditions": [],
    "medications": []
  },
  "data_summary": {
    "total_uploads": 15,
    "last_upload": "2025-01-14T20:15:00Z",
    "insights_generated": 8,
    "data_completeness": 0.87
  }
}
```

### Data Access Control

**User Isolation:**

- Each user can only access their own data
- Firebase UID used for all database queries
- Strict authorization on all endpoints
- No cross-user data access possible

**API Authorization Pattern:**

```python
# Every protected endpoint validates user identity
async def get_user_health_data(
    user_token: Annotated[dict, Depends(verify_firebase_token)]
):
    user_id = user_token["uid"]
    # Only returns data for authenticated user
    return await health_data_service.get_user_data(user_id)
```

## System Integration

### Health Data Management

- **Upload Tracking**: Monitor health data uploads per user
- **Processing Status**: Track analysis job status
- **Data Quality**: Maintain data completeness metrics
- **Storage Management**: Organize user data in Cloud Storage

### AI Insights Integration

- **Personalization**: Generate insights specific to user
- **Context Awareness**: Use user goals and preferences
- **History Tracking**: Maintain insight generation history
- **Recommendation Engine**: Tailor recommendations to user profile

### Privacy & Security

**Data Protection:**

- All user data encrypted at rest and in transit
- HIPAA-ready infrastructure design
- Minimal data collection principle
- User data deletion capabilities

**Access Controls:**

- Firebase security rules enforce user isolation
- API rate limiting per user
- Audit logging for all data access
- Secure token validation on every request

## User Data Export/Deletion

### Data Export (Future)

*Planned feature for GDPR compliance:*

- Export all user health data
- Include generated insights and analysis
- Provide machine-readable format
- Maintain data integrity

### Account Deletion (Future)

*Planned feature for user privacy:*

- Delete Firebase user account
- Remove all health data from storage
- Delete generated insights and analysis
- Audit trail of deletion

## Implementation Details

### Current Implementation

- **Authentication**: `src/clarity/api/v1/auth.py`
- **User Context**: Firebase JWT token validation
- **Data Isolation**: User ID-based filtering
- **Profile Storage**: Firestore collections

### Firebase Configuration

- **Project**: CLARITY Digital Twin Firebase project
- **Authentication**: Email/password and future OAuth providers
- **Firestore**: User profiles and settings
- **Security Rules**: Strict user-based access control

### Key Dependencies

- `firebase-admin`: Server-side Firebase integration
- `pydantic`: Data validation and serialization
- `fastapi-users`: User management utilities (future)

## API Examples

### Get Current User Context

```python
# Available in all protected endpoints
@router.get("/profile")
async def get_user_profile(
    current_user: Annotated[dict, Depends(verify_firebase_token)]
):
    user_id = current_user["uid"]
    email = current_user["email"]
    # Return user-specific data
```

### User Data Query Pattern

```python
# All user data queries follow this pattern
user_health_data = await db.collection("health_data")\
    .where("user_id", "==", user_id)\
    .order_by("timestamp", direction=firestore.Query.DESCENDING)\
    .limit(100)\
    .get()
```

## Testing

User management is thoroughly tested:

- **Authentication Tests**: `tests/auth/`
- **Authorization Tests**: Validate user isolation
- **Integration Tests**: End-to-end user flows
- **Security Tests**: Token validation and access control

## Future Enhancements

1. **OAuth Integration**: Google, Apple Sign-In
2. **Advanced Profiles**: Detailed health questionnaires
3. **Data Export**: GDPR compliance features
4. **Team Sharing**: Controlled data sharing (healthcare providers)
5. **Multi-factor Auth**: Enhanced security options
