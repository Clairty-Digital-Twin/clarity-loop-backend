# Authentication Fix Plan

## The Problem

The current authentication system has a critical gap:
- Firebase users created outside the backend (e.g., iOS app) don't have corresponding Firestore user records
- The middleware expects these records to exist for proper UserContext creation
- This causes 401 errors even with valid Firebase tokens

## The Solution

Implement automatic user record creation (upsert) when Firebase users authenticate for the first time.

### Implementation Steps

1. **Modify FirebaseAuthMiddleware** to use the enhanced auth provider
2. **Auto-create Firestore records** for new Firebase users
3. **Preserve existing functionality** for users who register through the backend

### Key Changes Needed

#### 1. Update the Middleware (`firebase_middleware.py`)

```python
# In _authenticate_request method, after token verification:
user_info = await self.auth_provider.verify_token(token)

# Add Firestore record check/creation
if hasattr(self.auth_provider, 'get_or_create_user_context'):
    # Use enhanced provider that handles Firestore
    user_context = await self.auth_provider.get_or_create_user_context(user_info)
else:
    # Fallback to current behavior
    user_context = self._create_user_context(user_info)
```

#### 2. Update Container to Inject Firestore Client

```python
# In container.py, when creating auth provider:
auth_provider = FirebaseAuthProvider(
    credentials_path=credentials_path,
    project_id=project_id,
    middleware_config=middleware_config,
    firestore_client=firestore_client  # Add this
)
```

#### 3. Enhance FirebaseAuthProvider

Add the upsert logic from `firebase_auth_enhanced.py` to the main provider.

## Benefits

1. **Seamless iOS Integration**: Users created in iOS app will automatically work
2. **Backward Compatible**: Existing registration flow remains unchanged
3. **Data Consistency**: All authenticated users will have Firestore records
4. **Proper Permissions**: UserContext will include database-derived permissions

## Testing Plan

1. Create a new user in Firebase Console or iOS app
2. Authenticate with that user's token
3. Verify Firestore record is created automatically
4. Verify all endpoints work with proper UserContext

## Alternative Quick Fix (Not Recommended)

If you need a temporary workaround:
1. Make all endpoints use `get_current_user_required` (Firebase only)
2. Skip Firestore lookups entirely
3. This loses user metadata and permissions - NOT RECOMMENDED for production

## Rollback Plan

If issues arise:
1. Revert to commit before authentication changes
2. Ensure Firebase project ID is correctly configured
3. Test with users created through proper registration flow