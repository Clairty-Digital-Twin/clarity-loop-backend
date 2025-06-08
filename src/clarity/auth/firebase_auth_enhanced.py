"""Enhanced Firebase authentication with automatic user record creation.

This module provides an enhanced authentication flow that automatically creates
Firestore user records for Firebase users on their first authentication.
"""

from datetime import UTC, datetime
import logging
from typing import Any

from firebase_admin import auth

from clarity.models.auth import AuthProvider, Permission, UserContext, UserRole, UserStatus
from clarity.storage.firestore_client import FirestoreClient

logger = logging.getLogger(__name__)


class EnhancedFirebaseAuthProvider:
    """Enhanced Firebase auth provider that handles user record creation."""

    def __init__(self, firestore_client: FirestoreClient, project_id: str | None = None):
        """Initialize enhanced auth provider.
        
        Args:
            firestore_client: Firestore client for user record management
            project_id: Firebase project ID
        """
        self.firestore_client = firestore_client
        self.project_id = project_id
        self.users_collection = "users"

    async def get_or_create_user_context(self, firebase_user_info: dict[str, Any]) -> UserContext:
        """Get user context, creating Firestore record if needed.
        
        Args:
            firebase_user_info: User info from Firebase token verification
            
        Returns:
            UserContext with complete user information
        """
        user_id = firebase_user_info["user_id"]

        # Try to get existing user record
        user_data = await self.firestore_client.get_document(
            collection=self.users_collection,
            document_id=user_id
        )

        if user_data is None:
            # User doesn't exist in Firestore, create it
            logger.info(f"Creating new Firestore user record for {user_id}")
            user_data = await self._create_user_record(firebase_user_info)
        else:
            # Update last login
            await self.firestore_client.update_document(
                collection=self.users_collection,
                document_id=user_id,
                data={
                    "last_login": datetime.now(UTC),
                    "login_count": user_data.get("login_count", 0) + 1,
                },
                user_id=user_id
            )

        # Create UserContext from database record
        return self._create_user_context_from_db(user_data, firebase_user_info)

    async def _create_user_record(self, firebase_user_info: dict[str, Any]) -> dict[str, Any]:
        """Create a new user record in Firestore.
        
        Args:
            firebase_user_info: User info from Firebase
            
        Returns:
            Created user data
        """
        user_id = firebase_user_info["user_id"]
        email = firebase_user_info.get("email", "")

        # Extract name from Firebase if available
        display_name = firebase_user_info.get("display_name", "")
        name_parts = display_name.split(" ", 1) if display_name else ["", ""]
        first_name = name_parts[0] if len(name_parts) > 0 else ""
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Determine role from custom claims
        custom_claims = firebase_user_info.get("custom_claims", {})
        if custom_claims.get("admin"):
            role = UserRole.ADMIN
        elif custom_claims.get("clinician"):
            role = UserRole.CLINICIAN
        else:
            role = UserRole.PATIENT

        user_data = {
            "user_id": user_id,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "display_name": display_name,
            "status": UserStatus.ACTIVE.value,  # Auto-activate Firebase users
            "role": role.value,
            "auth_provider": AuthProvider.FIREBASE.value,
            "email_verified": firebase_user_info.get("verified", False),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_login": datetime.now(UTC),
            "login_count": 1,
            "mfa_enabled": False,
            "mfa_methods": [],
            "custom_claims": custom_claims,
            "terms_accepted": True,  # Assume accepted if using Firebase
            "privacy_policy_accepted": True,
        }

        await self.firestore_client.create_document(
            collection=self.users_collection,
            data=user_data,
            document_id=user_id,
            user_id=user_id
        )

        logger.info(f"Created Firestore user record for {user_id}")
        return user_data

    def _create_user_context_from_db(
        self,
        user_data: dict[str, Any],
        firebase_info: dict[str, Any]
    ) -> UserContext:
        """Create UserContext from database record.
        
        Args:
            user_data: User data from Firestore
            firebase_info: Original Firebase token info
            
        Returns:
            Complete UserContext
        """
        # Determine role
        role_str = user_data.get("role", UserRole.PATIENT.value)
        role = UserRole(role_str) if role_str in [r.value for r in UserRole] else UserRole.PATIENT

        # Set permissions based on role
        permissions = set()
        if role == UserRole.ADMIN:
            permissions = {
                Permission.SYSTEM_ADMIN,
                Permission.MANAGE_USERS,
                Permission.READ_OWN_DATA,
                Permission.WRITE_OWN_DATA,
                Permission.READ_PATIENT_DATA,
                Permission.WRITE_PATIENT_DATA,
                Permission.READ_ANONYMIZED_DATA,
            }
        elif role == UserRole.CLINICIAN:
            permissions = {
                Permission.READ_OWN_DATA,
                Permission.WRITE_OWN_DATA,
                Permission.READ_PATIENT_DATA,
                Permission.WRITE_PATIENT_DATA,
            }
        else:  # PATIENT
            permissions = {Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA}

        # Check if user is active
        is_active = user_data.get("status") == UserStatus.ACTIVE.value

        # Store extra fields in custom_claims for access later
        enriched_claims = user_data.get("custom_claims", {}).copy()
        enriched_claims.update({
            "first_name": user_data.get("first_name"),
            "last_name": user_data.get("last_name"),
            "display_name": user_data.get("display_name"),
        })
        
        return UserContext(
            user_id=user_data["user_id"],
            email=user_data["email"],
            role=role,
            permissions=list(permissions),
            is_verified=user_data.get("email_verified", False),
            is_active=is_active,
            custom_claims=enriched_claims,
            created_at=user_data.get("created_at"),
            last_login=user_data.get("last_login"),
        )
