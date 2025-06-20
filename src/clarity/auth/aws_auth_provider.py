"""AWS Cognito authentication provider for CLARITY.

This module provides AWS Cognito integration with AWS-native solutions.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import json
import logging
import time
from typing import TYPE_CHECKING, Any, cast
import urllib.parse
import urllib.request

import boto3
from botocore.exceptions import ClientError
from jose import JWTError, jwt
from mypy_boto3_cognito_idp import CognitoIdentityProviderClient

if TYPE_CHECKING:
    pass  # Only for type stubs now

from clarity.models.auth import (
    AuthError,
    Permission,
    UserContext,
    UserRole,
    UserStatus,
)
from clarity.ports.auth_ports import IAuthProvider

logger = logging.getLogger(__name__)


class CognitoAuthProvider(IAuthProvider):
    """AWS Cognito authentication provider.

    Handles token verification and user information retrieval using AWS Cognito.
    """

    def __init__(
        self,
        user_pool_id: str,
        client_id: str,
        region: str = "us-east-1",
        dynamodb_service: Any = None,  # Optional DynamoDB service for user management
        middleware_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Cognito authentication provider.

        Args:
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            region: AWS region
            dynamodb_service: Optional DynamoDB service for user record management
            middleware_config: Middleware configuration options
        """
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.region = region
        self.dynamodb_service = dynamodb_service
        self.users_table = "clarity_users"

        # Initialize Cognito client
        self.cognito_client: CognitoIdentityProviderClient = boto3.client(
            "cognito-idp", region_name=region
        )

        # Get JWKS URL for token verification
        self.jwks_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"

        # Middleware config
        config_dict = middleware_config if middleware_config is not None else {}
        self.middleware_config = config_dict

        # Caching configuration
        auth_provider_config = self.middleware_config.get("auth_provider_config", {})
        self.cache_is_enabled = auth_provider_config.get("cache_enabled", True)
        self._token_cache_ttl_seconds = auth_provider_config.get(
            "cache_ttl_seconds", 300
        )
        self._token_cache_max_size = auth_provider_config.get("cache_max_size", 1000)
        self._token_cache: dict[str, dict[str, Any]] = {}
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl = 3600  # 1 hour

        self._initialized = False
        logger.info("Cognito Authentication Provider initialized.")
        logger.info("User Pool ID: %s", user_pool_id)
        logger.info("Region: %s", region)
        if dynamodb_service:
            logger.info("Enhanced mode: DynamoDB service available for user management")

    async def initialize(self) -> None:
        """Initialize Cognito provider and verify connection."""
        if self._initialized:
            return

        logger.info("🔐 Initializing Cognito Authentication Provider...")

        try:
            # Test connection by describing the user pool
            response = self.cognito_client.describe_user_pool(
                UserPoolId=self.user_pool_id
            )
            logger.info(
                "   -> Connected to Cognito User Pool: %s", response["UserPool"]["Name"]
            )

            # Pre-fetch JWKS for token verification
            await self._get_jwks()

            self._initialized = True
            logger.info("✅ Cognito Authentication Provider is ready.")

        except ClientError as e:
            logger.exception("💥 Failed to initialize Cognito Authentication Provider")
            msg = "Could not initialize Cognito Auth Provider"
            raise RuntimeError(msg) from e

    async def _get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Cognito for token verification."""
        current_time = time.time()

        # Check cache
        if (
            self._jwks_cache
            and (current_time - self._jwks_cache_time) < self._jwks_cache_ttl
        ):
            return self._jwks_cache

        try:
            # Validate URL scheme before opening
            parsed_url = urllib.parse.urlparse(self.jwks_url)
            if parsed_url.scheme != "https":
                msg = f"Invalid URL scheme: {parsed_url.scheme}. Only HTTPS is allowed."
                raise ValueError(msg)

            with urllib.request.urlopen(self.jwks_url) as response:  # noqa: S310
                jwks = json.loads(response.read())
                self._jwks_cache = jwks
                self._jwks_cache_time = current_time
                return cast("dict[str, Any]", jwks)
        except Exception as e:
            logger.exception("Failed to fetch JWKS: %s", e)
            if self._jwks_cache:
                # Return stale cache if available
                return self._jwks_cache
            raise

    def _remove_expired_tokens(self) -> None:
        """Remove expired tokens from the cache based on TTL."""
        if not self.cache_is_enabled:
            return
        current_time = time.time()
        expired_tokens = [
            t
            for t, data in self._token_cache.items()
            if current_time - data["timestamp"] > self._token_cache_ttl_seconds
        ]
        for t in expired_tokens:
            if t in self._token_cache:
                del self._token_cache[t]
                logger.debug("Removed expired token from cache")

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify Cognito ID token and return user information.

        Args:
            token: Cognito ID token

        Returns:
            User information dictionary if token is valid, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        self._remove_expired_tokens()

        # Check cache first if enabled
        if self.cache_is_enabled and token in self._token_cache:
            logger.debug("Token found in cache")
            return cast("dict[str, Any]", self._token_cache[token]["user_data"])

        logger.debug("🔐 COGNITO VERIFY_TOKEN CALLED")

        try:
            # Get JWKS for verification
            jwks = await self._get_jwks()

            # Decode and verify the token
            # First, decode without verification to get the header
            unverified_header = jwt.get_unverified_header(token)

            # Find the correct key
            rsa_key = None
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = key
                    break

            if not rsa_key:
                raise AuthError(
                    message="Unable to find appropriate key",
                    status_code=401,
                    error_code="invalid_key",
                )

            # Verify the token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer=f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}",
            )

            # Extract user information
            user_info = {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "verified": payload.get("email_verified", False),
                "custom_claims": payload.get("custom", {}),
                "cognito_username": payload.get("cognito:username"),
                "token_use": payload.get("token_use"),
            }

            # Cache the result
            if self.cache_is_enabled:
                self._token_cache[token] = {
                    "user_data": user_info,
                    "timestamp": time.time(),
                }

            logger.debug("✅ COGNITO TOKEN VERIFIED SUCCESSFULLY")
            return user_info

        except AuthError:
            # Re-raise AuthError as-is to preserve original message
            raise
        except JWTError as e:
            logger.exception("❌ COGNITO ERROR: JWT verification failed: %s", e)
            raise AuthError(
                message=f"Invalid Cognito token: {e}",
                status_code=401,
                error_code="invalid_token",
            ) from e
        except Exception as e:
            logger.exception("❌ UNKNOWN COGNITO AUTH ERROR")
            raise AuthError(
                message="An unexpected error occurred",
                status_code=500,
                error_code="unknown_auth_error",
            ) from e

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by Cognito username or email.

        Args:
            user_id: Cognito username or email

        Returns:
            User information dictionary if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Try to get user by username first
            attributes: dict[str, str] = {}
            try:
                cognito_response = self.cognito_client.admin_get_user(
                    UserPoolId=self.user_pool_id,
                    Username=user_id,
                )
                # Extract attributes from Cognito response
                attributes = {
                    attr["Name"]: attr["Value"]
                    for attr in cognito_response.get("UserAttributes", [])
                }
            except ClientError as e:
                if e.response["Error"]["Code"] == "UserNotFoundException":
                    # Try by email if username lookup failed
                    users = self.cognito_client.list_users(
                        UserPoolId=self.user_pool_id,
                        Filter=f'email = "{user_id}"',
                        Limit=1,
                    )
                    if not users.get("Users"):
                        return None
                    # Extract attributes from list users response
                    attributes = {
                        attr["Name"]: attr["Value"]
                        for attr in users["Users"][0].get("Attributes", [])
                    }
                else:
                    raise

            # Extract custom attributes to determine roles
            roles = []
            if attributes.get("custom:role") == "admin":
                roles.append("admin")
            if attributes.get("custom:role") == "clinician":
                roles.append("clinician")

            # Return user data
            return {
                "user_id": attributes.get("sub"),
                "email": attributes.get("email"),
                "verified": attributes.get("email_verified", "false").lower() == "true",
                "roles": roles,
                "custom_claims": {
                    "given_name": attributes.get("given_name"),
                    "family_name": attributes.get("family_name"),
                    "phone_number": attributes.get("phone_number"),
                },
                "created_at": datetime.now(
                    UTC
                ),  # Cognito doesn't provide creation time easily
                "last_login": datetime.now(UTC),  # Would need to track separately
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "UserNotFoundException":
                logger.debug("User not found: %s", user_id)
                return None
            logger.exception("Error fetching user info for %s", user_id)
            return None

    async def get_or_create_user_context(
        self, cognito_user_info: dict[str, Any]
    ) -> UserContext:
        """Get user context, creating DynamoDB record if needed.

        Args:
            cognito_user_info: User info from Cognito token verification

        Returns:
            UserContext with complete user information
        """
        if not self.dynamodb_service:
            # If no DynamoDB service, create basic context
            return self._create_basic_user_context(cognito_user_info)

        user_id = cognito_user_info["user_id"]

        try:
            # Try to get existing user record
            user_data = await self.dynamodb_service.get_item(
                table_name=self.users_table,
                key={"user_id": user_id},
            )

            if user_data is None:
                # User doesn't exist in DynamoDB, create it
                logger.info("Creating new DynamoDB user record for %s", user_id)
                user_data = await self._create_user_record(cognito_user_info)
            else:
                # Update last login
                await self.dynamodb_service.update_item(
                    table_name=self.users_table,
                    key={"user_id": user_id},
                    update_expression="SET last_login = :login_time, login_count = login_count + :inc",
                    expression_attribute_values={
                        ":login_time": datetime.now(UTC).isoformat(),
                        ":inc": 1,
                    },
                    user_id=user_id,
                )

            # Create UserContext from database record
            return self._create_user_context_from_db(user_data, cognito_user_info)

        except Exception:
            logger.exception("Error creating/fetching user context")
            # Fall back to basic context creation
            return self._create_basic_user_context(cognito_user_info)

    def _create_basic_user_context(self, user_info: dict[str, Any]) -> UserContext:
        """Create basic UserContext from Cognito user info."""
        # Extract user role from custom claims
        custom_claims = user_info.get("custom_claims", {})
        role_str = custom_claims.get("role", "patient")

        if role_str == "admin":
            role = UserRole.ADMIN
        elif role_str == "clinician":
            role = UserRole.CLINICIAN
        else:
            role = UserRole.PATIENT

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

        return UserContext(
            user_id=user_info["user_id"],
            email=user_info["email"],
            role=role,
            permissions=list(permissions),
            is_verified=user_info.get("verified", False),
            custom_claims=custom_claims,
            created_at=datetime.now(UTC),
            last_login=datetime.now(UTC),
        )

    async def _create_user_record(
        self, cognito_user_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new user record in DynamoDB.

        Args:
            cognito_user_info: User info from Cognito

        Returns:
            Created user data
        """
        user_id = cognito_user_info["user_id"]
        email = cognito_user_info.get("email", "")
        custom_claims = cognito_user_info.get("custom_claims", {})

        # Extract name from custom claims
        first_name = custom_claims.get("given_name", "")
        last_name = custom_claims.get("family_name", "")

        # Determine role
        role_str = custom_claims.get("role", "patient")
        if role_str == "admin":
            role = UserRole.ADMIN
        elif role_str == "clinician":
            role = UserRole.CLINICIAN
        else:
            role = UserRole.PATIENT

        user_data = {
            "user_id": user_id,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "display_name": f"{first_name} {last_name}".strip(),
            "status": UserStatus.ACTIVE.value,  # Auto-activate Cognito users
            "role": role.value,
            "auth_provider": "COGNITO",
            "email_verified": cognito_user_info.get("verified", False),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "last_login": datetime.now(UTC).isoformat(),
            "login_count": 1,
            "mfa_enabled": False,
            "mfa_methods": [],
            "custom_claims": custom_claims,
            "terms_accepted": True,  # Assume accepted if using Cognito
            "privacy_policy_accepted": True,
        }

        await self.dynamodb_service.put_item(
            table_name=self.users_table,
            item=user_data,
            user_id=user_id,
        )

        logger.info("Created DynamoDB user record for %s", user_id)
        return user_data

    def _create_user_context_from_db(
        self, user_data: dict[str, Any], _cognito_info: dict[str, Any]
    ) -> UserContext:
        """Create UserContext from database record.

        Args:
            user_data: User data from DynamoDB
            _cognito_info: Original Cognito token info (unused)

        Returns:
            Complete UserContext
        """
        # Determine role
        role_str = user_data.get("role", UserRole.PATIENT.value)
        role = (
            UserRole(role_str)
            if role_str in [r.value for r in UserRole]
            else UserRole.PATIENT
        )

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

        # Parse dates
        created_at = user_data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        last_login = user_data.get("last_login")
        if isinstance(last_login, str):
            last_login = datetime.fromisoformat(last_login)

        # Preserve original custom_claims without enrichment to match test contracts
        original_claims = user_data.get("custom_claims", {})

        return UserContext(
            user_id=user_data["user_id"],
            email=user_data["email"],
            role=role,
            permissions=list(permissions),
            is_verified=user_data.get("email_verified", False),
            is_active=is_active,
            custom_claims=original_claims,
            created_at=created_at,
            last_login=last_login,
        )

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        self._token_cache.clear()
        self._jwks_cache = None
        logger.info("Cognito authentication provider cleanup complete")
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
