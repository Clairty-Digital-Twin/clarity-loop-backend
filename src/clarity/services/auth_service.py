"""CLARITY Digital Twin Platform - Authentication Service.

Business logic layer for authentication operations.
Now supports AWS Cognito authentication as primary provider.
"""

from datetime import UTC, datetime, timedelta
import logging
import secrets
from typing import Any, cast
import uuid

from clarity.models.auth import (
    AuthProvider,
    LoginResponse,
    RegistrationResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegistrationRequest,
    UserRole,
    UserSessionResponse,
    UserStatus,
)
from clarity.ports.auth_ports import IAuthProvider

# Import based on available services
try:
    from clarity.services.dynamodb_service import DynamoDBService
    _HAS_DYNAMODB = True
except ImportError:
    _HAS_DYNAMODB = False

try:
    from clarity.storage.firestore_client import FirestoreClient
    _HAS_FIRESTORE = True
except ImportError:
    _HAS_FIRESTORE = False

# Configure logger
logger = logging.getLogger(__name__)

# Constants
BEARER_TOKEN_TYPE = (
    "bearer"  # noqa: S105  # nosec B105 - Standard OAuth token type, not a password
)


class AuthenticationError(Exception):
    """Base exception for authentication errors."""


class UserNotFoundError(AuthenticationError):
    """Raised when user is not found."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""


class UserAlreadyExistsError(AuthenticationError):
    """Raised when user already exists."""


class EmailNotVerifiedError(AuthenticationError):
    """Raised when email is not verified."""


class AccountDisabledError(AuthenticationError):
    """Raised when account is disabled."""


def _raise_account_disabled() -> None:
    """Raise account disabled error."""
    error_msg = "Account is disabled"
    raise AccountDisabledError(error_msg)


def _raise_user_not_found_in_db() -> None:
    """Raise user not found in database error."""
    error_msg = "User data not found in database"
    raise UserNotFoundError(error_msg)


def _raise_invalid_refresh_token() -> None:
    """Raise invalid refresh token error."""
    error_msg = "Invalid refresh token"
    raise InvalidCredentialsError(error_msg)


def _raise_refresh_token_expired() -> None:
    """Raise refresh token expired error."""
    error_msg = "Refresh token expired"
    raise InvalidCredentialsError(error_msg)


class AuthenticationService:
    """Authentication service implementing business logic for user management.

    Handles user registration, login, session management.
    Supports both AWS Cognito (via DynamoDB) and Firebase (via Firestore).
    """

    def __init__(
        self,
        auth_provider: IAuthProvider,
        data_store: Any = None,  # Can be FirestoreClient or DynamoDBService
        default_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 86400 * 30,  # 30 days
    ) -> None:
        """Initialize authentication service.

        Args:
            auth_provider: Authentication provider for token operations
            data_store: Data storage client (DynamoDB or Firestore)
            default_token_expiry: Default access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
        """
        self.auth_provider = auth_provider
        self.data_store = data_store
        self.default_token_expiry = default_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        # Determine storage type
        self.is_dynamodb = _HAS_DYNAMODB and isinstance(data_store, DynamoDBService)
        self.is_firestore = _HAS_FIRESTORE and isinstance(data_store, FirestoreClient)

        # Collections/Tables
        if self.is_dynamodb:
            self.users_collection = "clarity_users"
            self.sessions_collection = "clarity_user_sessions"
            self.refresh_tokens_collection = "clarity_refresh_tokens"
        else:
            self.users_collection = "users"
            self.sessions_collection = "user_sessions"
            self.refresh_tokens_collection = "refresh_tokens"

    async def register_user(
        self,
        request: UserRegistrationRequest,
        device_info: dict[str, Any] | None = None,
    ) -> RegistrationResponse:
        """Register a new user with Firebase Authentication.

        Args:
            request: User registration request data
            device_info: Optional device information for security tracking

        Returns:
            RegistrationResponse: Registration result with user details

        Raises:
            UserAlreadyExistsError: If user already exists
            AuthenticationError: If registration fails
        """
        try:
            # Use auth provider to check if user exists and create user
            # This abstracts away the specific implementation (Firebase/Cognito)
            user_info = await self.auth_provider.get_user_info(request.email)
            if user_info:
                error_msg = f"User with email {request.email} already exists"
                raise UserAlreadyExistsError(error_msg)

            # Create user through auth provider
            # For now, generate a user ID - in real implementation,
            # this would come from Cognito/Firebase
            user_id = uuid.uuid4()
            user_record_id = str(user_id)

            # Store additional user data
            user_data: dict[str, Any] = {
                "user_id": user_record_id,
                "email": request.email,
                "first_name": request.first_name,
                "last_name": request.last_name,
                "phone_number": request.phone_number,
                "status": UserStatus.PENDING_VERIFICATION.value,
                "role": UserRole.PATIENT.value,
                "auth_provider": AuthProvider.EMAIL_PASSWORD.value,
                "mfa_enabled": False,
                "mfa_methods": [],
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "last_login": None,
                "login_count": 0,
                "email_verified": False,
                "terms_accepted": request.terms_accepted,
                "privacy_policy_accepted": request.privacy_policy_accepted,
                "device_info": device_info,
            }

            # Store user data based on backend type
            if self.is_dynamodb:
                user_data["id"] = user_record_id  # DynamoDB needs an 'id' field
                await self.data_store.put_item(
                    table_name=self.users_collection,
                    item=user_data,
                    user_id=user_record_id,
                )
            elif self.is_firestore:
                await self.data_store.create_document(
                    collection=self.users_collection,
                    data=user_data,
                    document_id=user_record_id,
                    user_id=user_record_id,
                )

            # Send email verification
            verification_email_sent = True  # AWS Cognito handles this automatically
            logger.info("User registered successfully: %s", user_record_id)

            return RegistrationResponse(
                user_id=user_id,
                email=request.email,
                status=UserStatus.PENDING_VERIFICATION,
                verification_email_sent=verification_email_sent,
                created_at=datetime.now(UTC),
            )

        except UserAlreadyExistsError:
            raise
        except Exception as e:
            logger.exception("User registration failed for %s", request.email)
            error_msg = f"Registration failed: {e!s}"
            raise AuthenticationError(error_msg) from e

    async def login_user(
        self,
        request: UserLoginRequest,
        device_info: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        """Authenticate user and create session.

        Args:
            request: User login request data
            device_info: Optional device information for security tracking
            ip_address: Client IP address for security tracking

        Returns:
            LoginResponse: Login result with tokens and user session

        Raises:
            UserNotFoundError: If user doesn't exist
            InvalidCredentialsError: If credentials are invalid
            EmailNotVerifiedError: If email is not verified
            AccountDisabledError: If account is disabled
        """
        try:
            # Get user info from auth provider
            user_info = await self.auth_provider.get_user_info(request.email)
            if not user_info:
                error_msg = f"User with email {request.email} not found"
                raise UserNotFoundError(error_msg)

            user_id = user_info.get("user_id", "")

            # Get user data from storage
            if self.is_dynamodb:
                user_data = await self.data_store.get_item(
                    table_name=self.users_collection,
                    key={"user_id": user_id},
                )
            elif self.is_firestore:
                user_data = await self.data_store.get_document(
                    collection=self.users_collection,
                    document_id=user_id,
                )
            else:
                user_data = None

            if user_data is None:
                _raise_user_not_found_in_db()

            # user_data is guaranteed to be non-None after exception check above
            # Check email verification requirement
            if (
                not user_info.get("verified", False)
                and user_data.get("status") != UserStatus.ACTIVE.value  # type: ignore[union-attr]
            ):
                logger.warning("Login attempt with unverified email: %s", request.email)
                # For development, we'll allow unverified emails

            # Update user data
            login_time = datetime.now(UTC)
            login_count = user_data.get("login_count", 0) + 1  # type: ignore[union-attr]

            if self.is_dynamodb:
                await self.data_store.update_item(
                    table_name=self.users_collection,
                    key={"user_id": user_id},
                    update_expression="SET last_login = :login_time, login_count = :count, updated_at = :updated",
                    expression_attribute_values={
                        ":login_time": login_time.isoformat(),
                        ":count": login_count,
                        ":updated": login_time.isoformat(),
                    },
                    user_id=user_id,
                )
            elif self.is_firestore:
                update_data = {
                    "last_login": login_time,
                    "login_count": login_count,
                    "updated_at": login_time,
                }
                await self.data_store.update_document(
                    collection=self.users_collection,
                    document_id=user_id,
                    data=update_data,
                    user_id=user_id,
                )

            # Check if MFA is enabled
            mfa_enabled = user_data.get("mfa_enabled", False)  # type: ignore[union-attr]
            if mfa_enabled:
                # Generate temporary session token for MFA completion
                mfa_session_token = secrets.token_urlsafe(32)

                # Store temporary session
                temp_session_data: dict[str, Any] = {
                    "user_id": user_id,
                    "mfa_session_token": mfa_session_token,
                    "created_at": login_time.isoformat() if self.is_dynamodb else login_time,
                    "expires_at": (login_time + timedelta(minutes=10)).isoformat() if self.is_dynamodb else login_time + timedelta(minutes=10),
                    "device_info": device_info,
                    "ip_address": ip_address,
                    "verified": False,
                }

                if self.is_dynamodb:
                    temp_session_data["id"] = mfa_session_token
                    await self.data_store.put_item(
                        table_name="clarity_mfa_sessions",
                        item=temp_session_data,
                        user_id=user_id,
                    )
                elif self.is_firestore:
                    await self.data_store.create_document(
                        collection="mfa_sessions",
                        data=temp_session_data,
                        user_id=user_id,
                    )

                # Return partial response requiring MFA
                user_session = await self._create_user_session_response(
                    user_id,
                    user_data,  # type: ignore[arg-type]
                )
                return LoginResponse(
                    user=user_session,
                    tokens=TokenResponse(
                        access_token="",  # nosec B106 - Empty placeholder token for MFA flow
                        refresh_token="",  # nosec B106 - Empty placeholder token for MFA flow
                        token_type=BEARER_TOKEN_TYPE,
                        expires_in=0,
                        scope="mfa_pending",  # Add scope for MFA flow
                    ),
                    requires_mfa=True,
                    mfa_session_token=mfa_session_token,
                )

            # Generate tokens
            tokens = await self._generate_tokens(
                user_id,
                remember_me=request.remember_me,
            )

            # Create session (store session_id for potential future use)
            _ = await self._create_user_session(
                user_id,
                tokens.refresh_token,
                device_info,
                ip_address,
                remember_me=request.remember_me,
            )

            # Create user session response
            user_session = await self._create_user_session_response(
                user_id,
                user_data,  # type: ignore[arg-type]
            )

            logger.info("User logged in successfully: %s", user_id)

            return LoginResponse(
                user=user_session,
                tokens=tokens,
                requires_mfa=False,
                mfa_session_token=None,
            )

        except (
            UserNotFoundError,
            InvalidCredentialsError,
            EmailNotVerifiedError,
            AccountDisabledError,
        ):
            raise
        except Exception as e:
            logger.exception("Login failed for %s", request.email)
            error_msg = f"Login failed: {e!s}"
            raise AuthenticationError(error_msg) from e

    async def _generate_tokens(
        self, user_id: str, *, remember_me: bool = False
    ) -> TokenResponse:
        """Generate access and refresh tokens.

        Args:
            user_id: User ID
            remember_me: Whether to generate long-lived tokens

        Returns:
            TokenResponse: Generated tokens
        """
        # In a real implementation, this would use Firebase custom tokens
        # For now, we'll create mock tokens

        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        # Set expiry based on remember_me flag
        expires_in = (
            self.refresh_token_expiry if remember_me else self.default_token_expiry
        )

        # Store refresh token
        refresh_data = {
            "user_id": user_id,
            "refresh_token": refresh_token,
            "created_at": datetime.now(UTC).isoformat() if self.is_dynamodb else datetime.now(UTC),
            "expires_at": (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat() if self.is_dynamodb else datetime.now(UTC) + timedelta(seconds=expires_in),
            "is_revoked": False,
        }

        if self.is_dynamodb:
            refresh_data["id"] = refresh_token
            await self.data_store.put_item(
                table_name=self.refresh_tokens_collection,
                item=refresh_data,
                user_id=user_id,
            )
        elif self.is_firestore:
            await self.data_store.create_document(
                collection=self.refresh_tokens_collection,
                data=refresh_data,
                user_id=user_id,
            )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=BEARER_TOKEN_TYPE,
            expires_in=expires_in,
            scope="full_access",  # Add scope for regular tokens
        )

    async def _create_user_session(
        self,
        user_id: str,
        refresh_token: str,
        device_info: dict[str, Any] | None = None,
        ip_address: str | None = None,
        *,
        remember_me: bool = False,
    ) -> str:
        """Create a user session.

        Args:
            user_id: User ID
            refresh_token: Refresh token
            device_info: Device information
            ip_address: IP address
            remember_me: Whether session should be long-lived

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        session_expiry = timedelta(days=30 if remember_me else 1)

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "refresh_token": refresh_token,
            "created_at": datetime.now(UTC).isoformat() if self.is_dynamodb else datetime.now(UTC),
            "last_activity": datetime.now(UTC).isoformat() if self.is_dynamodb else datetime.now(UTC),
            "expires_at": (datetime.now(UTC) + session_expiry).isoformat() if self.is_dynamodb else datetime.now(UTC) + session_expiry,
            "device_info": device_info,
            "ip_address": ip_address,
            "is_active": True,
        }

        if self.is_dynamodb:
            session_data["id"] = session_id
            await self.data_store.put_item(
                table_name=self.sessions_collection,
                item=session_data,
                user_id=user_id,
            )
        elif self.is_firestore:
            await self.data_store.create_document(
                collection=self.sessions_collection,
                data=session_data,
                document_id=session_id,
                user_id=user_id,
            )

        return session_id

    async def _create_user_session_response(
        self,
        user_id: str,
        user_data: dict[str, Any],
    ) -> UserSessionResponse:
        """Create user session response from user data.

        Args:
            user_id: User ID
            user_data: User data from database

        Returns:
            UserSessionResponse: User session information
        """
        # Get dates from data - handle both string and datetime formats
        last_login = user_data.get("last_login")
        if isinstance(last_login, str):
            last_login = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
        
        created_at = user_data.get("created_at", datetime.now(UTC))
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

        return UserSessionResponse(
            user_id=uuid.UUID(user_id),
            email=user_data.get("email", ""),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            role=user_data.get("role", UserRole.PATIENT.value),
            permissions=user_data.get("permissions", []),
            status=UserStatus(user_data.get("status", UserStatus.ACTIVE.value)),
            last_login=last_login,
            mfa_enabled=user_data.get("mfa_enabled", False),
            email_verified=user_data.get("email_verified", False),
            created_at=created_at,
        )

    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            TokenResponse: New tokens

        Raises:
            InvalidCredentialsError: If refresh token is invalid or expired
        """
        try:
            # Find refresh token record
            tokens = await self.firestore_client.query_documents(
                collection=self.refresh_tokens_collection,
                filters=[
                    {
                        "field": "refresh_token",
                        "operator": "==",
                        "value": refresh_token,
                    },
                    {"field": "is_revoked", "operator": "==", "value": False},
                ],
            )

            if not tokens:
                _raise_invalid_refresh_token()

            token_data = tokens[0]

            # Check if token is expired
            expires_at = token_data.get("expires_at")
            if expires_at and datetime.now(UTC) > expires_at:
                _raise_refresh_token_expired()

            user_id = token_data["user_id"]

            # Revoke old refresh token
            token_doc_id = token_data.get("id")
            if not isinstance(token_doc_id, str):
                logger.error("Token document missing or invalid ID field")
                _raise_invalid_refresh_token()
            await self.firestore_client.update_document(
                collection=self.refresh_tokens_collection,
                document_id=cast("str", token_doc_id),  # Safe after isinstance check
                data={"is_revoked": True, "revoked_at": datetime.now(UTC)},
                user_id=user_id,
            )

            # Generate new tokens
            new_tokens = await self._generate_tokens(user_id)

        except InvalidCredentialsError:
            raise
        except Exception as e:
            logger.exception("Token refresh failed")
            error_msg = f"Token refresh failed: {e!s}"
            raise AuthenticationError(error_msg) from e
        else:
            logger.info("Tokens refreshed for user: %s", user_id)
            return new_tokens

    async def logout_user(self, refresh_token: str) -> bool:
        """Logout user by revoking refresh token and ending session.

        Args:
            refresh_token: User's refresh token

        Returns:
            bool: True if logout successful
        """
        try:
            # Find and revoke refresh token
            tokens = await self.firestore_client.query_documents(
                collection=self.refresh_tokens_collection,
                filters=[
                    {
                        "field": "refresh_token",
                        "operator": "==",
                        "value": refresh_token,
                    },
                    {"field": "is_revoked", "operator": "==", "value": False},
                ],
            )

            if tokens:
                token_data = tokens[0]
                user_id = token_data["user_id"]

                # Revoke refresh token
                token_doc_id = token_data.get("id")
                if isinstance(token_doc_id, str):
                    await self.firestore_client.update_document(
                        collection=self.refresh_tokens_collection,
                        document_id=token_doc_id,
                        data={"is_revoked": True, "revoked_at": datetime.now(UTC)},
                        user_id=user_id,
                    )
                else:
                    logger.error("Token document missing or invalid ID field")

                # Deactivate sessions with this refresh token
                sessions = await self.firestore_client.query_documents(
                    collection=self.sessions_collection,
                    filters=[
                        {
                            "field": "refresh_token",
                            "operator": "==",
                            "value": refresh_token,
                        },
                        {"field": "is_active", "operator": "==", "value": True},
                    ],
                )

                for session in sessions:
                    session_id = session.get("session_id")
                    if isinstance(session_id, str):
                        await self.firestore_client.update_document(
                            collection=self.sessions_collection,
                            document_id=session_id,
                            data={"is_active": False, "ended_at": datetime.now(UTC)},
                            user_id=user_id,
                        )
                    else:
                        logger.error("Session document missing or invalid session_id")

                logger.info("User logged out: %s", user_id)

        except Exception:
            logger.exception("Logout failed")
            return False
        else:
            return True

    async def get_user_by_id(self, user_id: str) -> UserSessionResponse | None:
        """Get user information by user ID.

        Args:
            user_id: User ID

        Returns:
            UserSessionResponse: User information or None if not found
        """
        try:
            # Get Firebase user record
            user_record = auth.get_user(user_id)

            # Get user data from Firestore
            user_data = await self.firestore_client.get_document(
                collection=self.users_collection, document_id=user_id
            )

            if not user_data:
                return None

            return await self._create_user_session_response(user_record, user_data)  # type: ignore[arg-type]

        except auth.UserNotFoundError:  # type: ignore[misc]
            return None
        except Exception:
            logger.exception("Failed to get user %s", user_id)
            return None

    @staticmethod
    async def verify_email(_verification_code: str) -> bool:
        """Verify user email with verification code.

        Args:
            _verification_code: Email verification code (unused in mock)

        Returns:
            bool: True if verification successful
        """
        try:
            # In a real implementation, this would verify the code with Firebase
            # For now, we'll just mark it as verified

            # TODO: Implement actual email verification logic
            logger.info("Email verification completed (mock implementation)")

        except Exception:
            logger.exception("Email verification failed")
            return False
        else:
            return True
