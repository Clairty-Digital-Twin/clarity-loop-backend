"""CLARITY Digital Twin Platform - Authentication Service.

Business logic layer for authentication operations.
Integrates with Firebase Authentication and handles user management.
"""

from datetime import UTC, datetime, timedelta
import logging
import secrets
from typing import Any, cast
import uuid

from firebase_admin import auth  # type: ignore[import-untyped]

from clarity.core.interfaces import IAuthProvider
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
from clarity.storage.firestore_client import FirestoreClient

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

    Handles user registration, login, session management, and integrates
    with Firebase Authentication for token operations.
    """

    def __init__(
        self,
        auth_provider: IAuthProvider,
        firestore_client: FirestoreClient,
        default_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 86400 * 30,  # 30 days
    ) -> None:
        """Initialize authentication service.

        Args:
            auth_provider: Authentication provider for token operations
            firestore_client: Firestore client for user data storage
            default_token_expiry: Default access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
        """
        self.auth_provider = auth_provider
        self.firestore_client = firestore_client
        self.default_token_expiry = default_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        # Collections
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
            # Check if user already exists
            try:
                existing_user = auth.get_user_by_email(request.email)  # type: ignore[misc]
                if existing_user:
                    error_msg = f"User with email {request.email} already exists"
                    raise UserAlreadyExistsError(error_msg)
            except auth.UserNotFoundError:  # type: ignore[misc]
                # User doesn't exist, which is what we want
                pass

            # Create Firebase user
            user_record = auth.create_user(  # type: ignore[misc]
                email=request.email,
                password=request.password,
                display_name=f"{request.first_name} {request.last_name}",
                email_verified=False,  # Require email verification
                disabled=False,
            )

            # Generate user ID
            user_id = uuid.UUID(user_record.uid)  # type: ignore[misc,arg-type]

            # Set custom claims for role-based access control
            custom_claims = {
                "role": UserRole.PATIENT.value,  # Default role
                "permissions": ["read_own_data", "write_own_data"],
                "created_at": datetime.now(UTC).isoformat(),
            }

            auth.set_custom_user_claims(user_record.uid, custom_claims)  # type: ignore[misc,arg-type]

            # Store additional user data in Firestore
            user_data = {
                "user_id": user_record.uid,  # type: ignore[misc]
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

            await self.firestore_client.create_document(
                collection=self.users_collection,
                data=user_data,  # type: ignore[arg-type]
                document_id=user_record.uid,  # type: ignore[misc,arg-type]
                user_id=user_record.uid,  # type: ignore[misc,arg-type]
            )

            # Send email verification
            verification_email_sent = False
            try:
                # Generate email verification link (unused for now)
                _ = auth.generate_email_verification_link(request.email)  # type: ignore[misc]
                # TODO: Send email using email service
                verification_email_sent = True
                logger.info("Email verification link generated for %s", request.email)
            except (auth.AuthError, ConnectionError, TimeoutError, OSError) as e:  # type: ignore[misc]
                logger.warning("Failed to send verification email: %s", e)  # type: ignore[misc]

            logger.info("User registered successfully: %s", user_record.uid)  # type: ignore[misc]

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
            # Get user by email
            try:
                user_record = auth.get_user_by_email(request.email)  # type: ignore[misc]
            except auth.UserNotFoundError as e:  # type: ignore[misc]
                error_msg = f"User with email {request.email} not found"
                raise UserNotFoundError(error_msg) from e

            # Check if account is disabled
            if user_record.disabled:  # type: ignore[misc]
                _raise_account_disabled()

            # For Firebase, password verification happens client-side
            # Here we simulate the verification process
            # In a real implementation, you would verify the password using Firebase client SDK

            # Get user data from Firestore
            user_data = await self.firestore_client.get_document(
                collection=self.users_collection, document_id=user_record.uid  # type: ignore[misc,arg-type]
            )

            if user_data is None:
                _raise_user_not_found_in_db()

            # user_data is guaranteed to be non-None after exception check above
            # Check email verification requirement
            if (
                not user_record.email_verified  # type: ignore[misc]
                and user_data.get("status") != UserStatus.ACTIVE.value  # type: ignore[union-attr]
            ):
                logger.warning("Login attempt with unverified email: %s", request.email)
                # For development, we'll allow unverified emails

            # Update user data
            login_time = datetime.now(UTC)
            update_data = {
                "last_login": login_time,
                "login_count": user_data.get("login_count", 0) + 1,  # type: ignore[union-attr]
                "updated_at": login_time,
            }

            await self.firestore_client.update_document(
                collection=self.users_collection,
                document_id=user_record.uid,  # type: ignore[misc,arg-type]
                data=update_data,
                user_id=user_record.uid,  # type: ignore[misc,arg-type]
            )

            # Check if MFA is enabled
            mfa_enabled = user_data.get("mfa_enabled", False)  # type: ignore[union-attr]
            if mfa_enabled:
                # Generate temporary session token for MFA completion
                mfa_session_token = secrets.token_urlsafe(32)

                # Store temporary session
                temp_session_data: dict[str, Any] = {
                    "user_id": user_record.uid,  # type: ignore[misc]
                    "mfa_session_token": mfa_session_token,
                    "created_at": login_time,
                    "expires_at": login_time
                    + timedelta(minutes=10),  # 10 minute expiry
                    "device_info": device_info,
                    "ip_address": ip_address,
                    "verified": False,
                }

                await self.firestore_client.create_document(
                    collection="mfa_sessions",
                    data=temp_session_data,  # type: ignore[arg-type]
                    user_id=user_record.uid,  # type: ignore[misc,arg-type]
                )

                # Return partial response requiring MFA
                user_session = await self._create_user_session_response(
                    user_record, user_data  # type: ignore[arg-type]
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

            # Generate tokens (in real implementation, this would be done by Firebase client SDK)
            tokens = await self._generate_tokens(
                user_record.uid, remember_me=request.remember_me  # type: ignore[misc,arg-type]
            )

            # Create session (store session_id for potential future use)
            _ = await self._create_user_session(
                user_record.uid,  # type: ignore[misc,arg-type]
                tokens.refresh_token,
                device_info,
                ip_address,
                remember_me=request.remember_me,
            )

            # Create user session response
            user_session = await self._create_user_session_response(
                user_record, user_data  # type: ignore[arg-type]
            )

            logger.info("User logged in successfully: %s", user_record.uid)  # type: ignore[misc,arg-type]

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
            "created_at": datetime.now(UTC),
            "expires_at": datetime.now(UTC) + timedelta(seconds=expires_in),
            "is_revoked": False,
        }

        await self.firestore_client.create_document(
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
            "created_at": datetime.now(UTC),
            "last_activity": datetime.now(UTC),
            "expires_at": datetime.now(UTC) + session_expiry,
            "device_info": device_info,
            "ip_address": ip_address,
            "is_active": True,
        }

        await self.firestore_client.create_document(
            collection=self.sessions_collection,
            data=session_data,
            document_id=session_id,
            user_id=user_id,
        )

        return session_id

    @staticmethod
    async def _create_user_session_response(
        user_record: auth.UserRecord, user_data: dict[str, Any]  # type: ignore[name-defined]
    ) -> UserSessionResponse:
        """Create user session response from user data.

        Args:
            user_record: Firebase user record
            user_data: User data from Firestore

        Returns:
            UserSessionResponse: User session information
        """
        # Validate email from Firebase user record
        email_value = user_record.email  # type: ignore[misc]
        if not email_value or not isinstance(email_value, str):
            error_msg = "Invalid email from Firebase user record"
            raise ValueError(error_msg)

        return UserSessionResponse(
            user_id=uuid.UUID(user_record.uid),  # type: ignore[misc]
            email=email_value,
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            role=user_data.get("role", UserRole.PATIENT.value),
            permissions=user_data.get("permissions", []),
            status=UserStatus(user_data.get("status", UserStatus.ACTIVE.value)),
            last_login=user_data.get("last_login"),
            mfa_enabled=user_data.get("mfa_enabled", False),
            email_verified=user_record.email_verified,
            created_at=user_data.get("created_at", datetime.now(UTC)),
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
            user_record = auth.get_user(user_id)  # type: ignore[misc]

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
