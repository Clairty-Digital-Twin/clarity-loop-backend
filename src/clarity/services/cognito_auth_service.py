"""CLARITY Digital Twin Platform - AWS Cognito Authentication Service.

Business logic layer for authentication operations using AWS Cognito.
Provides AWS-native authentication solution.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import logging
import secrets
from typing import TYPE_CHECKING, Any
import uuid

import boto3
from botocore.exceptions import ClientError

from clarity.models.auth import (
    UserLoginRequest,
    UserRegistrationRequest,
    AuthProvider,
    LoginResponse,
    RegistrationResponse,
    TokenResponse,
    UserRole,
    UserSessionResponse,
    UserStatus,
)
from clarity.services.dynamodb_service import DynamoDBService
from mypy_boto3_cognito_idp.type_defs import (
    AdminInitiateAuthResponseTypeDef,
    AttributeTypeTypeDef,
    GetUserResponseTypeDef,
    SignUpResponseTypeDef,
)
from clarity.ports.auth_ports import IAuthProvider
from mypy_boto3_cognito_idp import CognitoIdentityProviderClient

if TYPE_CHECKING:
    pass

# Configure logger
logger = logging.getLogger(__name__)

# Constants
BEARER_TOKEN_TYPE = "bearer"  # noqa: S105  # Standard OAuth token type


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


class CognitoAuthenticationService:
    """Authentication service using AWS Cognito.

    Provides AWS Cognito for user management, authentication, and authorization.
    """

    def __init__(
        self,
        auth_provider: IAuthProvider,
        dynamodb_service: DynamoDBService,
        user_pool_id: str,
        client_id: str,
        client_secret: str | None = None,
        region: str = "us-east-1",
        default_token_expiry: int = 3600,  # 1 hour
        refresh_token_expiry: int = 86400 * 30,  # 30 days
    ) -> None:
        """Initialize Cognito authentication service.

        Args:
            auth_provider: Authentication provider for token operations
            dynamodb_service: DynamoDB service for user data storage
            user_pool_id: Cognito User Pool ID
            client_id: Cognito App Client ID
            client_secret: Optional App Client Secret
            region: AWS region
            default_token_expiry: Default access token expiry in seconds
            refresh_token_expiry: Refresh token expiry in seconds
        """
        self.auth_provider = auth_provider
        self.dynamodb_service = dynamodb_service
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.region = region
        self.default_token_expiry = default_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        # Initialize Cognito client
        self.cognito_client: CognitoIdentityProviderClient = boto3.client(
            "cognito-idp", region_name=region
        )

        # Table names
        self.users_table = "clarity_users"
        self.sessions_table = "clarity_user_sessions"
        self.refresh_tokens_table = "clarity_refresh_tokens"

    def _compute_secret_hash(self, username: str) -> str | None:
        """Compute secret hash for Cognito API calls if client secret is configured."""
        if not self.client_secret:
            return None

        message = bytes(username + self.client_id, "utf-8")
        key = bytes(self.client_secret, "utf-8")
        return base64.b64encode(
            hmac.new(key, message, digestmod=hashlib.sha256).digest()
        ).decode()

    async def register_user(
        self,
        request: UserRegistrationRequest,
        device_info: dict[str, Any] | None = None,
    ) -> RegistrationResponse:
        """Register a new user with AWS Cognito.

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
            # Prepare Cognito parameters
            user_attributes: list[AttributeTypeTypeDef] = [
                {"Name": "email", "Value": request.email},
                {"Name": "given_name", "Value": request.first_name},
                {"Name": "family_name", "Value": request.last_name},
            ]

            if request.phone_number:
                user_attributes.append(
                    {"Name": "phone_number", "Value": request.phone_number}
                )

            # Create user in Cognito
            sign_up_params: dict[str, Any] = {
                "ClientId": self.client_id,
                "Username": request.email,
                "Password": request.password,
                "UserAttributes": user_attributes,
            }

            # Add secret hash if client secret is configured
            secret_hash = self._compute_secret_hash(request.email)
            if secret_hash:
                sign_up_params["SecretHash"] = secret_hash

            response: SignUpResponseTypeDef = self.cognito_client.sign_up(
                **sign_up_params
            )
            user_sub = response["UserSub"]

            # Generate user ID
            user_id = uuid.UUID(user_sub)

            # Store additional user data in DynamoDB
            user_data: dict[str, Any] = {
                "user_id": user_sub,
                "email": request.email,
                "first_name": request.first_name,
                "last_name": request.last_name,
                "phone_number": request.phone_number,
                "status": UserStatus.PENDING_VERIFICATION.value,
                "role": UserRole.PATIENT.value,
                "auth_provider": AuthProvider.EMAIL_PASSWORD.value,
                "mfa_enabled": False,
                "mfa_methods": [],
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "last_login": None,
                "login_count": 0,
                "email_verified": False,
                "terms_accepted": request.terms_accepted,
                "privacy_policy_accepted": request.privacy_policy_accepted,
                "device_info": device_info,
            }

            await self.dynamodb_service.put_item(
                table_name=self.users_table,
                item=user_data,
                user_id=user_sub,
            )

            logger.info("User registered successfully: %s", user_sub)

            return RegistrationResponse(
                user_id=user_id,
                email=request.email,
                status=UserStatus.PENDING_VERIFICATION,
                verification_email_sent=True,  # Cognito sends verification email automatically
                created_at=datetime.now(UTC),
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UsernameExistsException":
                msg = f"User with email {request.email} already exists"
                raise UserAlreadyExistsError(msg) from e
            logger.exception("Cognito registration failed for %s", request.email)
            msg = f"Registration failed: {e}"
            raise AuthenticationError(msg) from e
        except Exception as e:
            logger.exception("User registration failed for %s", request.email)
            msg = f"Registration failed: {e!s}"
            raise AuthenticationError(msg) from e

    async def login_user(
        self,
        request: UserLoginRequest,
        device_info: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        """Authenticate user and create session with Cognito.

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
            # Prepare authentication parameters
            auth_params: dict[str, str] = {
                "USERNAME": request.email,
                "PASSWORD": request.password,
            }

            # Add secret hash if client secret is configured
            secret_hash = self._compute_secret_hash(request.email)
            if secret_hash:
                auth_params["SECRET_HASH"] = secret_hash

            # Authenticate with Cognito
            response: AdminInitiateAuthResponseTypeDef = (
                self.cognito_client.admin_initiate_auth(
                    UserPoolId=self.user_pool_id,
                    ClientId=self.client_id,
                    AuthFlow="ADMIN_NO_SRP_AUTH",
                    AuthParameters=auth_params,
                )
            )

            # Handle different authentication challenges
            if "ChallengeName" in response:
                if response["ChallengeName"] == "NEW_PASSWORD_REQUIRED":
                    # Raise exception for password change requirement
                    logger.warning("User requires password change: %s", request.email)
                    msg = "Password change required. Please reset your password."
                    raise AuthenticationError(msg)
                if response["ChallengeName"] == "MFA_SETUP":
                    # Handle MFA setup
                    return self._handle_mfa_setup(
                        response, request, device_info, ip_address
                    )

            # Extract tokens
            auth_result = response["AuthenticationResult"]
            access_token = auth_result["AccessToken"]
            refresh_token = auth_result["RefreshToken"]

            # Get user info from Cognito
            user_info: GetUserResponseTypeDef = self.cognito_client.get_user(
                AccessToken=access_token
            )
            user_sub = next(
                attr["Value"]
                for attr in user_info["UserAttributes"]
                if attr["Name"] == "sub"
            )

            # Get user data from DynamoDB
            user_data = await self.dynamodb_service.get_item(
                table_name=self.users_table,
                key={"user_id": user_sub},
            )

            # Check user data after all Cognito operations
            user_data_exists = user_data is not None

            if not user_data_exists:
                # Handle missing user data by creating session without DynamoDB update
                logger.warning("User data not found in database for user: %s", user_sub)
                user_session = UserSessionResponse(
                    user_id=uuid.UUID(user_sub),
                    email=request.email,
                    first_name=request.email.split("@")[0],  # Default to email prefix
                    last_name="",  # Empty last name as default
                    role=UserRole.PATIENT.value,  # Default role
                    permissions=[],  # Default permissions
                    status=UserStatus.ACTIVE,
                    last_login=datetime.now(UTC),
                    mfa_enabled=False,
                    email_verified=True,  # Assume verified if logging in
                    created_at=datetime.now(UTC),
                )
            else:
                # Update user data
                login_time = datetime.now(UTC)
                await self.dynamodb_service.update_item(
                    table_name=self.users_table,
                    key={"user_id": user_sub},
                    update_expression="SET last_login = :login_time, login_count = login_count + :inc",
                    expression_attribute_values={
                        ":login_time": login_time.isoformat(),
                        ":inc": 1,
                    },
                    user_id=user_sub,
                )

                # Create session
                _session_id = await self._create_user_session(
                    user_sub,
                    refresh_token,
                    device_info,
                    ip_address,
                    remember_me=request.remember_me,
                )

                # Create user session response
                if user_data:
                    user_session = await self._create_user_session_response(
                        user_sub, user_data
                    )
                else:
                    # This shouldn't happen as we checked user_data_exists above
                    logger.error(
                        "User data is None when it should exist for user: %s", user_sub
                    )
                    user_session = UserSessionResponse(
                        user_id=uuid.UUID(user_sub),
                        email=request.email,
                        first_name=request.email.split("@")[0],
                        last_name="",
                        role=UserRole.PATIENT.value,
                        permissions=[],
                        status=UserStatus.ACTIVE,
                        last_login=datetime.now(UTC),
                        mfa_enabled=False,
                        email_verified=True,
                        created_at=datetime.now(UTC),
                    )

            # Create token response
            tokens = TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type=BEARER_TOKEN_TYPE,
                expires_in=self.default_token_expiry,
                scope="full_access",
            )

            logger.info("User logged in successfully: %s", user_sub)

            return LoginResponse(
                user=user_session,
                tokens=tokens,
                requires_mfa=False,
                mfa_session_token=None,
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UserNotFoundException":
                msg = f"User with email {request.email} not found"
                raise UserNotFoundError(msg) from e
            if error_code == "NotAuthorizedException":
                msg = "Invalid username or password"
                raise InvalidCredentialsError(msg) from e
            if error_code == "UserNotConfirmedException":
                msg = "Email verification required"
                raise EmailNotVerifiedError(msg) from e
            logger.exception("Login failed for %s", request.email)
            msg = f"Login failed: {e}"
            raise AuthenticationError(msg) from e
        except (UserNotFoundError, InvalidCredentialsError, EmailNotVerifiedError):
            raise
        except Exception as e:
            logger.exception("Login failed for %s", request.email)
            msg = f"Login failed: {e!s}"
            raise AuthenticationError(msg) from e

    async def _create_user_session(
        self,
        user_id: str,
        refresh_token: str,
        device_info: dict[str, Any] | None = None,
        ip_address: str | None = None,
        *,
        remember_me: bool = False,
    ) -> str:
        """Create a user session in DynamoDB."""
        session_id = str(uuid.uuid4())
        session_expiry = timedelta(days=30 if remember_me else 1)

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "refresh_token": refresh_token,
            "created_at": datetime.now(UTC).isoformat(),
            "last_activity": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + session_expiry).isoformat(),
            "device_info": device_info,
            "ip_address": ip_address,
            "is_active": True,
        }

        await self.dynamodb_service.put_item(
            table_name=self.sessions_table,
            item=session_data,
            user_id=user_id,
        )

        return session_id

    async def _create_user_session_response(
        self,
        user_id: str,
        user_data: dict[str, Any],
    ) -> UserSessionResponse:
        """Create user session response from user data."""
        return UserSessionResponse(
            user_id=uuid.UUID(user_id),
            email=user_data.get("email", ""),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            role=user_data.get("role", UserRole.PATIENT.value),
            permissions=user_data.get("permissions", []),
            status=UserStatus(user_data.get("status", UserStatus.ACTIVE.value)),
            last_login=(
                datetime.fromisoformat(user_data["last_login"])
                if user_data.get("last_login")
                else None
            ),
            mfa_enabled=user_data.get("mfa_enabled", False),
            email_verified=user_data.get("email_verified", False),
            created_at=(
                datetime.fromisoformat(user_data["created_at"])
                if user_data.get("created_at")
                else datetime.now(UTC)
            ),
        )

    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token with Cognito."""
        try:
            # Note: Refresh token flow doesn't require username, so no secret hash needed
            response: AdminInitiateAuthResponseTypeDef = (
                self.cognito_client.admin_initiate_auth(
                    UserPoolId=self.user_pool_id,
                    ClientId=self.client_id,
                    AuthFlow="REFRESH_TOKEN_AUTH",
                    AuthParameters={
                        "REFRESH_TOKEN": refresh_token,
                    },
                )
            )
            auth_result = response["AuthenticationResult"]

            return TokenResponse(
                access_token=auth_result["AccessToken"],
                refresh_token=refresh_token,  # Cognito doesn't rotate refresh tokens
                token_type=BEARER_TOKEN_TYPE,
                expires_in=self.default_token_expiry,
                scope="full_access",
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                msg = "Invalid refresh token"
                raise InvalidCredentialsError(msg) from e
            logger.exception("Token refresh failed")
            msg = f"Token refresh failed: {e}"
            raise AuthenticationError(msg) from e

    async def logout_user(self, refresh_token: str) -> bool:
        """Logout user by revoking tokens and ending session."""
        try:
            # Revoke the refresh token in Cognito
            self.cognito_client.revoke_token(
                Token=refresh_token,
                ClientId=self.client_id,
            )

            # Deactivate sessions in DynamoDB
            sessions = await self.dynamodb_service.query(
                table_name=self.sessions_table,
                key_condition_expression="refresh_token = :token",
                expression_attribute_values={":token": refresh_token},
            )

            for session in sessions.get("Items", []):
                await self.dynamodb_service.update_item(
                    table_name=self.sessions_table,
                    key={"session_id": session["session_id"]},
                    update_expression="SET is_active = :inactive, ended_at = :ended",
                    expression_attribute_values={
                        ":inactive": False,
                        ":ended": datetime.now(UTC).isoformat(),
                    },
                )

            logger.info("User logged out successfully")
            return True

        except Exception:
            logger.exception("Logout failed")
            return False

    async def get_user_by_id(self, user_id: str) -> UserSessionResponse | None:
        """Get user information by user ID."""
        try:
            # Get user data from DynamoDB
            user_data = await self.dynamodb_service.get_item(
                table_name=self.users_table,
                key={"user_id": user_id},
            )

            if not user_data:
                return None

            return await self._create_user_session_response(user_id, user_data)

        except Exception:
            logger.exception("Failed to get user %s", user_id)
            return None

    @staticmethod
    async def verify_email(verification_code: str) -> bool:
        """Verify user email with verification code.

        Note: With Cognito, email verification is handled through the
        confirm_sign_up API call, not through this service method.
        """
        logger.info(
            "Email verification should be handled through Cognito confirm_sign_up"
        )
        return True

    def _handle_mfa_setup(
        self,
        auth_response: AdminInitiateAuthResponseTypeDef,
        request: UserLoginRequest,
        device_info: dict[str, Any] | None,
        ip_address: str | None,
    ) -> LoginResponse:
        """Handle MFA setup challenge from Cognito."""
        # This is a placeholder for MFA handling
        # In a real implementation, you would handle the MFA setup flow
        mfa_session_token = secrets.token_urlsafe(32)

        # Return partial response requiring MFA
        return LoginResponse(
            user=UserSessionResponse(
                user_id=uuid.uuid4(),  # Placeholder
                email=request.email,
                first_name="",
                last_name="",
                role=UserRole.PATIENT.value,
                permissions=[],
                status=UserStatus.ACTIVE,
                last_login=None,
                mfa_enabled=True,
                email_verified=False,
                created_at=datetime.now(UTC),
            ),
            tokens=TokenResponse(
                access_token="",  # Empty for MFA flow
                refresh_token="",  # Empty for MFA flow
                token_type=BEARER_TOKEN_TYPE,
                expires_in=0,
                scope="mfa_pending",
            ),
            requires_mfa=True,
            mfa_session_token=mfa_session_token,
        )
