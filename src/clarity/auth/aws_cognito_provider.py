"""AWS Cognito Authentication Provider"""

from functools import lru_cache
import logging
import os
import time
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from jose import JWTError, jwk, jwt
from jose.utils import base64url_decode
import requests

from clarity.core.exceptions import AuthenticationError
from clarity.models.user import User
from clarity.ports.auth_ports import IAuthProvider

logger = logging.getLogger(__name__)


class CognitoAuthProvider(IAuthProvider):
    """AWS Cognito authentication provider"""

    def __init__(self, user_pool_id: str, client_id: str, region: str = "us-east-1"):
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.region = region
        self.cognito_client = boto3.client("cognito-idp", region_name=region)

        # Cognito URLs
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        self.jwks_url = f"{self.issuer}/.well-known/jwks.json"

        # Cache for JWKS keys
        self._jwks_cache = None
        self._jwks_cache_time = 0
        self._jwks_cache_ttl = 3600  # 1 hour

        logger.info(f"Initialized Cognito auth provider for pool: {user_pool_id}")

    @property
    def jwks(self) -> dict[str, Any]:
        """Get JWKS keys with caching"""
        current_time = time.time()
        if (
            self._jwks_cache is None
            or current_time - self._jwks_cache_time > self._jwks_cache_ttl
        ):
            try:
                response = requests.get(self.jwks_url)
                response.raise_for_status()
                self._jwks_cache = response.json()
                self._jwks_cache_time = current_time
                logger.debug("Updated JWKS cache")
            except Exception as e:
                logger.error(f"Failed to fetch JWKS: {e}")
                if self._jwks_cache is None:
                    raise AuthenticationError("Failed to fetch JWKS keys")
        return self._jwks_cache

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify Cognito JWT token"""
        try:
            # Get the kid from the headers prior to verification
            headers = jwt.get_unverified_headers(token)
            kid = headers["kid"]

            # Search for the kid in the downloaded public keys
            key_index = -1
            for i in range(len(self.jwks["keys"])):
                if kid == self.jwks["keys"][i]["kid"]:
                    key_index = i
                    break

            if key_index == -1:
                logger.error("Public key not found in jwks.json")
                return None

            # Construct the public key
            public_key = jwk.construct(self.jwks["keys"][key_index])

            # Get the last two sections of the token,
            # message and signature (encoded in base64)
            message, encoded_signature = str(token).rsplit(".", 1)

            # Decode the signature
            decoded_signature = base64url_decode(encoded_signature.encode("utf-8"))

            # Verify the signature
            if not public_key.verify(message.encode("utf8"), decoded_signature):
                logger.error("Signature verification failed")
                return None

            # Since we passed the verification, we can now safely
            # use the unverified claims
            claims = jwt.get_unverified_claims(token)

            # Additionally we can verify the token expiration
            if time.time() > claims["exp"]:
                logger.error("Token is expired")
                return None

            # And the Audience  (use claims['client_id'] if verifying an access token)
            if (
                claims.get("aud") != self.client_id
                and claims.get("client_id") != self.client_id
            ):
                logger.error("Token was not issued for this audience")
                return None

            # Now we can use the claims
            logger.debug(f"Token verified for user: {claims.get('sub')}")
            return claims

        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during token verification: {e}")
            return None

    async def get_user(self, uid: str) -> User | None:
        """Get user details from Cognito"""
        try:
            response = self.cognito_client.admin_get_user(
                UserPoolId=self.user_pool_id, Username=uid
            )

            # Extract user attributes
            attributes = {
                attr["Name"]: attr["Value"]
                for attr in response.get("UserAttributes", [])
            }

            # Create User object
            user = User(
                uid=uid,
                email=attributes.get("email", ""),
                display_name=attributes.get("name", attributes.get("email", uid)),
                created_at=response.get("UserCreateDate"),
                last_login=response.get("UserLastModifiedDate"),
                metadata={
                    "username": response.get("Username"),
                    "status": response.get("UserStatus"),
                    "enabled": response.get("Enabled", True),
                    "attributes": attributes,
                },
            )

            logger.debug(f"Retrieved user: {uid}")
            return user

        except ClientError as e:
            if e.response["Error"]["Code"] == "UserNotFoundException":
                logger.warning(f"User not found: {uid}")
                return None
            logger.error(f"Failed to get user: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting user: {e}")
            return None

    async def create_user(
        self, email: str, password: str, display_name: str | None = None
    ) -> User | None:
        """Create a new user in Cognito"""
        try:
            # Create user
            response = self.cognito_client.sign_up(
                ClientId=self.client_id,
                Username=email,
                Password=password,
                UserAttributes=[
                    {"Name": "email", "Value": email},
                    {"Name": "name", "Value": display_name or email},
                ],
            )

            user_sub = response["UserSub"]

            # Auto-confirm user in development
            if os.getenv("ENVIRONMENT") == "development":
                self.cognito_client.admin_confirm_sign_up(
                    UserPoolId=self.user_pool_id, Username=email
                )
                logger.info(f"Auto-confirmed user: {email}")

            # Return user object
            return User(
                uid=user_sub,
                email=email,
                display_name=display_name or email,
                created_at=None,
                last_login=None,
                metadata={"username": email, "status": "CONFIRMED"},
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UsernameExistsException":
                logger.warning(f"User already exists: {email}")
                raise AuthenticationError("User already exists")
            logger.error(f"Failed to create user: {e}")
            raise AuthenticationError(f"Failed to create user: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error creating user: {e}")
            raise AuthenticationError(f"Unexpected error: {e!s}")

    async def delete_user(self, uid: str) -> bool:
        """Delete a user from Cognito"""
        try:
            self.cognito_client.admin_delete_user(
                UserPoolId=self.user_pool_id, Username=uid
            )
            logger.info(f"Deleted user: {uid}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete user: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting user: {e}")
            return False

    async def update_user(self, uid: str, **kwargs) -> User | None:
        """Update user attributes in Cognito"""
        try:
            # Build attributes list
            attributes = []
            if "display_name" in kwargs:
                attributes.append({"Name": "name", "Value": kwargs["display_name"]})
            if "email" in kwargs:
                attributes.append({"Name": "email", "Value": kwargs["email"]})

            if attributes:
                self.cognito_client.admin_update_user_attributes(
                    UserPoolId=self.user_pool_id,
                    Username=uid,
                    UserAttributes=attributes,
                )
                logger.info(f"Updated user attributes: {uid}")

            # Return updated user
            return await self.get_user(uid)

        except ClientError as e:
            logger.error(f"Failed to update user: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error updating user: {e}")
            return None

    async def authenticate(self, email: str, password: str) -> dict[str, str] | None:
        """Authenticate user and return tokens"""
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": email, "PASSWORD": password},
            )

            if "AuthenticationResult" in response:
                result = response["AuthenticationResult"]
                logger.info(f"User authenticated: {email}")
                return {
                    "access_token": result["AccessToken"],
                    "id_token": result["IdToken"],
                    "refresh_token": result["RefreshToken"],
                    "expires_in": result["ExpiresIn"],
                }

            # Handle challenges (MFA, etc)
            if "ChallengeName" in response:
                logger.warning(
                    f"Authentication challenge required: {response['ChallengeName']}"
                )
                raise AuthenticationError(
                    f"Challenge required: {response['ChallengeName']}"
                )

            return None

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                logger.warning(f"Invalid credentials for: {email}")
                raise AuthenticationError("Invalid email or password")
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected authentication error: {e}")
            raise AuthenticationError(f"Unexpected error: {e!s}")

    async def initialize(self) -> None:
        """Initialize Cognito provider"""
        try:
            # Test connection by fetching JWKS
            _ = self.jwks
            logger.info("Cognito provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cognito provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._jwks_cache = None
        logger.info("Cognito provider shutdown complete")

    async def cleanup(self) -> None:
        """Cleanup resources - implements IAuthProvider interface"""
        await self.shutdown()

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by ID - implements IAuthProvider interface"""
        user = await self.get_user(user_id)
        if user:
            return {
                "uid": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "created_at": user.created_at,
                "metadata": user.metadata,
            }
        return None


@lru_cache(maxsize=1)
def get_cognito_provider() -> CognitoAuthProvider:
    """Get singleton Cognito provider instance"""
    user_pool_id = os.getenv("COGNITO_USER_POOL_ID", "")
    client_id = os.getenv("COGNITO_CLIENT_ID", "")
    region = os.getenv("COGNITO_REGION", os.getenv("AWS_REGION", "us-east-1"))

    if not user_pool_id or not client_id:
        raise ValueError("Cognito configuration missing")

    return CognitoAuthProvider(user_pool_id, client_id, region)
