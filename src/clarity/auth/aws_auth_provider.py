"""AWS Cognito authentication provider implementation."""

import json
import logging
import time
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
import httpx
import jwt

from clarity.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
)
from clarity.models.auth import UserContext
from clarity.ports.auth_ports import IAuthProvider

logger = logging.getLogger(__name__)


class AWSCognitoAuthProvider(IAuthProvider):
    """AWS Cognito authentication provider."""

    def __init__(
        self,
        region: str,
        user_pool_id: str,
        client_id: str,
        skip_validation: bool = False,
    ):
        self.region = region
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.skip_validation = skip_validation

        # Cognito public keys URL
        self.jwks_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._cache_duration = 3600  # 1 hour

    async def _get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Cognito with caching."""
        current_time = time.time()

        if (
            self._jwks_cache is None
            or current_time - self._jwks_cache_time > self._cache_duration
        ):

            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                self._jwks_cache = response.json()
                self._jwks_cache_time = current_time

        return self._jwks_cache

    def _get_public_key(self, token: str, jwks: dict[str, Any]) -> RSAPublicKey:
        """Extract public key from JWKS based on token's kid."""
        try:
            # Decode header without verification to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            if not kid:
                raise AuthenticationError("Token missing 'kid' header")

            # Find the key with matching kid
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    # Build RSA public key
                    return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))

            raise AuthenticationError(f"Public key not found for kid: {kid}")

        except Exception as e:
            logger.error(f"Error extracting public key: {e}")
            raise AuthenticationError("Invalid token format") from e

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify Cognito JWT token and extract user context."""
        if self.skip_validation:
            # For testing/development
            return {
                "uid": "test-user-123",
                "email": "test@example.com",
                "email_verified": True,
                "name": "Test User",
                "provider": "cognito",
                "custom_claims": {},
            }

        try:
            # Get JWKS
            jwks = await self._get_jwks()

            # Get public key for this token
            public_key = self._get_public_key(token, jwks)

            # Verify and decode token
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer=self.issuer,
                options={"verify_exp": True},
            )

            # Extract user context from Cognito claims
            return {
                "uid": decoded.get("sub", ""),
                "email": decoded.get("email", ""),
                "email_verified": decoded.get("email_verified", False),
                "name": decoded.get("name") or decoded.get("cognito:username", ""),
                "provider": "cognito",
                "custom_claims": {
                    k: v
                    for k, v in decoded.items()
                    if k not in ["sub", "email", "email_verified", "name"]
                },
            }

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e!s}")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise AuthenticationError("Token verification failed")

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user details from Cognito.

        Note: This would require AWS SDK (boto3) for admin operations.
        For now, returning None as this is primarily used for token verification.
        """
        # In a full implementation, you would use boto3 to get user attributes
        return None

    async def initialize(self) -> None:
        """Initialize the authentication provider."""
        # Pre-fetch JWKS to validate connection
        if not self.skip_validation:
            try:
                await self._get_jwks()
                logger.info("AWS Cognito auth provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Cognito auth provider: {e}")
                raise

    async def cleanup(self) -> None:
        """Cleanup the authentication provider."""
        # Clear JWKS cache
        self._jwks_cache = None
        self._jwks_cache_time = 0

    async def create_custom_token(
        self, uid: str, claims: dict[str, Any] | None = None
    ) -> str:
        """Create custom token (not typically used with Cognito)."""
        raise NotImplementedError("Custom tokens not supported with Cognito")

    async def revoke_refresh_tokens(self, uid: str) -> None:
        """Revoke user's refresh tokens.

        Note: This would require AWS SDK (boto3) for admin operations.
        """
        # In a full implementation, you would use boto3 to revoke tokens

    async def verify_session_cookie(self, session_cookie: str) -> UserContext:
        """Verify session cookie (not typically used with Cognito)."""
        raise NotImplementedError("Session cookies not supported with Cognito")
