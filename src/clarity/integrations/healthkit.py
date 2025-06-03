"""
Apple HealthKit Integration for Clarity Digital Twin

Handles OAuth 2.0 authorization, data fetching, and normalization
for Apple Watch health metrics including activity, sleep, and heart rate.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from pydantic import BaseModel, Field, validator

from clarity.core.config import settings
from clarity.core.logging import get_logger
from clarity.core.exceptions import (
    IntegrationError,
    AuthorizationError,
    DataValidationError
)

logger = get_logger(__name__)


class HealthDataType(str, Enum):
    """Supported Apple HealthKit data types"""
    HEART_RATE = "heart_rate"
    STEPS = "steps"
    ACTIVE_ENERGY = "active_energy"
    SLEEP_ANALYSIS = "sleep_analysis"
    WORKOUT = "workout"
    RESTING_HEART_RATE = "resting_heart_rate"
    HEART_RATE_VARIABILITY = "heart_rate_variability"


class HealthKitAuthScope(str, Enum):
    """HealthKit authorization scopes"""
    READ_HEART_RATE = "https://www.healthkit.apple.com/heart_rate"
    READ_STEPS = "https://www.healthkit.apple.com/steps"
    READ_ACTIVE_ENERGY = "https://www.healthkit.apple.com/active_energy"
    READ_SLEEP = "https://www.healthkit.apple.com/sleep_analysis"
    READ_WORKOUT = "https://www.healthkit.apple.com/workout"
    READ_HRV = "https://www.healthkit.apple.com/heart_rate_variability"


@dataclass
class HealthDataPoint:
    """Individual health data measurement"""
    timestamp: datetime
    value: Union[float, int, str]
    unit: str
    source: str = "apple_watch"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthDataBatch(BaseModel):
    """Batch of health data for processing"""
    user_id: str
    data_type: HealthDataType
    start_date: datetime
    end_date: datetime
    data_points: List[HealthDataPoint]
    total_count: int = Field(ge=0)
    
    @validator('data_points')
    def validate_data_points(cls, v):
        if not v:
            raise ValueError("Data points cannot be empty")
        return v
    
    @validator('total_count')
    def validate_count_matches(cls, v, values):
        if 'data_points' in values and v != len(values['data_points']):
            raise ValueError("Total count must match data points length")
        return v


class HealthKitTokens(BaseModel):
    """HealthKit OAuth tokens"""
    access_token: str
    refresh_token: str
    expires_at: datetime
    scope: List[str]
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at


class HealthKitClient:
    """
    Apple HealthKit API client for fetching health data
    
    Handles OAuth 2.0 authentication flow and data retrieval
    from Apple's HealthKit servers.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        base_url: str = "https://www.healthkit.apple.com",
    ):
        self.client_id = client_id or settings.APPLE_HEALTHKIT_CLIENT_ID
        self.client_secret = client_secret or settings.APPLE_HEALTHKIT_CLIENT_SECRET
        self.redirect_uri = redirect_uri or settings.APPLE_HEALTHKIT_REDIRECT_URI
        self.base_url = base_url
        
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "User-Agent": f"Clarity-Digital-Twin/{settings.VERSION}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        
        logger.info("HealthKit client initialized", extra={
            "client_id": self.client_id[:8] + "..." if self.client_id else None,
            "base_url": self.base_url
        })
    
    async def get_authorization_url(
        self,
        state: str,
        scopes: List[HealthKitAuthScope]
    ) -> str:
        """
        Generate OAuth 2.0 authorization URL for HealthKit
        
        Args:
            state: CSRF protection state parameter
            scopes: List of HealthKit permissions to request
            
        Returns:
            Authorization URL for user to visit
        """
        scope_string = " ".join(scopes)
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope_string,
            "state": state,
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        auth_url = f"{self.base_url}/oauth/authorize?{query_string}"
        
        logger.info("Generated HealthKit authorization URL", extra={
            "state": state,
            "scopes": scopes,
            "redirect_uri": self.redirect_uri
        })
        
        return auth_url
    
    async def exchange_code_for_tokens(
        self,
        authorization_code: str,
        state: str
    ) -> HealthKitTokens:
        """
        Exchange authorization code for access tokens
        
        Args:
            authorization_code: Code received from OAuth callback
            state: State parameter for CSRF validation
            
        Returns:
            HealthKit access and refresh tokens
            
        Raises:
            AuthorizationError: If token exchange fails
        """
        try:
            response = await self._http_client.post(
                f"{self.base_url}/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": authorization_code,
                    "redirect_uri": self.redirect_uri,
                }
            )
            
            if response.status_code != 200:
                raise AuthorizationError(
                    f"Token exchange failed: {response.status_code} {response.text}"
                )
            
            token_data = response.json()
            
            tokens = HealthKitTokens(
                access_token=token_data["access_token"],
                refresh_token=token_data["refresh_token"],
                expires_at=datetime.now() + timedelta(seconds=token_data["expires_in"]),
                scope=token_data.get("scope", "").split(" ")
            )
            
            logger.info("Successfully exchanged authorization code for tokens", extra={
                "expires_at": tokens.expires_at.isoformat(),
                "scopes": tokens.scope
            })
            
            return tokens
            
        except Exception as e:
            logger.error("Failed to exchange authorization code", extra={
                "error": str(e),
                "code_preview": authorization_code[:8] + "..." if authorization_code else None
            })
            raise AuthorizationError(f"Token exchange failed: {e}") from e
    
    async def refresh_access_token(self, refresh_token: str) -> HealthKitTokens:
        """
        Refresh expired access token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access and refresh tokens
            
        Raises:
            AuthorizationError: If refresh fails
        """
        try:
            response = await self._http_client.post(
                f"{self.base_url}/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": refresh_token,
                }
            )
            
            if response.status_code != 200:
                raise AuthorizationError(
                    f"Token refresh failed: {response.status_code} {response.text}"
                )
            
            token_data = response.json()
            
            tokens = HealthKitTokens(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token", refresh_token),
                expires_at=datetime.now() + timedelta(seconds=token_data["expires_in"]),
                scope=token_data.get("scope", "").split(" ")
            )
            
            logger.info("Successfully refreshed access token")
            return tokens
            
        except Exception as e:
            logger.error("Failed to refresh access token", extra={"error": str(e)})
            raise AuthorizationError(f"Token refresh failed: {e}") from e
    
    async def fetch_health_data(
        self,
        tokens: HealthKitTokens,
        data_type: HealthDataType,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> HealthDataBatch:
        """
        Fetch health data from HealthKit API
        
        Args:
            tokens: Valid HealthKit tokens
            data_type: Type of health data to fetch
            start_date: Start of data range
            end_date: End of data range
            limit: Maximum number of data points
            
        Returns:
            Batch of health data points
            
        Raises:
            IntegrationError: If data fetch fails
            DataValidationError: If data is invalid
        """
        if tokens.is_expired:
            raise AuthorizationError("Access token has expired")
        
        try:
            # Map our data types to HealthKit API endpoints
            endpoint_map = {
                HealthDataType.HEART_RATE: "/v1/healthkit/heart_rate",
                HealthDataType.STEPS: "/v1/healthkit/steps",
                HealthDataType.ACTIVE_ENERGY: "/v1/healthkit/active_energy",
                HealthDataType.SLEEP_ANALYSIS: "/v1/healthkit/sleep",
                HealthDataType.WORKOUT: "/v1/healthkit/workouts",
                HealthDataType.RESTING_HEART_RATE: "/v1/healthkit/resting_heart_rate",
                HealthDataType.HEART_RATE_VARIABILITY: "/v1/healthkit/hrv",
            }
            
            endpoint = endpoint_map.get(data_type)
            if not endpoint:
                raise DataValidationError(f"Unsupported data type: {data_type}")
            
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "limit": limit,
            }
            
            response = await self._http_client.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {tokens.access_token}"}
            )
            
            if response.status_code != 200:
                raise IntegrationError(
                    f"HealthKit API error: {response.status_code} {response.text}"
                )
            
            raw_data = response.json()
            
            # Transform HealthKit data to our format
            data_points = []
            for item in raw_data.get("data", []):
                point = HealthDataPoint(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    value=item["value"],
                    unit=item.get("unit", ""),
                    source=item.get("source", "apple_watch"),
                    metadata=item.get("metadata")
                )
                data_points.append(point)
            
            batch = HealthDataBatch(
                user_id=raw_data.get("user_id", "unknown"),
                data_type=data_type,
                start_date=start_date,
                end_date=end_date,
                data_points=data_points,
                total_count=len(data_points)
            )
            
            logger.info("Successfully fetched health data", extra={
                "data_type": data_type,
                "count": len(data_points),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            return batch
            
        except Exception as e:
            logger.error("Failed to fetch health data", extra={
                "data_type": data_type,
                "error": str(e),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            raise IntegrationError(f"Health data fetch failed: {e}") from e
    
    async def fetch_latest_7_days(
        self,
        tokens: HealthKitTokens,
        data_types: List[HealthDataType]
    ) -> Dict[HealthDataType, HealthDataBatch]:
        """
        Fetch latest 7 days of data for multiple data types
        
        This is the main method for the demo - gets all the data we need
        to show impressive AI insights.
        
        Args:
            tokens: Valid HealthKit tokens
            data_types: List of data types to fetch
            
        Returns:
            Dictionary mapping data types to batches
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        tasks = []
        for data_type in data_types:
            task = self.fetch_health_data(
                tokens=tokens,
                data_type=data_type,
                start_date=start_date,
                end_date=end_date
            )
            tasks.append((data_type, task))
        
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )
        
        for (data_type, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {data_type}: {result}")
                # Continue with other data types
                continue
            else:
                results[data_type] = result
        
        logger.info("Completed 7-day data fetch", extra={
            "successful_types": list(results.keys()),
            "total_requested": len(data_types)
        })
        
        return results
    
    async def close(self):
        """Clean up HTTP client"""
        await self._http_client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()