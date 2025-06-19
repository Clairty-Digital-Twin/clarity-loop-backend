"""🚀 Professional Pagination System - Enterprise-Grade API Responses.

Comprehensive pagination with cursor-based and offset-based support.
Designed to scale from thousands to millions of records.
"""

# removed - breaks FastAPI

import base64
import json
from typing import Any, ClassVar, Generic, TypeVar
from urllib.parse import urlencode

from pydantic import BaseModel, Field, validator

T = TypeVar("T")

# Constants
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000


class PaginationInfo(BaseModel):
    """Pagination metadata for API responses."""

    total_count: int | None = Field(
        None,
        description="Total number of items (if efficiently calculable)",
        examples=[15420],
    )
    page_size: int = Field(
        ..., description="Number of items per page", examples=[50], ge=1, le=1000
    )
    has_next: bool = Field(
        ..., description="Whether there are more items available", examples=[True]
    )
    has_previous: bool = Field(
        ..., description="Whether there are previous items available", examples=[False]
    )
    next_cursor: str | None = Field(
        None,
        description="Cursor for fetching the next page",
        examples=["eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0="],
    )
    previous_cursor: str | None = Field(
        None,
        description="Cursor for fetching the previous page",
        examples=["eyJpZCI6MTUzNzAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MzA6MDBaIn0="],
    )


class PaginationLinks(BaseModel):
    """HAL-style pagination links for navigation."""

    self: str = Field(
        ...,
        description="Link to current page",
        examples=["https://api.clarity.health/health-data?limit=50&cursor=abc123"],
    )
    first: str | None = Field(
        None,
        description="Link to first page",
        examples=["https://api.clarity.health/health-data?limit=50"],
    )
    previous: str | None = Field(
        None,
        description="Link to previous page",
        examples=["https://api.clarity.health/health-data?limit=50&cursor=xyz789"],
    )
    next: str | None = Field(
        None,
        description="Link to next page",
        examples=["https://api.clarity.health/health-data?limit=50&cursor=def456"],
    )
    last: str | None = Field(
        None,
        description="Link to last page (if total count is known)",
        examples=["https://api.clarity.health/health-data?limit=50&cursor=last123"],
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """🔥 Enterprise-grade paginated API response.

    Follows REST best practices with HAL-style links and comprehensive metadata.
    """

    data: list[T] = Field(..., description="Array of items for this page")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")
    links: PaginationLinks = Field(
        ..., description="Navigation links following HAL specification"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar[dict[str, object]] = {
            # Custom encoders if needed
        }


class PaginationParams(BaseModel):
    """Query parameters for pagination requests."""

    limit: int = Field(
        50,
        description="Number of items to return per page",
        ge=1,
        le=1000,
        examples=[50],
    )
    cursor: str | None = Field(
        None,
        description="Pagination cursor for next/previous page",
        examples=["eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0="],
    )

    # Offset-based pagination (alternative to cursor)
    offset: int | None = Field(
        None,
        description="Number of items to skip (alternative to cursor)",
        ge=0,
        examples=[100],
    )

    @validator("limit")
    def validate_limit(cls, v: int) -> int:  # noqa: N805
        """Ensure reasonable page size limits."""
        min_limit_msg = "Limit must be at least 1"
        max_limit_msg = f"Limit cannot exceed {MAX_PAGE_SIZE}"

        if v < 1:
            raise ValueError(min_limit_msg)
        if v > MAX_PAGE_SIZE:
            raise ValueError(max_limit_msg)
        return v


class CursorInfo(BaseModel):
    """Information encoded in pagination cursor."""

    id: str | None = None
    timestamp: str | None = None
    sort_key: str | None = None
    direction: str = "next"  # "next" or "previous"


class PaginationBuilder:
    """🔧 Builder for creating paginated responses with proper links."""

    def __init__(self, base_url: str, endpoint: str) -> None:
        """Initialize pagination builder.

        Args:
            base_url: Base API URL (e.g., "https://api.clarity.health")
            endpoint: Endpoint path (e.g., "/api/v1/health-data")
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint

    def build_response(
        self,
        data: list[T],
        params: PaginationParams,
        *,
        has_next: bool,
        has_previous: bool,
        total_count: int | None = None,
        next_cursor: str | None = None,
        previous_cursor: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> PaginatedResponse[T]:
        """Build complete paginated response with links and metadata.

        Args:
            data: Items for current page
            params: Original pagination parameters
            has_next: Whether more items exist after current page
            has_previous: Whether items exist before current page
            total_count: Total number of items (if known)
            next_cursor: Cursor for next page
            previous_cursor: Cursor for previous page
            additional_params: Additional query parameters to preserve
        """
        # Build pagination info
        pagination = PaginationInfo(
            total_count=total_count,
            page_size=params.limit,
            has_next=has_next,
            has_previous=has_previous,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
        )

        # Build navigation links
        links = self._build_links(
            params=params,
            has_next=has_next,
            has_previous=has_previous,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            additional_params=additional_params,
        )

        return PaginatedResponse(data=data, pagination=pagination, links=links)

    def _build_links(
        self,
        params: PaginationParams,
        *,
        has_next: bool,
        has_previous: bool,
        next_cursor: str | None,
        previous_cursor: str | None,
        additional_params: dict[str, Any] | None = None,
    ) -> PaginationLinks:
        """Build HAL-style navigation links."""
        base_params = additional_params or {}
        base_params["limit"] = params.limit

        # Current page link
        current_params = base_params.copy()
        if params.cursor:
            current_params["cursor"] = params.cursor
        elif params.offset:
            current_params["offset"] = params.offset

        self_link = f"{self.base_url}{self.endpoint}?{urlencode(current_params)}"

        # First page link
        first_params = base_params.copy()
        first_link = f"{self.base_url}{self.endpoint}?{urlencode(first_params)}"

        # Next page link
        next_link = None
        if has_next and next_cursor:
            next_params = base_params.copy()
            next_params["cursor"] = next_cursor
            next_link = f"{self.base_url}{self.endpoint}?{urlencode(next_params)}"
        elif has_next and params.offset is not None:
            next_params = base_params.copy()
            next_params["offset"] = params.offset + params.limit
            next_link = f"{self.base_url}{self.endpoint}?{urlencode(next_params)}"

        # Previous page link
        previous_link = None
        if has_previous and previous_cursor:
            prev_params = base_params.copy()
            prev_params["cursor"] = previous_cursor
            previous_link = f"{self.base_url}{self.endpoint}?{urlencode(prev_params)}"
        elif has_previous and params.offset is not None and params.offset > 0:
            prev_params = base_params.copy()
            prev_params["offset"] = max(0, params.offset - params.limit)
            previous_link = f"{self.base_url}{self.endpoint}?{urlencode(prev_params)}"

        return PaginationLinks(
            self=self_link,
            first=first_link,
            next=next_link,
            previous=previous_link,
            last=None,  # Can be computed if total_count is available
        )


def create_cursor(cursor_info: CursorInfo) -> str:
    """Create base64-encoded cursor from cursor information.

    Args:
        cursor_info: Information to encode in cursor

    Returns:
        Base64-encoded cursor string
    """
    cursor_data = cursor_info.dict(exclude_none=True)
    cursor_json = json.dumps(cursor_data, sort_keys=True)
    cursor_bytes = cursor_json.encode("utf-8")
    return base64.b64encode(cursor_bytes).decode("utf-8")


def decode_cursor(cursor: str) -> CursorInfo:
    """Decode base64 cursor back to cursor information.

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Decoded cursor information

    Raises:
        ValueError: If cursor is invalid
    """
    try:
        cursor_bytes = base64.b64decode(cursor.encode("utf-8"))
        cursor_json = cursor_bytes.decode("utf-8")
        cursor_data = json.loads(cursor_json)
        return CursorInfo(**cursor_data)
    except Exception as e:
        error_msg = f"Invalid cursor format: {e}"
        raise ValueError(error_msg) from e


# 🎯 Common pagination utilities

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000


def validate_pagination_params(
    limit: int | None = None, cursor: str | None = None, offset: int | None = None
) -> PaginationParams:
    """Validate and normalize pagination parameters.

    Args:
        limit: Page size limit
        cursor: Pagination cursor
        offset: Offset for offset-based pagination

    Returns:
        Validated pagination parameters

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate limit
    if limit is None:
        limit = DEFAULT_PAGE_SIZE
    elif limit < 1:
        min_limit_msg = "Limit must be at least 1"
        raise ValueError(min_limit_msg)
    elif limit > MAX_PAGE_SIZE:
        max_limit_msg = f"Limit cannot exceed {MAX_PAGE_SIZE}"
        raise ValueError(max_limit_msg)

    # Validate cursor if provided
    if cursor:
        try:
            decode_cursor(cursor)
        except ValueError as e:
            invalid_cursor_msg = f"Invalid cursor: {e}"
            raise ValueError(invalid_cursor_msg) from e

    # Validate offset if provided
    if offset is not None and offset < 0:
        offset_msg = "Offset must be non-negative"
        raise ValueError(offset_msg)

    return PaginationParams(limit=limit, cursor=cursor, offset=offset)
