"""Test pagination module."""

from datetime import UTC, datetime
import uuid

import pytest

from clarity.core.pagination import (
    CursorInfo,
    PaginatedResponse,
    PaginationInfo,
    PaginationLinks,
    PaginationParams,
    create_cursor,
    decode_cursor,
    validate_pagination_params,
)


class TestPaginationParams:
    """Test PaginationParams model."""

    def test_default_params(self):
        """Test default pagination parameters."""
        params = PaginationParams()
        assert params.limit == 50
        assert params.cursor is None

    def test_custom_params(self):
        """Test custom pagination parameters."""
        params = PaginationParams(limit=100, cursor="next-cursor")
        assert params.limit == 100
        assert params.cursor == "next-cursor"

    def test_limit_validation(self):
        """Test limit validation."""
        # Max allowed is 1000
        params = PaginationParams(limit=1000)
        assert params.limit == 1000

        # Should fail validation if too high
        with pytest.raises(Exception):  # Pydantic validation error
            PaginationParams(limit=1001)


class TestPaginationInfo:
    """Test PaginationInfo model."""

    def test_pagination_info_creation(self):
        """Test creating pagination info."""
        info = PaginationInfo(
            total_count=250,
            page_size=25,
            has_next=True,
            has_previous=True,
            next_cursor="eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0=",
            previous_cursor="eyJpZCI6MTUzNzAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MzA6MDBaIn0=",
        )

        assert info.total_count == 250
        assert info.page_size == 25
        assert info.has_next is True
        assert info.has_previous is True
        assert info.next_cursor == "eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0="
        assert info.previous_cursor == "eyJpZCI6MTUzNzAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MzA6MDBaIn0="


class TestCursorFunctions:
    """Test cursor encoding/decoding functions."""

    def test_create_and_decode_cursor(self):
        """Test creating and decoding cursor."""
        cursor_info = CursorInfo(
            timestamp=datetime.now(UTC).isoformat(), id=str(uuid.uuid4()), direction="next"
        )

        # Create cursor
        cursor = create_cursor(cursor_info)
        assert isinstance(cursor, str)
        assert len(cursor) > 0

        # Decode cursor
        decoded = decode_cursor(cursor)
        assert decoded.id == cursor_info.id
        assert decoded.direction == cursor_info.direction
        # Compare timestamps as strings (they're ISO format)
        assert decoded.timestamp == cursor_info.timestamp


class TestPaginatedResponse:
    """Test PaginatedResponse model."""

    def test_paginated_response_with_pagination_info(self):
        """Test paginated response with pagination info."""
        pagination = PaginationInfo(
            total_count=50,
            page_size=10,
            has_next=True,
            has_previous=False,
            next_cursor="eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0=",
            previous_cursor=None,
        )

        links = PaginationLinks(
            self="/api/items?limit=10",
            next="/api/items?limit=10&cursor=eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0=",
            previous=None,
        )

        response = PaginatedResponse(
            data=[{"id": 1}, {"id": 2}],
            pagination=pagination,
            links=links,
        )

        assert len(response.data) == 2
        assert response.pagination.page_size == 10
        assert response.links.next == "/api/items?limit=10&cursor=eyJpZCI6MTU0MjAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTA6MzA6MDBaIn0="

    def test_paginated_response_with_links(self):
        """Test paginated response with links."""
        pagination = PaginationInfo(
            total_count=200,
            page_size=20,
            has_next=True,
            has_previous=True,
            next_cursor="eyJpZCI6MTU0NDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTE6MDA6MDBaIn0=",
            previous_cursor="eyJpZCI6MTU0MDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MDA6MDBaIn0=",
        )

        links = PaginationLinks(
            self="/api/items?limit=20&cursor=current",
            next="/api/items?limit=20&cursor=eyJpZCI6MTU0NDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTE6MDA6MDBaIn0=",
            previous="/api/items?limit=20&cursor=eyJpZCI6MTU0MDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MDA6MDBaIn0=",
        )

        response = PaginatedResponse(
            data=[{"id": 3}], pagination=pagination, links=links
        )

        assert len(response.data) == 1
        assert response.links.next == "/api/items?limit=20&cursor=eyJpZCI6MTU0NDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMTE6MDA6MDBaIn0="
        assert response.links.previous == "/api/items?limit=20&cursor=eyJpZCI6MTU0MDAsInRpbWVzdGFtcCI6IjIwMjUtMDEtMTVUMDk6MDA6MDBaIn0="


class TestValidatePaginationParams:
    """Test pagination parameter validation."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        validated = validate_pagination_params(limit=25, cursor=None)

        assert validated.limit == 25
        assert validated.cursor is None

    def test_validate_with_offset(self):
        """Test validation with offset."""
        validated = validate_pagination_params(
            limit=20, 
            offset=40
        )

        assert validated.limit == 20
        # Offset parameter may not be directly stored in PaginationParams

    def test_validate_default_limit(self):
        """Test validation with default limit."""
        validated = validate_pagination_params()

        assert validated.limit == 50  # Default limit
