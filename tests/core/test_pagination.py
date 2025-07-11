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
            page=2,
            page_size=25,
            total_pages=10,
            total_items=250,
            has_next=True,
            has_previous=True,
        )

        assert info.page == 2
        assert info.page_size == 25
        assert info.total_pages == 10
        assert info.total_items == 250
        assert info.has_next is True
        assert info.has_previous is True


class TestCursorFunctions:
    """Test cursor encoding/decoding functions."""

    def test_create_and_decode_cursor(self):
        """Test creating and decoding cursor."""
        cursor_info = CursorInfo(
            timestamp=datetime.now(UTC), id=str(uuid.uuid4()), direction="next"
        )

        # Create cursor
        cursor = create_cursor(cursor_info)
        assert isinstance(cursor, str)
        assert len(cursor) > 0

        # Decode cursor
        decoded = decode_cursor(cursor)
        assert decoded.id == cursor_info.id
        assert decoded.direction == cursor_info.direction
        assert decoded.timestamp.replace(
            microsecond=0
        ) == cursor_info.timestamp.replace(microsecond=0)


class TestPaginatedResponse:
    """Test PaginatedResponse model."""

    def test_paginated_response_with_pagination_info(self):
        """Test paginated response with pagination info."""
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_pages=5,
            total_items=50,
            has_next=True,
            has_previous=False,
        )

        response = PaginatedResponse(
            data=[{"id": 1}, {"id": 2}],
            pagination=pagination,
            meta={"query_time": 0.123},
        )

        assert len(response.data) == 2
        assert response.pagination.page == 1
        assert response.meta["query_time"] == 0.123

    def test_paginated_response_with_links(self):
        """Test paginated response with links."""
        pagination = PaginationInfo(
            page=2,
            page_size=20,
            total_pages=10,
            total_items=200,
            has_next=True,
            has_previous=True,
        )

        links = PaginationLinks(
            self="/api/items?page=2",
            first="/api/items?page=1",
            last="/api/items?page=10",
            next="/api/items?page=3",
            previous="/api/items?page=1",
        )

        response = PaginatedResponse(
            data=[{"id": 3}], pagination=pagination, links=links
        )

        assert len(response.data) == 1
        assert response.links.next == "/api/items?page=3"
        assert response.links.previous == "/api/items?page=1"


class TestValidatePaginationParams:
    """Test pagination parameter validation."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        params = PaginationParams(page=5, page_size=25)
        validated = validate_pagination_params(params)

        assert validated.page == 5
        assert validated.page_size == 25

    def test_validate_invalid_page(self):
        """Test validation with invalid page."""
        params = PaginationParams(page=-1, page_size=20)
        validated = validate_pagination_params(params)

        assert validated.page == 1  # Should be corrected to 1

    def test_validate_invalid_page_size(self):
        """Test validation with invalid page size."""
        params = PaginationParams(page=1, page_size=5000)
        validated = validate_pagination_params(params)

        assert validated.page_size == 100  # Should be capped at max
