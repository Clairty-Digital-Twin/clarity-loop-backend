"""Tests for analysis subscriber service."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, Request
import pytest

from clarity.services.messaging.analysis_subscriber import AnalysisSubscriber


@pytest.fixture
def subscriber() -> AnalysisSubscriber:
    with (
        patch("google.cloud.storage.Client"),
        patch(
            "clarity.services.messaging.publisher.get_publisher",
            return_value=MagicMock(),
        ),
    ):
        return AnalysisSubscriber()


def test_extract_message_data_success(subscriber: AnalysisSubscriber):
    data = {"user_id": "123", "upload_id": "456", "gcs_path": "gs://bucket/path"}
    encoded_data = base64.b64encode(json.dumps(data).encode("utf-8")).decode("utf-8")
    pubsub_body = {"message": {"data": encoded_data}}

    extracted_data = subscriber._extract_message_data(pubsub_body)
    assert extracted_data == data


def test_extract_message_data_missing_field(subscriber: AnalysisSubscriber):
    data = {"user_id": "123", "upload_id": "456"}  # Missing gcs_path
    encoded_data = base64.b64encode(json.dumps(data).encode("utf-8")).decode("utf-8")
    pubsub_body = {"message": {"data": encoded_data}}

    with pytest.raises(HTTPException) as excinfo:
        subscriber._extract_message_data(pubsub_body)
    assert excinfo.value.status_code == 400
    assert "Missing required field: gcs_path" in excinfo.value.detail


@pytest.mark.asyncio
async def test_download_health_data_success(subscriber: AnalysisSubscriber):
    gcs_path = "gs://bucket/path"
    expected_data: dict[str, list[Any]] = {"metrics": []}
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = json.dumps(expected_data)

    with patch.object(subscriber.storage_client, "bucket") as mock_bucket:
        mock_bucket.return_value.blob.return_value = mock_blob
        data = await subscriber._download_health_data(gcs_path)
        assert data == expected_data


@pytest.mark.asyncio
async def test_download_health_data_not_found(subscriber: AnalysisSubscriber):
    gcs_path = "gs://bucket/path"
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False

    with patch.object(subscriber.storage_client, "bucket") as mock_bucket:
        mock_bucket.return_value.blob.return_value = mock_blob
        with pytest.raises(FileNotFoundError):
            await subscriber._download_health_data(gcs_path)


@pytest.mark.asyncio
async def test_process_health_data_message(subscriber: AnalysisSubscriber):
    user_id = "test_user"
    upload_id = "test_upload"
    gcs_path = "gs://bucket/path"
    health_data = {"metrics": [{"value": 80}]}
    analysis_results = {"hrv": 50}

    mock_request = AsyncMock(spec=Request)
    mock_request.json.return_value = {
        "message": {
            "data": base64.b64encode(
                json.dumps(
                    {
                        "user_id": user_id,
                        "upload_id": upload_id,
                        "gcs_path": gcs_path,
                    }
                ).encode("utf-8")
            ).decode("utf-8")
        }
    }
    subscriber.publisher = AsyncMock()
    subscriber.environment = "production"

    with (
        patch.object(subscriber, "_download_health_data", return_value=health_data),
        patch(
            "clarity.services.messaging.analysis_subscriber.run_analysis_pipeline",
            return_value=analysis_results,
        ) as mock_run_pipeline,
        patch.object(subscriber, "_verify_pubsub_token") as mock_verify_token,
    ):
        result = await subscriber.process_health_data_message(mock_request)

    assert result["status"] == "success"
    assert result["user_id"] == user_id
    mock_run_pipeline.assert_called_once_with(user_id=user_id, health_data=health_data)
    subscriber.publisher.publish_insight_request.assert_called_once_with(
        user_id=user_id,
        upload_id=upload_id,
        analysis_results=analysis_results,
    )
    mock_verify_token.assert_called_once()
