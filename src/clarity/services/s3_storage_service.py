"""AWS S3 storage service for file uploads."""

from datetime import datetime, timedelta
import io
import json
import logging
import mimetypes
from typing import Any, BinaryIO, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from clarity.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class S3StorageService:
    """AWS S3 storage service for handling file uploads."""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ):
        self.bucket_name = bucket_name
        self.region = region

        # Create S3 client
        if endpoint_url:  # For local testing with LocalStack
            self.s3_client = boto3.client(
                "s3", region_name=region, endpoint_url=endpoint_url
            )
        else:
            self.s3_client = boto3.client("s3", region_name=region)

    async def upload_file(
        self,
        file_data: BinaryIO,
        file_name: str,
        user_id: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload file to S3 and return the S3 key."""
        try:
            # Generate S3 key with user namespace
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"users/{user_id}/uploads/{timestamp}_{file_name}"

            # Detect content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_name)
                content_type = content_type or "application/octet-stream"

            # Prepare metadata
            s3_metadata = {
                "user_id": user_id,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "original_filename": file_name,
            }
            if metadata:
                s3_metadata.update(metadata)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type,
                Metadata=s3_metadata,
                ServerSideEncryption="AES256",  # Enable encryption at rest
            )

            logger.info(f"Successfully uploaded file to S3: {s3_key}")
            return s3_key

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise StorageError("Storage service credentials not configured")
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise StorageError(f"Failed to upload file: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            raise StorageError(f"Failed to upload file: {e!s}")

    async def download_file(self, s3_key: str) -> bytes:
        """Download file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            return response["Body"].read()

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise StorageError(f"File not found: {s3_key}")
            logger.error(f"S3 download error: {e}")
            raise StorageError(f"Failed to download file: {e!s}")

    async def get_download_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for downloading a file."""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )

            return url

        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise StorageError(f"Failed to generate download URL: {e!s}")

    async def get_upload_url(
        self, file_name: str, user_id: str, content_type: str, expiration: int = 3600
    ) -> dict[str, Any]:
        """Generate a presigned URL for direct upload to S3."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"users/{user_id}/uploads/{timestamp}_{file_name}"

            # Generate presigned POST URL
            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    "Content-Type": content_type,
                    "x-amz-server-side-encryption": "AES256",
                    "x-amz-meta-user_id": user_id,
                    "x-amz-meta-upload_timestamp": datetime.utcnow().isoformat(),
                },
                Conditions=[
                    {"Content-Type": content_type},
                    ["content-length-range", 0, 100 * 1024 * 1024],  # Max 100MB
                ],
                ExpiresIn=expiration,
            )

            return {
                "url": response["url"],
                "fields": response["fields"],
                "s3_key": s3_key,
                "expires_at": (
                    datetime.utcnow() + timedelta(seconds=expiration)
                ).isoformat(),
            }

        except ClientError as e:
            logger.error(f"Error generating upload URL: {e}")
            raise StorageError(f"Failed to generate upload URL: {e!s}")

    async def delete_file(self, s3_key: str) -> None:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

            logger.info(f"Successfully deleted file from S3: {s3_key}")

        except ClientError as e:
            logger.error(f"S3 delete error: {e}")
            raise StorageError(f"Failed to delete file: {e!s}")

    async def list_user_files(
        self, user_id: str, prefix: str | None = None, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """List files for a user."""
        try:
            # Build prefix
            base_prefix = f"users/{user_id}/uploads/"
            if prefix:
                base_prefix += prefix

            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=base_prefix, MaxKeys=max_results
            )

            files = []
            for obj in response.get("Contents", []):
                # Get object metadata
                head_response = self.s3_client.head_object(
                    Bucket=self.bucket_name, Key=obj["Key"]
                )

                files.append(
                    {
                        "s3_key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "content_type": head_response.get("ContentType", "unknown"),
                        "metadata": head_response.get("Metadata", {}),
                    }
                )

            return files

        except ClientError as e:
            logger.error(f"S3 list error: {e}")
            raise StorageError(f"Failed to list files: {e!s}")

    async def get_file_metadata(self, s3_key: str) -> dict[str, Any]:
        """Get file metadata from S3."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            return {
                "s3_key": s3_key,
                "size": response["ContentLength"],
                "content_type": response.get("ContentType", "unknown"),
                "last_modified": response["LastModified"].isoformat(),
                "metadata": response.get("Metadata", {}),
                "etag": response.get("ETag", "").strip('"'),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise StorageError(f"File not found: {s3_key}")
            logger.error(f"S3 metadata error: {e}")
            raise StorageError(f"Failed to get file metadata: {e!s}")
