"""AWS SQS messaging service for asynchronous processing."""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

import boto3
from botocore.exceptions import ClientError

from clarity.core.exceptions import MessagingError

logger = logging.getLogger(__name__)


class SQSMessagingService:
    """AWS SQS service for async message processing."""
    
    def __init__(
        self,
        queue_url: str,
        region: str = "us-east-1",
        sns_topic_arn: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        self.queue_url = queue_url
        self.sns_topic_arn = sns_topic_arn
        self.region = region
        
        # Create SQS client
        if endpoint_url:  # For local testing with LocalStack
            self.sqs_client = boto3.client(
                'sqs',
                region_name=region,
                endpoint_url=endpoint_url
            )
            if sns_topic_arn:
                self.sns_client = boto3.client(
                    'sns',
                    region_name=region,
                    endpoint_url=endpoint_url
                )
        else:
            self.sqs_client = boto3.client('sqs', region_name=region)
            if sns_topic_arn:
                self.sns_client = boto3.client('sns', region_name=region)
    
    async def publish_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish message to SQS queue."""
        try:
            # Prepare message
            message_id = str(uuid.uuid4())
            message_body = {
                'id': message_id,
                'type': message_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            # Prepare message attributes
            message_attributes = {
                'MessageType': {
                    'DataType': 'String',
                    'StringValue': message_type
                }
            }
            
            if attributes:
                for key, value in attributes.items():
                    message_attributes[key] = {
                        'DataType': 'String',
                        'StringValue': str(value)
                    }
            
            # Send to SQS
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body),
                MessageAttributes=message_attributes
            )
            
            logger.info(f"Published message {message_id} to SQS")
            return response['MessageId']
            
        except ClientError as e:
            logger.error(f"SQS publish error: {e}")
            raise MessagingError(f"Failed to publish message: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error publishing message: {e}")
            raise MessagingError(f"Failed to publish message: {str(e)}")
    
    async def receive_messages(
        self,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Receive messages from SQS queue."""
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=visibility_timeout,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
            
            messages = []
            for msg in response.get('Messages', []):
                try:
                    body = json.loads(msg['Body'])
                    messages.append({
                        'receipt_handle': msg['ReceiptHandle'],
                        'message_id': msg['MessageId'],
                        'body': body,
                        'attributes': msg.get('MessageAttributes', {}),
                        'system_attributes': msg.get('Attributes', {})
                    })
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {msg['MessageId']}")
                    continue
            
            return messages
            
        except ClientError as e:
            logger.error(f"SQS receive error: {e}")
            raise MessagingError(f"Failed to receive messages: {str(e)}")
    
    async def delete_message(self, receipt_handle: str) -> None:
        """Delete message from SQS queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info("Successfully deleted message from SQS")
            
        except ClientError as e:
            logger.error(f"SQS delete error: {e}")
            raise MessagingError(f"Failed to delete message: {str(e)}")
    
    async def batch_delete_messages(
        self,
        receipt_handles: List[str]
    ) -> Dict[str, Any]:
        """Batch delete messages from SQS."""
        try:
            entries = [
                {
                    'Id': str(i),
                    'ReceiptHandle': handle
                }
                for i, handle in enumerate(receipt_handles)
            ]
            
            response = self.sqs_client.delete_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
            
            return {
                'successful': response.get('Successful', []),
                'failed': response.get('Failed', [])
            }
            
        except ClientError as e:
            logger.error(f"SQS batch delete error: {e}")
            raise MessagingError(f"Failed to batch delete messages: {str(e)}")
    
    async def publish_to_sns(
        self,
        subject: str,
        message: Dict[str, Any],
        attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish message to SNS topic for fan-out."""
        if not self.sns_topic_arn:
            raise MessagingError("SNS topic ARN not configured")
        
        try:
            # Prepare message attributes
            message_attributes = {}
            if attributes:
                for key, value in attributes.items():
                    message_attributes[key] = {
                        'DataType': 'String',
                        'StringValue': str(value)
                    }
            
            # Publish to SNS
            response = self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject,
                Message=json.dumps(message),
                MessageAttributes=message_attributes
            )
            
            logger.info(f"Published message to SNS: {response['MessageId']}")
            return response['MessageId']
            
        except ClientError as e:
            logger.error(f"SNS publish error: {e}")
            raise MessagingError(f"Failed to publish to SNS: {str(e)}")
    
    async def get_queue_attributes(self) -> Dict[str, Any]:
        """Get queue attributes and statistics."""
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            return response.get('Attributes', {})
            
        except ClientError as e:
            logger.error(f"SQS get attributes error: {e}")
            raise MessagingError(f"Failed to get queue attributes: {str(e)}")
    
    async def purge_queue(self) -> None:
        """Purge all messages from queue (use with caution)."""
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            logger.warning(f"Purged all messages from queue: {self.queue_url}")
            
        except ClientError as e:
            logger.error(f"SQS purge error: {e}")
            raise MessagingError(f"Failed to purge queue: {str(e)}")


class HealthDataMessageTypes:
    """Message types for health data processing."""
    HEALTH_DATA_UPLOADED = "health_data_uploaded"
    ANALYSIS_REQUESTED = "analysis_requested"
    ANALYSIS_COMPLETED = "analysis_completed"
    INSIGHT_GENERATED = "insight_generated"
    ERROR_OCCURRED = "error_occurred"