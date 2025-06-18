"""
Alerting System

Comprehensive alerting with:
- Alert rule management
- Multi-channel notifications (Slack, PagerDuty, Email)
- Alert deduplication and throttling
- Runbook integration
- SLO/SLI tracking
"""
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from urllib.parse import urljoin

import httpx
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved" 
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # PromQL or custom condition
    threshold: float
    duration: int  # seconds
    enabled: bool = True
    runbook_url: Optional[str] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}


@dataclass  
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    fingerprint: str = ""
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for alert deduplication."""
        data = f"{self.rule_name}:{json.dumps(self.labels, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    async def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, name: str, webhook_url: str, channel: str = None, enabled: bool = True):
        super().__init__(name, enabled)
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.enabled:
            return False
            
        try:
            # Format message
            color = self._get_color(alert.severity)
            message = self._format_slack_message(alert)
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🚨 {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Status", "value": alert.status.value.upper(), "short": True},
                        {"title": "Fired At", "value": alert.fired_at.isoformat(), "short": True},
                    ],
                    "footer": "Clarity Observability",
                    "ts": int(alert.fired_at.timestamp())
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            # Add runbook link if available
            if alert.annotations.get("runbook_url"):
                payload["attachments"][0]["actions"] = [{
                    "type": "button",
                    "text": "📖 Runbook",
                    "url": alert.annotations["runbook_url"]
                }]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error("Failed to send Slack notification", error=str(e), alert_id=alert.id)
            return False
    
    def _get_color(self, severity: AlertSeverity) -> str:
        """Get color for alert severity."""
        color_map = {
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.HIGH: "#ff8000", 
            AlertSeverity.MEDIUM: "#ffff00",
            AlertSeverity.LOW: "#0080ff",
            AlertSeverity.INFO: "#00ff00"
        }
        return color_map.get(severity, "#808080")
    
    def _format_slack_message(self, alert: Alert) -> str:
        """Format Slack message."""
        return f"{alert.message}\n\nLabels: {json.dumps(alert.labels)}"


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel."""
    
    def __init__(self, name: str, integration_key: str, enabled: bool = True):
        super().__init__(name, enabled)
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        if not self.enabled:
            return False
            
        try:
            event_action = "trigger" if alert.status == AlertStatus.FIRING else "resolve"
            
            payload = {
                "routing_key": self.integration_key,
                "event_action": event_action,
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"{alert.rule_name}: {alert.message}",
                    "source": "Clarity Backend",
                    "severity": self._map_severity(alert.severity),
                    "timestamp": alert.fired_at.isoformat(),
                    "component": "clarity-backend",
                    "group": "observability",
                    "class": "alert",
                    "custom_details": {
                        "labels": alert.labels,
                        "annotations": alert.annotations,
                        "alert_id": alert.id
                    }
                }
            }
            
            # Add links
            if alert.annotations.get("runbook_url"):
                payload["links"] = [{
                    "href": alert.annotations["runbook_url"],
                    "text": "Runbook"
                }]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error("Failed to send PagerDuty notification", error=str(e), alert_id=alert.id)
            return False
    
    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map alert severity to PagerDuty severity."""
        severity_map = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.LOW: "info",
            AlertSeverity.INFO: "info"
        }
        return severity_map.get(severity, "info")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, name: str, smtp_config: Dict[str, Any], 
                 to_emails: List[str], enabled: bool = True):
        super().__init__(name, enabled)
        self.smtp_config = smtp_config
        self.to_emails = to_emails
    
    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.enabled:
            return False
            
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # Format body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                if self.smtp_config.get('use_tls'):
                    server.starttls()
                if self.smtp_config.get('username'):
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error("Failed to send email notification", error=str(e), alert_id=alert.id)
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body."""
        return f"""
        <html>
        <body>
            <h2>🚨 Alert: {alert.rule_name}</h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Status:</strong> {alert.status.value.upper()}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            <p><strong>Fired At:</strong> {alert.fired_at.isoformat()}</p>
            
            <h3>Labels</h3>
            <ul>
                {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in alert.labels.items())}
            </ul>
            
            <h3>Annotations</h3>
            <ul>
                {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in alert.annotations.items())}
            </ul>
            
            {f'<p><a href="{alert.annotations["runbook_url"]}">📖 View Runbook</a></p>' if alert.annotations.get("runbook_url") else ""}
            
            <hr>
            <p><em>Generated by Clarity Observability System</em></p>
        </body>
        </html>
        """


class AlertManager:
    """Central alert management system."""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_history: List[Alert] = []
        self.deduplication_window = 300  # 5 minutes
        self.throttle_window = 3600  # 1 hour
        self.max_notifications_per_hour = 5
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules[rule.name] = rule
        logger.info("Added alert rule", rule_name=rule.name, severity=rule.severity.value)
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info("Removed alert rule", rule_name=rule_name)
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self.notification_channels[channel.name] = channel
        logger.info("Added notification channel", channel_name=channel.name, enabled=channel.enabled)
    
    def fire_alert(self, rule_name: str, message: str, labels: Dict[str, str] = None, 
                   annotations: Dict[str, str] = None) -> Alert:
        """Fire an alert."""
        rule = self.rules.get(rule_name)
        if not rule or not rule.enabled:
            raise ValueError(f"Alert rule '{rule_name}' not found or disabled")
        
        # Create alert
        alert = Alert(
            id=f"{rule_name}_{int(time.time())}",
            rule_name=rule_name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=message,
            labels=labels or {},
            annotations={**(annotations or {}), **(rule.annotations or {})},
            fired_at=datetime.utcnow()
        )
        
        # Add runbook URL if available
        if rule.runbook_url:
            alert.annotations["runbook_url"] = rule.runbook_url
        
        # Check for deduplication
        existing_alert = self._find_duplicate(alert)
        if existing_alert:
            logger.debug("Alert deduplicated", alert_id=alert.id, existing_id=existing_alert.id)
            return existing_alert
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        logger.warning(
            "Alert fired",
            alert_id=alert.id,
            rule_name=rule_name,
            severity=rule.severity.value,
            message=message
        )
        
        return alert
    
    def resolve_alert(self, alert_id_or_fingerprint: str) -> Optional[Alert]:
        """Resolve an alert."""
        alert = None
        
        # Find by ID or fingerprint
        if alert_id_or_fingerprint in self.active_alerts:
            alert = self.active_alerts[alert_id_or_fingerprint]
        else:
            for a in self.active_alerts.values():
                if a.fingerprint == alert_id_or_fingerprint:
                    alert = a
                    break
        
        if not alert:
            return None
        
        # Update alert
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        # Remove from active alerts
        del self.active_alerts[alert.id]
        
        # Send resolution notification
        asyncio.create_task(self._send_notifications(alert))
        
        logger.info(
            "Alert resolved",
            alert_id=alert.id,
            rule_name=alert.rule_name,
            duration=(alert.resolved_at - alert.fired_at).total_seconds()
        )
        
        return alert
    
    def _find_duplicate(self, new_alert: Alert) -> Optional[Alert]:
        """Find duplicate alert within deduplication window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.deduplication_window)
        
        for alert in self.active_alerts.values():
            if (alert.fingerprint == new_alert.fingerprint and 
                alert.fired_at > cutoff_time):
                return alert
        
        return None
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for alert."""
        # Check throttling
        if not self._should_notify(alert):
            logger.debug("Alert notification throttled", alert_id=alert.id)
            return
        
        # Send to all enabled channels
        tasks = []
        for channel in self.notification_channels.values():
            if channel.enabled:
                tasks.append(channel.send(alert))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            
            # Update notification tracking
            alert.last_notification = datetime.utcnow()
            alert.notification_count += 1
            
            logger.info(
                "Alert notifications sent",
                alert_id=alert.id,
                channels_sent=success_count,
                total_channels=len(tasks)
            )
    
    def _should_notify(self, alert: Alert) -> bool:
        """Check if alert should trigger notification based on throttling rules."""
        if not alert.last_notification:
            return True
        
        # Check throttle window
        throttle_cutoff = datetime.utcnow() - timedelta(seconds=self.throttle_window)
        if alert.last_notification < throttle_cutoff:
            return True
        
        # Check max notifications per hour
        return alert.notification_count < self.max_notifications_per_hour
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                severity=AlertSeverity.HIGH,
                condition="rate(clarity_http_requests_total{status_code=~'5..'}[5m]) > 0.01",
                threshold=0.01,
                duration=300,
                runbook_url="https://docs.clarity.com/runbooks/high-error-rate",
                annotations={
                    "summary": "API error rate is above 1%",
                    "description": "The API is returning errors at a rate higher than acceptable threshold"
                }
            ),
            AlertRule(
                name="slow_response_time",
                description="Slow response time detected",
                severity=AlertSeverity.MEDIUM,
                condition="histogram_quantile(0.99, clarity_http_request_duration_seconds) > 1.0",
                threshold=1.0,
                duration=300,
                runbook_url="https://docs.clarity.com/runbooks/slow-response-time",
                annotations={
                    "summary": "99th percentile response time is above 1 second",
                    "description": "API response times are slower than expected"
                }
            ),
            AlertRule(
                name="high_memory_usage",
                description="High memory usage detected",
                severity=AlertSeverity.HIGH,
                condition="clarity_process_memory_usage_bytes > 1073741824",  # 1GB
                threshold=1073741824,
                duration=600,
                runbook_url="https://docs.clarity.com/runbooks/high-memory-usage",
                annotations={
                    "summary": "Process memory usage is above 1GB",
                    "description": "The application is using more memory than expected"
                }
            ),
            AlertRule(
                name="ml_model_error_rate",
                description="ML model error rate too high",
                severity=AlertSeverity.HIGH,
                condition="rate(clarity_ml_predictions_total{status='error'}[10m]) > 0.05",
                threshold=0.05,
                duration=600,
                runbook_url="https://docs.clarity.com/runbooks/ml-model-errors",
                annotations={
                    "summary": "ML model error rate is above 5%",
                    "description": "Machine learning models are failing at a high rate"
                }
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Average resolution time
        resolved_alerts = [a for a in self.alert_history if a.resolved_at]
        if resolved_alerts:
            avg_resolution_time = sum(
                (a.resolved_at - a.fired_at).total_seconds() 
                for a in resolved_alerts
            ) / len(resolved_alerts)
        else:
            avg_resolution_time = 0
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "severity_distribution": severity_counts,
            "average_resolution_time_seconds": avg_resolution_time,
            "notification_channels": len(self.notification_channels),
            "alert_rules": len(self.rules)
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager