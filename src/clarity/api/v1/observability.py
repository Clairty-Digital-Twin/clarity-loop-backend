"""
Observability API Endpoints

Provides endpoints for the observability dashboard:
- Real-time metrics
- Active traces
- Log streaming
- Alert management
- System health
"""
import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel, Field

from ...observability.metrics import get_metrics
from ...observability.alerting import get_alert_manager, Alert, AlertRule, AlertSeverity
from ...observability.correlation import get_correlation_id

router = APIRouter(prefix="/observability", tags=["observability"])


# Pydantic models
class MetricDataPoint(BaseModel):
    timestamp: float
    value: float
    labels: Dict[str, str] = {}


class MetricSeries(BaseModel):
    name: str
    description: str
    datapoints: List[MetricDataPoint]


class SystemHealthResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, str]
    metrics: Dict[str, float]


class AlertResponse(BaseModel):
    id: str
    rule_name: str
    severity: str
    status: str
    message: str
    labels: Dict[str, str]
    fired_at: datetime
    resolved_at: Optional[datetime] = None


class AlertRuleResponse(BaseModel):
    name: str
    description: str
    severity: str
    condition: str
    threshold: float
    enabled: bool
    runbook_url: Optional[str] = None


class DashboardSummary(BaseModel):
    total_requests: int
    error_rate: float
    avg_response_time: float
    active_alerts: int
    system_health: str
    active_connections: int


# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get comprehensive system health information."""
    metrics = get_metrics()
    alert_manager = get_alert_manager()
    
    # Get current metrics
    try:
        metrics.update_system_metrics()
        
        # Component health checks
        components = {
            "api": "healthy",
            "database": "healthy",  # TODO: Add actual health checks
            "redis": "healthy",
            "ml_models": "healthy",
        }
        
        # Get key metrics
        key_metrics = {
            "cpu_usage": metrics.process_cpu_usage._value._value if hasattr(metrics.process_cpu_usage, '_value') else 0,
            "memory_usage": metrics.process_memory_usage._value._value if hasattr(metrics.process_memory_usage, '_value') else 0,
            "active_connections": metrics.active_connections._value._value if hasattr(metrics.active_connections, '_value') else 0,
            "active_alerts": len(alert_manager.get_active_alerts()),
        }
        
        # Determine overall health
        active_alerts = alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity.value == "critical"]
        
        if critical_alerts:
            overall_health = "unhealthy"
        elif active_alerts:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        return SystemHealthResponse(
            status=overall_health,
            timestamp=time.time(),
            components=components,
            metrics=key_metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format."""
    metrics = get_metrics()
    try:
        prometheus_data = metrics.get_metrics()
        return PlainTextResponse(prometheus_data, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/summary", response_model=DashboardSummary)
async def get_metrics_summary():
    """Get high-level metrics summary for dashboard."""
    metrics = get_metrics()
    alert_manager = get_alert_manager()
    
    try:
        # TODO: Calculate actual metrics from Prometheus data
        # For now, return mock data
        summary = DashboardSummary(
            total_requests=1000,  # TODO: Get from metrics
            error_rate=0.5,       # TODO: Calculate error rate
            avg_response_time=0.15,  # TODO: Calculate average
            active_alerts=len(alert_manager.get_active_alerts()),
            system_health="healthy",  # TODO: Get from health check
            active_connections=10     # TODO: Get from metrics
        )
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return")
):
    """Get alerts with optional filtering."""
    alert_manager = get_alert_manager()
    
    try:
        # Get alerts (both active and from history)
        active_alerts = alert_manager.get_active_alerts()
        historical_alerts = alert_manager.get_alert_history(limit)
        
        # Combine and sort by fired_at (most recent first)
        all_alerts = list(set(active_alerts + historical_alerts))
        all_alerts.sort(key=lambda x: x.fired_at, reverse=True)
        
        # Apply filters
        filtered_alerts = all_alerts
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.status.value == status]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity.value == severity]
        
        # Limit results
        filtered_alerts = filtered_alerts[:limit]
        
        # Convert to response model
        return [
            AlertResponse(
                id=alert.id,
                rule_name=alert.rule_name,
                severity=alert.severity.value,
                status=alert.status.value,
                message=alert.message,
                labels=alert.labels,
                fired_at=alert.fired_at,
                resolved_at=alert.resolved_at
            )
            for alert in filtered_alerts
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/alerts/rules", response_model=List[AlertRuleResponse])
async def get_alert_rules():
    """Get all alert rules."""
    alert_manager = get_alert_manager()
    
    try:
        rules = []
        for rule in alert_manager.rules.values():
            rules.append(AlertRuleResponse(
                name=rule.name,
                description=rule.description,
                severity=rule.severity.value,
                condition=rule.condition,
                threshold=rule.threshold,
                enabled=rule.enabled,
                runbook_url=rule.runbook_url
            ))
        
        return rules
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an active alert."""
    alert_manager = get_alert_manager()
    
    try:
        resolved_alert = alert_manager.resolve_alert(alert_id)
        if not resolved_alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "message": "Alert resolved successfully",
            "alert_id": alert_id,
            "resolved_at": resolved_alert.resolved_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.get("/traces")
async def get_traces(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of traces"),
    min_duration: Optional[float] = Query(None, description="Minimum trace duration in seconds"),
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    operation_name: Optional[str] = Query(None, description="Filter by operation name")
):
    """Get recent traces (mock data for now)."""
    # TODO: Integrate with actual trace backend (Jaeger/Zipkin)
    # For now, return mock trace data
    
    traces = []
    for i in range(min(limit, 20)):
        trace_id = f"trace_{i:04d}"
        duration = 0.1 + (i * 0.05)
        
        if min_duration and duration < min_duration:
            continue
        
        traces.append({
            "trace_id": trace_id,
            "operation_name": f"GET /api/v1/health-data",
            "service_name": "clarity-backend",
            "start_time": time.time() - (i * 60),
            "duration": duration,
            "span_count": 5 + i,
            "error": i % 10 == 0,
            "tags": {
                "http.method": "GET",
                "http.status_code": "500" if i % 10 == 0 else "200",
                "correlation.id": get_correlation_id() or f"corr_{i:04d}"
            }
        })
    
    return {"traces": traces}


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time observability data."""
    await manager.connect(websocket)
    
    try:
        # Send initial data
        await websocket.send_json({
            "type": "connected",
            "timestamp": time.time(),
            "message": "Connected to Clarity Observability Stream"
        })
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
            # Get current data
            metrics = get_metrics()
            alert_manager = get_alert_manager()
            
            # Send real-time update
            update_data = {
                "type": "metrics_update",
                "timestamp": time.time(),
                "data": {
                    "active_alerts": len(alert_manager.get_active_alerts()),
                    "correlation_id": get_correlation_id(),
                    # Add more real-time metrics here
                }
            }
            
            await websocket.send_json(update_data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/logs")
async def get_recent_logs(
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None, description="Filter by log level"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    since: Optional[int] = Query(None, description="Get logs since timestamp")
):
    """Get recent logs (mock data for now)."""
    # TODO: Integrate with actual log aggregation system (ELK, Grafana Loki, etc.)
    # For now, return mock log data
    
    logs = []
    for i in range(min(limit, 50)):
        log_level = ["DEBUG", "INFO", "WARNING", "ERROR"][i % 4]
        
        if level and log_level.lower() != level.lower():
            continue
        
        timestamp = time.time() - (i * 10)
        if since and timestamp < since:
            continue
        
        logs.append({
            "timestamp": timestamp,
            "level": log_level,
            "logger": "clarity.api.v1.health_data",
            "message": f"Processing health data request {i}",
            "correlation_id": correlation_id or f"corr_{i:04d}",
            "extra": {
                "user_id": f"user_{i % 10}",
                "endpoint": "/api/v1/health-data",
                "duration_ms": 150 + (i * 5)
            }
        })
    
    return {"logs": logs[::-1]}  # Most recent first


@router.get("/stats")
async def get_observability_stats():
    """Get observability system statistics."""
    alert_manager = get_alert_manager()
    
    try:
        stats = alert_manager.get_alert_stats()
        
        # Add additional stats
        stats.update({
            "observability_enabled": True,
            "tracing_enabled": True,
            "metrics_collection": True,
            "log_aggregation": True,
            "websocket_connections": len(manager.active_connections),
            "uptime_seconds": time.time() - 1000000,  # TODO: Track actual uptime
        })
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")