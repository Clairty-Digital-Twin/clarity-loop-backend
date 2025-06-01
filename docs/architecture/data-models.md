# Data Models and Database Design

This document defines the comprehensive data models, database schemas, and data flow patterns for the Clarity Loop Backend system.

## Data Architecture Overview

The system uses a hybrid data storage approach optimized for different use cases:

- **Cloud Firestore**: Real-time user data, preferences, and insights
- **Cloud Storage**: Raw health data files and ML model artifacts
- **Cloud SQL (Optional)**: Analytics and reporting data warehouse
- **Redis (Caching)**: High-performance data caching layer

## Core Data Models

### 1. User Profile Model

#### Firestore Document Structure
```json
{
  "users/{userId}": {
    "profile": {
      "email": "user@example.com",
      "displayName": "John Doe",
      "dateOfBirth": "1990-01-15",
      "gender": "male",
      "timezone": "America/New_York",
      "preferredUnits": "metric",
      "privacySettings": {
        "dataSharing": true,
        "researchParticipation": false,
        "anonymousAnalytics": true
      }
    },
    "healthProfile": {
      "height": 175.5,
      "weight": 70.2,
      "activityLevel": "moderate",
      "chronicConditions": [],
      "medications": [],
      "allergies": []
    },
    "preferences": {
      "notificationSettings": {
        "insights": true,
        "achievements": true,
        "reminders": false
      },
      "dataRetention": {
        "rawData": "1_year",
        "insights": "indefinite",
        "analytics": "5_years"
      }
    },
    "metadata": {
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-20T14:45:00Z",
      "lastLoginAt": "2024-01-20T14:45:00Z",
      "dataVersion": "1.2.0"
    }
  }
}
```

#### Pydantic Model Definition
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"

class PrivacySettings(BaseModel):
    data_sharing: bool = True
    research_participation: bool = False
    anonymous_analytics: bool = True

class HealthProfile(BaseModel):
    height: Optional[float] = Field(None, gt=0, le=300, description="Height in cm")
    weight: Optional[float] = Field(None, gt=0, le=500, description="Weight in kg")
    activity_level: Optional[ActivityLevel] = ActivityLevel.MODERATE
    chronic_conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []

class UserProfile(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    display_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: Optional[datetime] = None
    gender: Optional[Gender] = None
    timezone: str = "UTC"
    preferred_units: str = "metric"
    privacy_settings: PrivacySettings = PrivacySettings()
    health_profile: HealthProfile = HealthProfile()
    
    @validator('date_of_birth')
    def validate_age(cls, v):
        if v and v.year < 1900:
            raise ValueError('Birth year must be after 1900')
        return v
```

### 2. Health Data Model

#### Raw Health Data Structure
```json
{
  "healthData/{userId}/sessions/{sessionId}": {
    "sessionInfo": {
      "sessionId": "session_20240120_143000",
      "startTime": "2024-01-20T14:30:00Z",
      "endTime": "2024-01-20T15:30:00Z",
      "dataSource": "apple_watch_series_9",
      "dataTypes": ["heart_rate", "steps", "activity"]
    },
    "heartRate": {
      "samples": [
        {
          "timestamp": "2024-01-20T14:30:00Z",
          "value": 72,
          "confidence": 0.95,
          "context": "resting"
        }
      ],
      "statistics": {
        "min": 65,
        "max": 85,
        "average": 72.5,
        "standardDeviation": 4.2
      }
    },
    "activity": {
      "steps": {
        "total": 1250,
        "cadence": 110,
        "confidence": 0.98
      },
      "distance": {
        "value": 875.5,
        "unit": "meters"
      },
      "activeEnergy": {
        "value": 85.2,
        "unit": "calories"
      },
      "workout": {
        "type": "walking",
        "intensity": "moderate",
        "duration": 3600
      }
    },
    "sleep": {
      "stages": [
        {
          "startTime": "2024-01-20T23:00:00Z",
          "endTime": "2024-01-21T01:30:00Z",
          "stage": "light",
          "confidence": 0.92
        }
      ],
      "quality": {
        "score": 85,
        "efficiency": 0.89,
        "disturbances": 2
      }
    }
  }
}
```

#### Health Data Pydantic Models
```python
from typing import List, Union, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class HeartRateSample(BaseModel):
    timestamp: datetime
    value: int = Field(..., ge=30, le=220, description="Heart rate in BPM")
    confidence: float = Field(..., ge=0.0, le=1.0)
    context: Optional[str] = None

class HeartRateData(BaseModel):
    samples: List[HeartRateSample]
    statistics: Dict[str, float]

class ActivitySample(BaseModel):
    timestamp: datetime
    steps: Optional[int] = Field(None, ge=0)
    distance: Optional[float] = Field(None, ge=0.0)
    active_energy: Optional[float] = Field(None, ge=0.0)
    workout_type: Optional[str] = None

class SleepStage(BaseModel):
    start_time: datetime
    end_time: datetime
    stage: str = Field(..., regex=r'^(awake|light|deep|rem)$')
    confidence: float = Field(..., ge=0.0, le=1.0)

class SleepData(BaseModel):
    stages: List[SleepStage]
    quality_score: Optional[int] = Field(None, ge=0, le=100)
    efficiency: Optional[float] = Field(None, ge=0.0, le=1.0)
    disturbances: Optional[int] = Field(None, ge=0)

class HealthSession(BaseModel):
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    data_source: str
    heart_rate: Optional[HeartRateData] = None
    activity: Optional[ActivitySample] = None
    sleep: Optional[SleepData] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### 3. Insights and Analytics Model

#### AI-Generated Insights Structure
```json
{
  "insights/{userId}/daily/{date}": {
    "insightId": "insight_20240120_daily",
    "userId": "user_12345",
    "date": "2024-01-20",
    "type": "daily_summary",
    "status": "completed",
    "data": {
      "sleepAnalysis": {
        "totalSleep": 7.5,
        "sleepEfficiency": 0.89,
        "sleepStages": {
          "deep": 1.2,
          "light": 5.1,
          "rem": 1.2
        },
        "insights": [
          "Your deep sleep increased by 15% compared to last week",
          "Consider reducing screen time before bed for better sleep quality"
        ]
      },
      "activityAnalysis": {
        "totalSteps": 8750,
        "activeMinutes": 45,
        "caloriesBurned": 2150,
        "insights": [
          "You exceeded your daily step goal by 750 steps",
          "Your activity pattern shows consistent movement throughout the day"
        ]
      },
      "heartRateAnalysis": {
        "restingHeartRate": 65,
        "averageHeartRate": 78,
        "heartRateVariability": 42,
        "insights": [
          "Your resting heart rate has improved by 3 BPM this month",
          "HRV indicates good recovery from yesterday's workout"
        ]
      }
    },
    "narrative": {
      "summary": "You had a great day with quality sleep and consistent activity...",
      "recommendations": [
        "Continue your current sleep schedule",
        "Consider adding 10 minutes of morning stretching"
      ],
      "achievements": [
        "Step goal exceeded",
        "Improved sleep quality"
      ]
    },
    "metadata": {
      "generatedAt": "2024-01-21T06:00:00Z",
      "modelVersion": "actigraphy_v2.1",
      "processingTime": 2.3,
      "confidence": 0.94
    }
  }
}
```

#### Insights Pydantic Models
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date

class SleepInsight(BaseModel):
    total_sleep: float = Field(..., ge=0.0, le=24.0)
    sleep_efficiency: float = Field(..., ge=0.0, le=1.0)
    sleep_stages: Dict[str, float]
    insights: List[str]

class ActivityInsight(BaseModel):
    total_steps: int = Field(..., ge=0)
    active_minutes: int = Field(..., ge=0)
    calories_burned: int = Field(..., ge=0)
    insights: List[str]

class HeartRateInsight(BaseModel):
    resting_heart_rate: int = Field(..., ge=30, le=100)
    average_heart_rate: int = Field(..., ge=30, le=220)
    heart_rate_variability: Optional[int] = Field(None, ge=0, le=200)
    insights: List[str]

class NarrativeInsight(BaseModel):
    summary: str = Field(..., min_length=10, max_length=1000)
    recommendations: List[str]
    achievements: List[str]

class DailyInsight(BaseModel):
    insight_id: str
    user_id: str
    date: date
    type: str = "daily_summary"
    status: str = Field(..., regex=r'^(pending|processing|completed|failed)$')
    sleep_analysis: Optional[SleepInsight] = None
    activity_analysis: Optional[ActivityInsight] = None
    heart_rate_analysis: Optional[HeartRateInsight] = None
    narrative: Optional[NarrativeInsight] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    generated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
```

### 4. ML Model Metadata

#### Model Tracking Structure
```json
{
  "models/actigraphy_transformer": {
    "modelInfo": {
      "name": "actigraphy_transformer",
      "version": "2.1.0",
      "type": "transformer",
      "framework": "pytorch",
      "description": "Advanced sleep and activity pattern analysis"
    },
    "performance": {
      "accuracy": 0.945,
      "precision": 0.938,
      "recall": 0.952,
      "f1Score": 0.945,
      "lastEvaluated": "2024-01-15T10:00:00Z"
    },
    "deployment": {
      "status": "active",
      "endpoint": "https://actigraphy-transformer-service.run.app",
      "resourceRequirements": {
        "cpu": "2 vCPU",
        "memory": "4 GB",
        "gpu": "optional"
      },
      "scalingConfig": {
        "minInstances": 1,
        "maxInstances": 10,
        "targetConcurrency": 100
      }
    },
    "dataRequirements": {
      "inputFormat": "time_series_json",
      "minSamples": 100,
      "maxSamples": 10000,
      "requiredFields": ["timestamp", "heart_rate", "activity"],
      "optionalFields": ["sleep_stage", "workout_type"]
    }
  }
}
```

### 5. Processing Jobs and Tasks

#### Job Queue Structure
```json
{
  "jobs/{jobId}": {
    "jobId": "job_20240120_143000_001",
    "userId": "user_12345",
    "type": "health_analysis",
    "status": "processing",
    "priority": "high",
    "createdAt": "2024-01-20T14:30:00Z",
    "startedAt": "2024-01-20T14:30:05Z",
    "estimatedCompletion": "2024-01-20T14:32:00Z",
    "input": {
      "dataSessionIds": ["session_001", "session_002"],
      "analysisType": "comprehensive",
      "modelVersion": "2.1.0"
    },
    "progress": {
      "stage": "ml_processing",
      "percentage": 65,
      "currentStep": "feature_extraction",
      "stepsCompleted": 3,
      "totalSteps": 5
    },
    "output": {
      "insightId": "insight_20240120_daily",
      "artifacts": [
        "gs://bucket/processed_data/user_12345/20240120.json"
      ]
    },
    "metrics": {
      "processingTime": 95.2,
      "dataProcessed": 1024,
      "resourceUsage": {
        "cpu": 1.2,
        "memory": 512,
        "networkIO": 15.3
      }
    },
    "errors": []
  }
}
```

## Database Relationships and Indexing

### Firestore Collection Structure
```
/users/{userId}
  /healthSessions/{sessionId}
  /insights/{insightType}/{date}
  /preferences
  /permissions

/jobs/{jobId}
  /progress
  /logs

/models/{modelName}
  /versions/{version}
  /performance
  /deployments

/system
  /configuration
  /monitoring
  /audit_logs
```

### Firestore Indexing Strategy
```javascript
// Composite indexes for efficient queries
const indexes = [
  {
    collection: 'users',
    fields: [
      { field: 'email', order: 'ASCENDING' },
      { field: 'metadata.createdAt', order: 'DESCENDING' }
    ]
  },
  {
    collection: 'healthSessions',
    fields: [
      { field: 'userId', order: 'ASCENDING' },
      { field: 'startTime', order: 'DESCENDING' }
    ]
  },
  {
    collection: 'insights',
    fields: [
      { field: 'userId', order: 'ASCENDING' },
      { field: 'date', order: 'DESCENDING' },
      { field: 'type', order: 'ASCENDING' }
    ]
  },
  {
    collection: 'jobs',
    fields: [
      { field: 'status', order: 'ASCENDING' },
      { field: 'priority', order: 'DESCENDING' },
      { field: 'createdAt', order: 'ASCENDING' }
    ]
  }
];
```

### Cloud SQL Schema (Analytics)

#### User Analytics Table
```sql
CREATE TABLE user_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    
    -- Sleep metrics
    total_sleep_hours DECIMAL(4,2),
    sleep_efficiency DECIMAL(3,2),
    deep_sleep_hours DECIMAL(4,2),
    light_sleep_hours DECIMAL(4,2),
    rem_sleep_hours DECIMAL(4,2),
    
    -- Activity metrics
    total_steps INTEGER,
    active_minutes INTEGER,
    calories_burned INTEGER,
    distance_meters DECIMAL(8,2),
    
    -- Heart rate metrics
    resting_heart_rate INTEGER,
    average_heart_rate INTEGER,
    max_heart_rate INTEGER,
    heart_rate_variability INTEGER,
    
    -- Metadata
    data_quality_score DECIMAL(3,2),
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, date)
);

-- Indexes for efficient querying
CREATE INDEX idx_user_analytics_user_date ON user_analytics(user_id, date DESC);
CREATE INDEX idx_user_analytics_date ON user_analytics(date);
CREATE INDEX idx_user_analytics_quality ON user_analytics(data_quality_score);
```

#### Aggregated Trends Table
```sql
CREATE TABLE user_trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    trend_type VARCHAR(50) NOT NULL, -- 'weekly', 'monthly', 'quarterly'
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Trend metrics (JSON for flexibility)
    trend_data JSONB NOT NULL,
    
    -- Trend indicators
    improvement_score DECIMAL(3,2),
    trend_direction VARCHAR(20), -- 'improving', 'stable', 'declining'
    confidence_level DECIMAL(3,2),
    
    -- Metadata
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, trend_type, period_start)
);

-- GIN index for JSONB queries
CREATE INDEX idx_user_trends_data ON user_trends USING gin(trend_data);
CREATE INDEX idx_user_trends_user_period ON user_trends(user_id, trend_type, period_start DESC);
```

## Data Validation and Constraints

### Input Validation Rules
```python
from pydantic import BaseModel, validator, root_validator
from typing import Optional, List
from datetime import datetime, timedelta

class HealthDataValidation(BaseModel):
    """Comprehensive validation for health data inputs"""
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if v is not None:
            if v < 30 or v > 220:
                raise ValueError('Heart rate must be between 30-220 BPM')
            if isinstance(v, list):
                # Validate heart rate variability
                if len(v) > 1:
                    max_change = max(abs(v[i] - v[i-1]) for i in range(1, len(v)))
                    if max_change > 50:
                        raise ValueError('Heart rate changes too rapid, possible invalid data')
        return v
    
    @validator('sleep_stages')
    def validate_sleep_stages(cls, v):
        if v:
            valid_stages = {'awake', 'light', 'deep', 'rem'}
            for stage in v:
                if stage.get('stage') not in valid_stages:
                    raise ValueError(f'Invalid sleep stage: {stage.get("stage")}')
            
            # Validate stage transitions
            transitions = [(v[i]['stage'], v[i+1]['stage']) for i in range(len(v)-1)]
            invalid_transitions = [
                ('deep', 'awake'),  # Unlikely direct transition
                ('rem', 'deep')     # REM typically follows light sleep
            ]
            
            for transition in transitions:
                if transition in invalid_transitions:
                    raise ValueError(f'Unlikely sleep stage transition: {transition[0]} -> {transition[1]}')
        
        return v
    
    @root_validator
    def validate_temporal_consistency(cls, values):
        """Ensure timestamps are consistent and logical"""
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        
        if start_time and end_time:
            if end_time <= start_time:
                raise ValueError('End time must be after start time')
            
            duration = end_time - start_time
            if duration > timedelta(hours=24):
                raise ValueError('Session duration cannot exceed 24 hours')
            
            # Validate against future timestamps
            if start_time > datetime.utcnow():
                raise ValueError('Start time cannot be in the future')
        
        return values
```

## Data Retention and Archival

### Retention Policies
```yaml
data_retention_policies:
  user_profiles:
    retention_period: indefinite
    deletion_trigger: user_account_deletion
    backup_frequency: daily
    
  raw_health_data:
    retention_period: 2_years
    archival_period: 5_years
    deletion_trigger: retention_expiry
    backup_frequency: daily
    
  processed_insights:
    retention_period: 5_years
    archival_period: 10_years
    deletion_trigger: user_request_or_expiry
    backup_frequency: weekly
    
  system_logs:
    retention_period: 1_year
    archival_period: 3_years
    deletion_trigger: automatic
    backup_frequency: daily
    
  audit_logs:
    retention_period: 7_years
    archival_period: indefinite
    deletion_trigger: legal_requirement_only
    backup_frequency: real_time
```

### Automated Data Lifecycle
```python
from google.cloud import firestore
from datetime import datetime, timedelta
import asyncio

class DataLifecycleManager:
    """Manage automated data retention and archival"""
    
    def __init__(self):
        self.db = firestore.AsyncClient()
    
    async def archive_old_data(self):
        """Archive data older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=365*2)  # 2 years
        
        # Query old health sessions
        old_sessions = self.db.collection_group('healthSessions').where(
            'startTime', '<', cutoff_date
        ).limit(1000)
        
        async for session in old_sessions.stream():
            await self._archive_session(session)
    
    async def _archive_session(self, session):
        """Archive individual session to cold storage"""
        # Move to Cloud Storage for long-term archival
        # Update Firestore with archival metadata
        # Remove from active database
        pass
    
    async def cleanup_expired_data(self):
        """Remove data that has exceeded all retention periods"""
        # Implement secure data deletion
        pass
```

This comprehensive data model documentation provides the foundation for implementing a robust, scalable, and compliant health data backend system.
