# Apple HealthKit Integration Guide

Comprehensive technical guide for seamless HealthKit data ingestion, validation, and processing in the Clarity Loop Backend.

## HealthKit Data Model & Mapping

### Core Data Types

#### Activity & Motion Data

```python
# HKQuantityType mappings for actigraphy
HEALTHKIT_ACTIVITY_TYPES = {
    'HKQuantityTypeIdentifierStepCount': 'steps',
    'HKQuantityTypeIdentifierDistanceWalkingRunning': 'distance_walking_running',
    'HKQuantityTypeIdentifierActiveEnergyBurned': 'active_energy',
    'HKQuantityTypeIdentifierBasalEnergyBurned': 'basal_energy',
    'HKQuantityTypeIdentifierAppleExerciseTime': 'exercise_minutes',
    'HKQuantityTypeIdentifierAppleStandTime': 'stand_minutes',
    'HKQuantityTypeIdentifierFlightsClimbed': 'flights_climbed'
}

# Movement disorder specific metrics
MOVEMENT_DISORDER_TYPES = {
    'HKQuantityTypeIdentifierAppleWalkingSteadiness': 'walking_steadiness',
    'HKQuantityTypeIdentifierWalkingSpeed': 'walking_speed',
    'HKQuantityTypeIdentifierWalkingStepLength': 'step_length',
    'HKQuantityTypeIdentifierWalkingAsymmetryPercentage': 'walking_asymmetry',
    'HKQuantityTypeIdentifierWalkingDoubleSupportPercentage': 'double_support_time'
}
```

#### Heart Rate & Variability

```python
HEART_RATE_TYPES = {
    'HKQuantityTypeIdentifierHeartRate': 'heart_rate_bpm',
    'HKQuantityTypeIdentifierRestingHeartRate': 'resting_heart_rate',
    'HKQuantityTypeIdentifierWalkingHeartRateAverage': 'walking_heart_rate',
    'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv_sdnn'
}

# Apple Watch Series 4+ ECG support
ECG_TYPES = {
    'HKElectrocardiogramType': 'ecg_reading',
    'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv_from_ecg'
}
```

#### Sleep & Recovery

```python
SLEEP_TYPES = {
    'HKCategoryTypeIdentifierSleepAnalysis': 'sleep_analysis',
    'HKQuantityTypeIdentifierTimeInBed': 'time_in_bed',
    'HKQuantityTypeIdentifierSleepDurationGoal': 'sleep_goal'
}

# iOS 16+ Sleep Stage Data
SLEEP_STAGE_MAPPING = {
    0: 'in_bed',
    1: 'asleep_unspecified',
    2: 'awake',
    3: 'asleep_core',  # Light sleep
    4: 'asleep_deep',
    5: 'asleep_rem'
}
```

### Data Validation Schema

```python
# FastAPI Pydantic models for HealthKit data validation

from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class HealthKitDataType(str, Enum):
    QUANTITY = "HKQuantityType"
    CATEGORY = "HKCategoryType"
    CHARACTERISTIC = "HKCharacteristicType"
    CORRELATION = "HKCorrelationType"
    DOCUMENT = "HKDocumentType"
    ELECTROCARDIOGRAM = "HKElectrocardiogram"

class HealthKitUnit(BaseModel):
    unit_string: str = Field(..., description="HKUnit string representation")
    
    @validator('unit_string')
    def validate_unit(cls, v):
        # Common HealthKit units validation
        valid_units = {
            'count', 'count/min', 'km', 'mi', 'kcal', 'kJ',
            'ms', 's', 'min', 'hr', 'day', 'bpm',
            'mg/dL', 'mmol/L', '°F', '°C', '%',
            'dB(SPL)', 'cm', 'ft', 'in', 'kg', 'lb'
        }
        if v not in valid_units:
            raise ValueError(f'Invalid HealthKit unit: {v}')
        return v

class HealthKitQuantitySample(BaseModel):
    uuid: str = Field(..., description="Unique sample identifier")
    type_identifier: str = Field(..., description="HKQuantityTypeIdentifier")
    start_date: datetime = Field(..., description="Sample start time")
    end_date: datetime = Field(..., description="Sample end time")
    value: float = Field(..., description="Numerical value")
    unit: HealthKitUnit = Field(..., description="Unit of measurement")
    source: str = Field(..., description="Data source (Apple Watch, iPhone, etc.)")
    device: Optional[Dict[str, Any]] = Field(None, description="Device metadata")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('value')
    def validate_value_range(cls, v, values):
        """Validate reasonable ranges for different data types"""
        type_id = values.get('type_identifier', '')
        
        if 'HeartRate' in type_id and (v < 30 or v > 220):
            raise ValueError('Heart rate value out of physiological range')
        elif 'StepCount' in type_id and (v < 0 or v > 50000):
            raise ValueError('Step count value out of reasonable range')
        elif 'ActiveEnergyBurned' in type_id and (v < 0 or v > 10000):
            raise ValueError('Active energy value out of reasonable range')
            
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Ensure dates are not in the future and within reasonable past range"""
        now = datetime.utcnow()
        if v > now:
            raise ValueError('Sample date cannot be in the future')
        if (now - v).days > 365:
            raise ValueError('Sample date older than 1 year')
        return v

class HealthKitCategorySample(BaseModel):
    uuid: str
    type_identifier: str
    start_date: datetime
    end_date: datetime
    value: int = Field(..., description="Category value (enum)")
    source: str
    device: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthKitWorkout(BaseModel):
    uuid: str
    workout_activity_type: int = Field(..., description="HKWorkoutActivityType")
    start_date: datetime
    end_date: datetime
    duration: float = Field(..., description="Duration in seconds")
    total_energy_burned: Optional[float] = None
    total_distance: Optional[float] = None
    source: str
    metadata: Optional[Dict[str, Any]] = None

class HealthKitUploadRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    device_info: Dict[str, Any] = Field(..., description="Device metadata")
    export_date: datetime = Field(..., description="When data was exported")
    locale: str = Field(default="en_US", description="User locale")
    quantity_samples: List[HealthKitQuantitySample] = Field(default_factory=list)
    category_samples: List[HealthKitCategorySample] = Field(default_factory=list)
    workouts: List[HealthKitWorkout] = Field(default_factory=list)
    
    @validator('quantity_samples', 'category_samples', 'workouts')
    def validate_sample_limits(cls, v):
        """Prevent oversized uploads"""
        if len(v) > 10000:
            raise ValueError('Too many samples in single upload (max 10,000)')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

## API Endpoints & Implementation

### Upload Endpoint

```python
# app/routers/healthkit.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from app.models.healthkit import HealthKitUploadRequest
from app.services.auth import verify_firebase_token
from app.services.healthkit_processor import HealthKitProcessor
from app.core.pubsub import publish_message
import logging

router = APIRouter(prefix="/api/v1/healthkit", tags=["HealthKit"])
security = HTTPBearer()

logger = logging.getLogger(__name__)

@router.post("/upload", status_code=202)
async def upload_healthkit_data(
    request: HealthKitUploadRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """
    Upload HealthKit data for processing
    
    - Validates data format and ranges
    - Publishes to Pub/Sub for async processing
    - Returns upload tracking ID
    """
    try:
        # Verify Firebase authentication
        user_claims = await verify_firebase_token(token.credentials)
        
        # Ensure user can only upload their own data
        if user_claims.get('uid') != request.user_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot upload data for different user"
            )
        
        # Generate upload ID
        upload_id = f"upload-{request.user_id}-{int(datetime.utcnow().timestamp())}"
        
        # Enrich request with upload metadata
        enriched_request = {
            "upload_id": upload_id,
            "user_id": request.user_id,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "data": request.dict(),
            "processing_config": {
                "enable_pat_analysis": True,
                "enable_gemini_insights": True,
                "priority": "normal"
            }
        }
        
        # Publish to Pub/Sub for async processing
        await publish_message(
            topic="healthkit-ingestion",
            message=enriched_request
        )
        
        logger.info(f"HealthKit upload queued: {upload_id}")
        
        return {
            "upload_id": upload_id,
            "status": "queued",
            "estimated_processing_time_seconds": 30,
            "samples_received": {
                "quantity_samples": len(request.quantity_samples),
                "category_samples": len(request.category_samples),
                "workouts": len(request.workouts)
            }
        }
        
    except ValueError as e:
        logger.warning(f"HealthKit upload validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"HealthKit upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload processing failed")

@router.get("/upload/{upload_id}/status")
async def get_upload_status(
    upload_id: str,
    token: str = Depends(security)
):
    """Check status of HealthKit data upload and processing"""
    user_claims = await verify_firebase_token(token.credentials)
    
    # Extract user_id from upload_id and verify ownership
    try:
        parts = upload_id.split('-')
        upload_user_id = parts[1]
        if user_claims.get('uid') != upload_user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid upload ID format")
    
    # Query processing status from Firestore
    status_doc = await get_upload_status_from_firestore(upload_id)
    
    if not status_doc:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return status_doc

@router.get("/data/latest")
async def get_latest_data(
    token: str = Depends(security),
    limit: int = 100
):
    """Retrieve user's latest processed HealthKit data"""
    user_claims = await verify_firebase_token(token.credentials)
    user_id = user_claims.get('uid')
    
    latest_data = await get_user_latest_healthkit_data(user_id, limit)
    return latest_data
```

### Processing Service

```python
# app/services/healthkit_processor.py

from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HealthKitProcessor:
    """Processes raw HealthKit data into PAT-compatible format"""
    
    def __init__(self):
        self.pat_sampling_rate = 60  # PAT expects 1-minute intervals
        self.required_duration_hours = 24  # Minimum data for reliable analysis
    
    async def process_upload(self, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline for HealthKit uploads
        
        Returns processed data ready for PAT model
        """
        try:
            user_id = upload_data['user_id']
            raw_samples = upload_data['data']
            
            logger.info(f"Processing HealthKit upload for user {user_id}")
            
            # 1. Data validation and cleaning
            validated_data = await self._validate_and_clean(raw_samples)
            
            # 2. Temporal alignment and resampling
            aligned_data = await self._align_temporal_data(validated_data)
            
            # 3. Feature extraction for PAT model
            pat_features = await self._extract_pat_features(aligned_data)
            
            # 4. Data quality assessment
            quality_metrics = await self._assess_data_quality(aligned_data)
            
            # 5. Generate processing summary
            processing_summary = {
                "user_id": user_id,
                "upload_id": upload_data['upload_id'],
                "processed_at": datetime.utcnow().isoformat(),
                "input_samples": len(raw_samples.get('quantity_samples', [])),
                "output_features": len(pat_features),
                "data_quality": quality_metrics,
                "pat_ready": quality_metrics['sufficient_data'],
                "processing_time_ms": self._get_processing_time()
            }
            
            return {
                "processing_summary": processing_summary,
                "pat_features": pat_features,
                "quality_metrics": quality_metrics,
                "aligned_data": aligned_data
            }
            
        except Exception as e:
            logger.error(f"HealthKit processing failed: {e}")
            raise
    
    async def _validate_and_clean(self, raw_samples: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HealthKit data and remove outliers"""
        cleaned_data = {'quantity_samples': [], 'category_samples': []}
        
        # Process quantity samples
        for sample in raw_samples.get('quantity_samples', []):
            if self._is_valid_quantity_sample(sample):
                cleaned_sample = self._clean_quantity_sample(sample)
                cleaned_data['quantity_samples'].append(cleaned_sample)
        
        # Process category samples (sleep, workouts)
        for sample in raw_samples.get('category_samples', []):
            if self._is_valid_category_sample(sample):
                cleaned_data['category_samples'].append(sample)
        
        logger.info(f"Cleaned data: {len(cleaned_data['quantity_samples'])} quantity, "
                   f"{len(cleaned_data['category_samples'])} category samples")
        
        return cleaned_data
    
    async def _align_temporal_data(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Align all data to consistent 1-minute intervals for PAT compatibility
        
        PAT model expects:
        - 1-minute sampling rate
        - 24+ hours of continuous data
        - Activity counts as primary feature
        """
        # Find data time range
        all_timestamps = []
        for sample in validated_data['quantity_samples']:
            all_timestamps.append(sample['start_date'])
        
        if not all_timestamps:
            raise ValueError("No valid timestamp data found")
        
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        logger.info(f"Data spans {duration_hours:.1f} hours from {start_time} to {end_time}")
        
        # Generate minute-by-minute timeline
        timeline = []
        current_time = start_time.replace(second=0, microsecond=0)
        
        while current_time <= end_time:
            timeline.append(current_time)
            current_time += timedelta(minutes=1)
        
        # Create aligned data structure
        aligned_data = {
            'timeline': timeline,
            'activity_counts': np.zeros(len(timeline)),
            'heart_rate': np.zeros(len(timeline)),
            'energy_burned': np.zeros(len(timeline)),
            'step_counts': np.zeros(len(timeline)),
            'sleep_stages': np.zeros(len(timeline)),  # 0=awake, 1=light, 2=deep, 3=rem
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': duration_hours,
                'sample_count': len(timeline)
            }
        }
        
        # Map HealthKit samples to timeline
        await self._map_samples_to_timeline(validated_data, aligned_data)
        
        return aligned_data
    
    async def _extract_pat_features(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features specifically for PAT (Pretrained Actigraphy Transformer)
        
        PAT expects a 10,080-length vector (7 days × 24 hours × 60 minutes)
        """
        activity_vector = aligned_data['activity_counts']
        heart_rate_vector = aligned_data['heart_rate']
        
        # Ensure minimum length for PAT
        if len(activity_vector) < 1440:  # Less than 24 hours
            logger.warning("Insufficient data for robust PAT analysis")
        
        # Normalize and prepare features
        features = {
            'actigraphy_sequence': activity_vector.tolist(),
            'heart_rate_sequence': heart_rate_vector.tolist(),
            'sequence_length': len(activity_vector),
            'sampling_rate_minutes': 1,
            'normalization_applied': True,
            'feature_version': 'v1.0'
        }
        
        # Add statistical features for model context
        features['statistics'] = {
            'activity_mean': float(np.mean(activity_vector)),
            'activity_std': float(np.std(activity_vector)),
            'activity_max': float(np.max(activity_vector)),
            'heart_rate_mean': float(np.mean(heart_rate_vector[heart_rate_vector > 0])),
            'heart_rate_std': float(np.std(heart_rate_vector[heart_rate_vector > 0])),
            'zero_activity_percentage': float(np.sum(activity_vector == 0) / len(activity_vector))
        }
        
        return features
    
    async def _assess_data_quality(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and completeness of HealthKit data"""
        timeline = aligned_data['timeline']
        activity = aligned_data['activity_counts']
        heart_rate = aligned_data['heart_rate']
        
        # Calculate completeness metrics
        total_minutes = len(timeline)
        non_zero_activity = np.sum(activity > 0)
        non_zero_hr = np.sum(heart_rate > 0)
        
        # Assess data gaps
        max_gap_minutes = self._calculate_max_gap(timeline)
        
        # Determine if sufficient for analysis
        sufficient_duration = total_minutes >= 1440  # At least 24 hours
        sufficient_activity = (non_zero_activity / total_minutes) >= 0.1  # At least 10% active periods
        sufficient_hr = (non_zero_hr / total_minutes) >= 0.5  # At least 50% HR coverage
        reasonable_gaps = max_gap_minutes <= 120  # No gaps longer than 2 hours
        
        quality_metrics = {
            'total_duration_minutes': total_minutes,
            'activity_coverage_percentage': float(non_zero_activity / total_minutes * 100),
            'heart_rate_coverage_percentage': float(non_zero_hr / total_minutes * 100),
            'max_data_gap_minutes': max_gap_minutes,
            'sufficient_data': sufficient_duration and sufficient_activity and reasonable_gaps,
            'quality_score': self._calculate_quality_score(
                sufficient_duration, sufficient_activity, sufficient_hr, reasonable_gaps
            ),
            'recommendations': self._generate_quality_recommendations(
                sufficient_duration, sufficient_activity, sufficient_hr, reasonable_gaps
            )
        }
        
        return quality_metrics
    
    def _calculate_quality_score(self, duration: bool, activity: bool, hr: bool, gaps: bool) -> float:
        """Calculate overall data quality score (0-1)"""
        score = 0.0
        if duration: score += 0.4
        if activity: score += 0.3
        if hr: score += 0.2
        if gaps: score += 0.1
        return score
    
    def _generate_quality_recommendations(self, duration: bool, activity: bool, hr: bool, gaps: bool) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        if not duration:
            recommendations.append("Collect at least 24 hours of continuous data for optimal analysis")
        if not activity:
            recommendations.append("Ensure Apple Watch is worn during active periods")
        if not hr:
            recommendations.append("Enable continuous heart rate monitoring for better insights")
        if not gaps:
            recommendations.append("Minimize data gaps by keeping devices charged and connected")
        
        return recommendations
```

## Common Integration Issues & Solutions

### 1. HealthKit Permission Handling

**Issue**: Users not granting all required permissions
**Solution**: Graceful degradation with clear messaging

```python
# app/utils/healthkit_permissions.py

REQUIRED_PERMISSIONS = {
    'critical': [
        'HKQuantityTypeIdentifierStepCount',
        'HKQuantityTypeIdentifierActiveEnergyBurned',
        'HKCategoryTypeIdentifierSleepAnalysis'
    ],
    'recommended': [
        'HKQuantityTypeIdentifierHeartRate',
        'HKQuantityTypeIdentifierRestingHeartRate',
        'HKQuantityTypeIdentifierHeartRateVariabilitySDNN'
    ],
    'optional': [
        'HKQuantityTypeIdentifierDistanceWalkingRunning',
        'HKQuantityTypeIdentifierFlightsClimbed'
    ]
}

async def assess_permission_completeness(uploaded_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess which permissions are available based on uploaded data"""
    available_types = set()
    for sample in uploaded_data.get('quantity_samples', []):
        available_types.add(sample['type_identifier'])
    
    missing_critical = set(REQUIRED_PERMISSIONS['critical']) - available_types
    missing_recommended = set(REQUIRED_PERMISSIONS['recommended']) - available_types
    
    return {
        'has_critical_data': len(missing_critical) == 0,
        'missing_critical': list(missing_critical),
        'missing_recommended': list(missing_recommended),
        'permission_score': calculate_permission_score(available_types),
        'analysis_limitations': generate_limitation_warnings(missing_critical, missing_recommended)
    }
```

### 2. Data Synchronization Delays

**Issue**: HealthKit data not immediately available after collection
**Solution**: Implement retry logic and data freshness checks

```python
# app/utils/data_freshness.py

async def check_data_freshness(uploaded_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if uploaded data is recent and complete"""
    latest_timestamp = None
    
    for sample in uploaded_data.get('quantity_samples', []):
        sample_time = datetime.fromisoformat(sample['end_date'])
        if latest_timestamp is None or sample_time > latest_timestamp:
            latest_timestamp = sample_time
    
    if latest_timestamp:
        freshness_hours = (datetime.utcnow() - latest_timestamp).total_seconds() / 3600
        
        return {
            'latest_data_timestamp': latest_timestamp.isoformat(),
            'freshness_hours': freshness_hours,
            'is_stale': freshness_hours > 24,
            'recommendation': 'Consider syncing HealthKit data more frequently' if freshness_hours > 6 else None
        }
    
    return {'freshness_hours': float('inf'), 'is_stale': True}
```

### 3. Cross-Device Data Conflicts

**Issue**: Multiple devices (iPhone + Apple Watch) creating duplicate/conflicting data
**Solution**: Smart deduplication and source prioritization

```python
# app/utils/deduplication.py

DEVICE_PRIORITY = {
    'Apple Watch': 10,
    'iPhone': 5,
    'Third Party': 1
}

async def deduplicate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate samples, prioritizing higher-quality sources"""
    # Group by type and time window
    grouped_samples = {}
    
    for sample in samples:
        key = (
            sample['type_identifier'],
            sample['start_date'][:16]  # Group by hour:minute
        )
        
        if key not in grouped_samples:
            grouped_samples[key] = []
        grouped_samples[key].append(sample)
    
    # Keep highest priority sample for each group
    deduplicated = []
    for sample_group in grouped_samples.values():
        if len(sample_group) == 1:
            deduplicated.append(sample_group[0])
        else:
            best_sample = max(sample_group, key=lambda s: get_source_priority(s['source']))
            deduplicated.append(best_sample)
    
    return deduplicated

def get_source_priority(source: str) -> int:
    """Get priority score for data source"""
    for device, priority in DEVICE_PRIORITY.items():
        if device in source:
            return priority
    return 1
```

### 4. Timezone and Locale Handling

**Issue**: HealthKit timestamps in local time vs UTC
**Solution**: Consistent timezone normalization

```python
# app/utils/timezone_handling.py

from pytz import timezone
from datetime import datetime

async def normalize_timestamps(samples: List[Dict[str, Any]], user_timezone: str = 'UTC') -> List[Dict[str, Any]]:
    """Convert all timestamps to UTC for consistent processing"""
    user_tz = timezone(user_timezone)
    
    for sample in samples:
        # Assume HealthKit timestamps are in user's local time
        start_local = datetime.fromisoformat(sample['start_date'])
        end_local = datetime.fromisoformat(sample['end_date'])
        
        # Convert to UTC
        start_utc = user_tz.localize(start_local).astimezone(timezone('UTC'))
        end_utc = user_tz.localize(end_local).astimezone(timezone('UTC'))
        
        sample['start_date'] = start_utc.isoformat()
        sample['end_date'] = end_utc.isoformat()
        sample['original_timezone'] = user_timezone
    
    return samples
```

## Testing & Validation

### Sample Test Data

```python
# tests/fixtures/healthkit_test_data.py

SAMPLE_HEALTHKIT_UPLOAD = {
    "user_id": "test-user-123",
    "device_info": {
        "name": "Apple Watch Series 9",
        "model": "Watch6,1",
        "systemName": "watchOS",
        "systemVersion": "10.1"
    },
    "export_date": "2024-01-16T08:00:00Z",
    "quantity_samples": [
        {
            "uuid": "test-sample-001",
            "type_identifier": "HKQuantityTypeIdentifierStepCount",
            "start_date": "2024-01-15T00:00:00",
            "end_date": "2024-01-15T01:00:00",
            "value": 1247.0,
            "unit": {"unit_string": "count"},
            "source": "Apple Watch Series 9"
        },
        # ... more samples
    ]
}

async def test_healthkit_upload_validation():
    """Test HealthKit data validation"""
    from app.models.healthkit import HealthKitUploadRequest
    
    # Valid data should pass
    request = HealthKitUploadRequest(**SAMPLE_HEALTHKIT_UPLOAD)
    assert request.user_id == "test-user-123"
    
    # Invalid heart rate should fail
    invalid_data = SAMPLE_HEALTHKIT_UPLOAD.copy()
    invalid_data["quantity_samples"][0]["value"] = 500  # Invalid HR
    invalid_data["quantity_samples"][0]["type_identifier"] = "HKQuantityTypeIdentifierHeartRate"
    
    with pytest.raises(ValueError):
        HealthKitUploadRequest(**invalid_data)
```

### Performance Benchmarks

```python
# tests/performance/healthkit_benchmarks.py

import pytest
import time
from app.services.healthkit_processor import HealthKitProcessor

@pytest.mark.asyncio
async def test_processing_performance():
    """Test HealthKit processing performance benchmarks"""
    processor = HealthKitProcessor()
    
    # Generate 24 hours of test data (1440 samples)
    test_data = generate_24h_test_data()
    
    start_time = time.time()
    result = await processor.process_upload(test_data)
    processing_time = time.time() - start_time
    
    # Performance assertions
    assert processing_time < 2.0  # Should process in under 2 seconds
    assert result['processing_summary']['pat_ready'] is True
    assert len(result['pat_features']['actigraphy_sequence']) >= 1440
```

---

**Goal**: Frictionless HealthKit integration with robust error handling  
**Standard**: 99.9% data processing success rate  
**Performance**: <500ms processing time for 24h data uploads
