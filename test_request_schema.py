#!/usr/bin/env python3
"""Test the request schema validation for the PAT step-analysis endpoint."""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator

class StepDataRequest(BaseModel):
    """Request for Apple HealthKit step data analysis."""

    step_counts: list[int] = Field(
        description="Minute-by-minute step counts (10,080 values for 1 week)",
        min_length=1,
        max_length=20160,  # 2 weeks max
    )
    timestamps: list[datetime] = Field(
        description="Corresponding timestamps for each step count"
    )
    user_metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional user demographics (age_group, sex, etc.) - limited size",
        max_length=50,  # Prevent oversized metadata dictionaries
    )

    @validator("timestamps")
    @classmethod
    def validate_timestamps_match_steps(
        cls, v: list[datetime], values: dict[str, Any]
    ) -> list[datetime]:
        """Ensure timestamps match step count length."""
        if "step_counts" in values and len(v) != len(values["step_counts"]):
            msg = "Timestamps length must match step_counts length"
            raise ValueError(msg)
        return v

def test_request_schema():
    """Test the request schema validation."""
    
    # Test valid request
    valid_request = {
        "step_counts": [150, 200, 180, 220, 190, 170, 160, 140, 130, 120],
        "timestamps": [
            "2024-01-15T08:00:00Z",
            "2024-01-15T09:00:00Z", 
            "2024-01-15T10:00:00Z",
            "2024-01-15T11:00:00Z",
            "2024-01-15T12:00:00Z",
            "2024-01-15T13:00:00Z",
            "2024-01-15T14:00:00Z",
            "2024-01-15T15:00:00Z",
            "2024-01-15T16:00:00Z",
            "2024-01-15T17:00:00Z"
        ]
    }
    
    print("🧪 Testing request schema validation...")
    
    try:
        # Test with valid data
        validated = StepDataRequest(**valid_request)
        print("✅ Valid request schema passed validation")
        print(f"   Step counts: {len(validated.step_counts)} values")
        print(f"   Timestamps: {len(validated.timestamps)} values")
        
        # Test with mismatched lengths
        invalid_request = {
            "step_counts": [150, 200, 180],  # 3 values
            "timestamps": ["2024-01-15T08:00:00Z"]  # 1 timestamp
        }
        
        try:
            StepDataRequest(**invalid_request)
            print("❌ Mismatched lengths should have failed validation")
        except ValueError as e:
            print(f"✅ Mismatched lengths correctly failed: {e}")
            
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
    
    # Test JSON serialization
    print("\n📋 Testing JSON serialization...")
    json_str = json.dumps(valid_request, indent=2)
    print(f"Request JSON:\n{json_str}")
    
if __name__ == "__main__":
    test_request_schema() 