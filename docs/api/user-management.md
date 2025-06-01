# User Management API

This document provides comprehensive documentation for user profile and account management endpoints in the Clarity Loop Backend API.

## Overview

The User Management API handles user profiles, preferences, settings, and account lifecycle operations. It integrates with Firebase Authentication for identity management while maintaining additional health-specific user data and preferences.

## User Data Architecture

### User Profile Hierarchy

```
User Account (Firebase)
├── Profile Information (API)
├── Health Profile (API)
├── Privacy Settings (API)
├── Preferences (API)
├── Subscription Details (API)
└── Data Management (API)
```

### Data Separation

- **Firebase Auth**: Core identity (email, display name, authentication)
- **Firestore Profile**: Extended profile and health information
- **Preferences**: Application settings and personalization
- **Health Data**: Separate collection with access controls

## User Management Endpoints

### Get User Profile

Retrieve comprehensive user profile information for the authenticated user.

#### Request

```http
GET /v1/user/profile
Authorization: Bearer <firebase-jwt-token>
```

#### Query Parameters

- `include_health_profile` (optional): Include health-specific profile data (default: true)
- `include_preferences` (optional): Include user preferences (default: true)
- `include_subscription` (optional): Include subscription details (default: true)

#### Response

```json
{
  "success": true,
  "data": {
    "user_id": "user_12345",
    "basic_profile": {
      "email": "user@example.com",
      "display_name": "John Doe",
      "photo_url": "https://example.com/photos/user_12345.jpg",
      "email_verified": true,
      "phone_number": "+1234567890",
      "phone_verified": false
    },
    "personal_info": {
      "first_name": "John",
      "last_name": "Doe",
      "date_of_birth": "1990-01-15",
      "gender": "male",
      "timezone": "America/New_York",
      "country": "US",
      "language": "en",
      "preferred_units": "metric"
    },
    "health_profile": {
      "height": 175.5,
      "height_unit": "cm",
      "weight": 70.2,
      "weight_unit": "kg",
      "activity_level": "moderate",
      "fitness_goals": ["improve_sleep", "increase_endurance", "weight_maintenance"],
      "chronic_conditions": [],
      "medications": [],
      "allergies": [],
      "emergency_contact": {
        "name": "Jane Doe",
        "relationship": "spouse",
        "phone": "+1234567891"
      }
    },
    "account_info": {
      "account_created": "2023-06-15T10:00:00Z",
      "last_login": "2024-01-20T14:30:00Z",
      "login_count": 145,
      "account_status": "active",
      "verification_status": {
        "email": true,
        "phone": false,
        "identity": false
      }
    },
    "subscription": {
      "tier": "premium",
      "status": "active",
      "started_date": "2023-06-15T10:00:00Z",
      "expires_date": "2024-06-15T10:00:00Z",
      "auto_renewal": true,
      "features": [
        "advanced_insights",
        "data_export",
        "extended_retention",
        "priority_support"
      ]
    },
    "privacy_settings": {
      "data_sharing": {
        "anonymous_research": true,
        "product_improvement": true,
        "marketing_communications": false
      },
      "data_retention": {
        "raw_health_data": "2_years",
        "processed_insights": "5_years",
        "account_data": "indefinite"
      },
      "visibility": {
        "profile_public": false,
        "achievements_public": false,
        "leaderboard_participation": true
      }
    }
  },
  "metadata": {
    "request_id": "req_profile_001",
    "timestamp": "2024-01-20T14:30:00Z",
    "profile_version": "1.2.0",
    "last_updated": "2024-01-18T09:15:00Z"
  }
}
```

### Update User Profile

Update user profile information with validation and change tracking.

#### Request

```http
PUT /v1/user/profile
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "personal_info": {
    "display_name": "John Smith",
    "timezone": "Europe/London",
    "preferred_units": "imperial"
  },
  "health_profile": {
    "weight": 72.5,
    "activity_level": "active",
    "fitness_goals": ["improve_sleep", "increase_strength", "weight_loss"]
  },
  "update_reason": "profile_optimization",
  "notify_changes": true
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "user_id": "user_12345",
    "update_summary": {
      "fields_updated": 5,
      "changes": [
        {
          "field": "personal_info.display_name",
          "old_value": "John Doe",
          "new_value": "John Smith",
          "change_type": "modification"
        },
        {
          "field": "health_profile.weight",
          "old_value": 70.2,
          "new_value": 72.5,
          "change_type": "health_metric_update"
        },
        {
          "field": "health_profile.fitness_goals",
          "added": ["increase_strength", "weight_loss"],
          "removed": ["increase_endurance", "weight_maintenance"],
          "change_type": "goal_modification"
        }
      ]
    },
    "profile_version": "1.3.0",
    "updated_at": "2024-01-20T14:35:00Z",
    "requires_recalibration": {
      "insights_engine": true,
      "recommendation_system": true,
      "goal_tracking": true
    }
  }
}
```

### Update User Preferences

Manage application preferences and notification settings.

#### Request

```http
POST /v1/user/preferences
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "notification_preferences": {
    "push_notifications": {
      "daily_insights": true,
      "goal_achievements": true,
      "health_reminders": false,
      "workout_suggestions": true,
      "sleep_reminders": true
    },
    "email_notifications": {
      "weekly_summaries": true,
      "monthly_reports": true,
      "feature_updates": false,
      "research_invitations": false
    },
    "notification_timing": {
      "quiet_hours_start": "22:00",
      "quiet_hours_end": "07:00",
      "timezone": "America/New_York"
    }
  },
  "app_preferences": {
    "theme": "auto",
    "default_dashboard": "today_overview",
    "chart_preferences": {
      "time_range": "7_days",
      "metrics_visible": ["sleep", "activity", "heart_rate"],
      "comparison_enabled": true
    },
    "units": {
      "distance": "miles",
      "weight": "pounds",
      "temperature": "fahrenheit"
    }
  },
  "insight_preferences": {
    "insight_frequency": "daily",
    "detail_level": "standard",
    "focus_areas": ["sleep_optimization", "fitness_improvement"],
    "coaching_style": "encouraging",
    "include_comparisons": true
  },
  "privacy_preferences": {
    "data_usage_analytics": true,
    "personalized_ads": false,
    "research_participation": "anonymous_only",
    "data_sharing_partners": []
  }
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "preferences_updated": true,
    "changes_applied": 12,
    "effective_immediately": true,
    "sync_to_devices": {
      "ios_app": "pending",
      "watch_app": "pending",
      "web_dashboard": "completed"
    },
    "impact_summary": {
      "notification_changes": "Reduced daily notifications by 2",
      "insight_changes": "Focus areas updated to sleep and fitness",
      "privacy_changes": "Research participation set to anonymous only"
    }
  }
}
```

### Get User Statistics

Retrieve user engagement and usage statistics.

#### Request

```http
GET /v1/user/statistics?period=3months&include_health_metrics=true
Authorization: Bearer <firebase-jwt-token>
```

#### Query Parameters

- `period` (optional): 1month|3months|6months|1year|all_time (default: 3months)
- `include_health_metrics` (optional): Include health data statistics (default: false)
- `include_app_usage` (optional): Include app usage statistics (default: true)

#### Response

```json
{
  "success": true,
  "data": {
    "user_id": "user_12345",
    "analysis_period": "3months",
    "date_range": {
      "start_date": "2023-10-20",
      "end_date": "2024-01-20"
    },
    "engagement_statistics": {
      "app_usage": {
        "total_sessions": 145,
        "average_session_duration": "4.2 minutes",
        "most_used_features": [
          {"feature": "daily_insights", "usage_count": 89},
          {"feature": "health_data_review", "usage_count": 67},
          {"feature": "goal_tracking", "usage_count": 45}
        ],
        "streak_statistics": {
          "current_streak": 12,
          "longest_streak": 28,
          "total_active_days": 78
        }
      },
      "data_contribution": {
        "health_data_uploads": 245,
        "average_daily_data_points": 850,
        "data_consistency_score": 0.89,
        "missing_data_days": 3
      },
      "goal_engagement": {
        "goals_set": 8,
        "goals_achieved": 6,
        "goal_completion_rate": 0.75,
        "average_time_to_goal": "21 days"
      }
    },
    "health_metrics_summary": {
      "data_quality": {
        "overall_score": 0.91,
        "heart_rate_quality": 0.94,
        "activity_quality": 0.88,
        "sleep_quality": 0.92
      },
      "improvement_metrics": {
        "sleep_quality_improvement": "+15%",
        "activity_consistency": "+22%",
        "goal_achievement_rate": "+8%"
      },
      "milestone_achievements": [
        "7-day sleep consistency streak",
        "Monthly step goal achieved 3 months in a row",
        "Improved resting heart rate by 5 BPM"
      ]
    },
    "personalization_effectiveness": {
      "recommendation_follow_rate": 0.68,
      "insight_usefulness_rating": 4.2,
      "goal_relevance_score": 0.84
    }
  }
}
```

### Manage Data Retention

Configure data retention periods and deletion preferences.

#### Request

```http
POST /v1/user/data-retention
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "retention_preferences": {
    "raw_health_data": {
      "retention_period": "2_years",
      "auto_delete": true,
      "export_before_deletion": true
    },
    "processed_insights": {
      "retention_period": "5_years",
      "auto_delete": false,
      "archive_old_data": true
    },
    "activity_logs": {
      "retention_period": "1_year",
      "auto_delete": true
    }
  },
  "deletion_preferences": {
    "secure_deletion": true,
    "deletion_notifications": true,
    "export_before_deletion": true
  }
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "retention_policy_updated": true,
    "effective_date": "2024-01-20T14:30:00Z",
    "impact_summary": {
      "data_affected": {
        "raw_health_data": "~500MB, 50,000 data points",
        "processed_insights": "~50MB, 500 insights",
        "activity_logs": "~10MB, 1,000 log entries"
      },
      "next_auto_deletion": "2026-01-20T00:00:00Z",
      "export_schedule": "30 days before deletion"
    },
    "compliance_note": "Updated retention policy meets GDPR and CCPA requirements"
  }
}
```

### Export User Data

Generate comprehensive data export for user download or transfer.

#### Request

```http
POST /v1/user/data-export
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "export_type": "complete",
  "format": "json",
  "include_data_types": [
    "profile",
    "health_data",
    "insights",
    "preferences",
    "activity_logs"
  ],
  "date_range": {
    "start_date": "2023-01-01",
    "end_date": "2024-01-20"
  },
  "compression": "gzip",
  "encryption": {
    "enabled": true,
    "user_provided_key": false
  }
}
```

#### Response (Async Job)

```json
{
  "success": true,
  "data": {
    "export_job_id": "export_20240120_001",
    "status": "queued",
    "estimated_completion": "2024-01-20T15:00:00Z",
    "estimated_file_size": "25-30 MB",
    "data_points_estimate": 75000,
    "export_summary": {
      "profile_data": "included",
      "health_data_sessions": 245,
      "insights_count": 156,
      "date_range_days": 385
    }
  }
}
```

### Delete User Account

Initiate account deletion process with grace period and data handling options.

#### Request

```http
DELETE /v1/user/account
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "deletion_reason": "privacy_concerns",
  "immediate_deletion": false,
  "grace_period_days": 30,
  "data_handling": {
    "export_before_deletion": true,
    "anonymize_research_data": true,
    "delete_all_data": true
  },
  "confirmation": {
    "user_understands_consequences": true,
    "backup_important_data": true,
    "confirm_phrase": "DELETE MY ACCOUNT"
  }
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "deletion_request_id": "del_req_20240120_001",
    "deletion_scheduled": "2024-02-19T14:30:00Z",
    "grace_period_expires": "2024-02-19T14:30:00Z",
    "immediate_actions": {
      "account_deactivated": true,
      "data_export_initiated": true,
      "services_access_revoked": true
    },
    "cancellation_info": {
      "can_cancel_until": "2024-02-18T14:30:00Z",
      "cancellation_url": "https://app.clarityloop.com/account/restore",
      "email_notifications": true
    },
    "data_handling_summary": {
      "personal_data": "Will be permanently deleted",
      "anonymized_research_data": "Will be retained for research",
      "export_file": "Available for 30 days"
    }
  }
}
```

## Account Verification and Security

### Verify Email Address

Send verification email and confirm email ownership.

#### Request

```http
POST /v1/user/verify/email
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "email": "newemail@example.com",
  "verification_type": "change_email"
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "verification_sent": true,
    "email": "newemail@example.com",
    "verification_code_length": 6,
    "expires_in": 900,
    "attempt_limit": 3,
    "cooldown_period": 300
  }
}
```

### Update Password Requirements

Set or update password security requirements.

#### Request

```http
POST /v1/user/security/password-policy
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "require_two_factor": true,
  "password_strength": "high",
  "password_expiry_days": 90,
  "password_history_check": 5
}
```

## Error Handling

### Profile Update Validation Errors

```json
{
  "error": {
    "code": "PROFILE_VALIDATION_ERROR",
    "message": "Profile update contains invalid data",
    "details": {
      "validation_errors": [
        {
          "field": "health_profile.weight",
          "error": "Weight must be between 20-500 kg",
          "provided_value": 600
        },
        {
          "field": "personal_info.date_of_birth",
          "error": "Birth date cannot be in the future",
          "provided_value": "2025-01-01"
        }
      ]
    }
  }
}
```

### Account State Errors

```json
{
  "error": {
    "code": "ACCOUNT_STATE_ERROR",
    "message": "Account is in a state that prevents this action",
    "details": {
      "current_state": "deletion_pending",
      "action_attempted": "profile_update",
      "resolution": "Cancel account deletion to enable profile updates",
      "support_contact": "support@clarityloop.com"
    }
  }
}
```

## Rate Limiting

### Endpoint Rate Limits

- **Profile Updates**: 10 per hour per user
- **Preference Updates**: 20 per hour per user
- **Data Exports**: 3 per day per user
- **Account Deletion**: 1 per 24 hours per user
- **Email Verification**: 5 per hour per user

This comprehensive user management API documentation provides all necessary endpoints for complete user account lifecycle management with robust security and privacy controls.
