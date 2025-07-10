"""Test data factories for generating realistic test data.

Uses factory_boy to create consistent, realistic test data
for various entities in the CLARITY system.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import factory
from factory import fuzzy
from faker import Faker

fake = Faker()


class UserFactory(factory.Factory):
    """Factory for generating user data."""
    
    class Meta:
        model = dict
    
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    email = factory.LazyFunction(fake.email)
    username = factory.LazyAttribute(lambda obj: obj.email.split("@")[0])
    first_name = factory.LazyFunction(fake.first_name)
    last_name = factory.LazyFunction(fake.last_name)
    phone_number = factory.LazyFunction(fake.phone_number)
    date_of_birth = factory.LazyFunction(lambda: fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat())
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    verified = factory.Faker("boolean", chance_of_getting_true=80)
    role = factory.fuzzy.FuzzyChoice(["patient", "provider", "admin"])
    
    @factory.lazy_attribute
    def attributes(self) -> dict[str, Any]:
        return {
            "height_cm": random.randint(150, 200),
            "weight_kg": round(random.uniform(50, 120), 1),
            "gender": random.choice(["male", "female", "other"]),
            "timezone": fake.timezone(),
        }


class HealthMetricFactory(factory.Factory):
    """Factory for generating health metric data."""
    
    class Meta:
        model = dict
    
    metric_id = factory.LazyFunction(lambda: f"METRIC#{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    metric_type = factory.fuzzy.FuzzyChoice([
        "heart_rate", "blood_pressure", "steps", "sleep_duration",
        "weight", "glucose", "oxygen_saturation", "temperature"
    ])
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    
    @factory.lazy_attribute
    def value(self) -> float | dict[str, float]:
        if self.metric_type == "heart_rate":
            return round(random.uniform(60, 100))
        elif self.metric_type == "blood_pressure":
            return {
                "systolic": random.randint(110, 140),
                "diastolic": random.randint(70, 90),
            }
        elif self.metric_type == "steps":
            return random.randint(0, 20000)
        elif self.metric_type == "sleep_duration":
            return round(random.uniform(4, 10), 1)
        elif self.metric_type == "weight":
            return round(random.uniform(50, 120), 1)
        elif self.metric_type == "glucose":
            return round(random.uniform(70, 140))
        elif self.metric_type == "oxygen_saturation":
            return round(random.uniform(95, 100))
        elif self.metric_type == "temperature":
            return round(random.uniform(36.0, 37.5), 1)
        return 0.0
    
    @factory.lazy_attribute
    def unit(self) -> str:
        units = {
            "heart_rate": "bpm",
            "blood_pressure": "mmHg",
            "steps": "count",
            "sleep_duration": "hours",
            "weight": "kg",
            "glucose": "mg/dL",
            "oxygen_saturation": "%",
            "temperature": "°C",
        }
        return units.get(self.metric_type, "unknown")
    
    device_id = factory.LazyFunction(lambda: f"DEVICE#{fake.uuid4()}")
    device_name = factory.fuzzy.FuzzyChoice([
        "Apple Watch", "Fitbit Charge", "Garmin Venu", "Omron BP Monitor", "Dexcom G6"
    ])
    confidence = factory.fuzzy.FuzzyFloat(0.8, 1.0)


class ActivityDataFactory(factory.Factory):
    """Factory for generating activity/actigraphy data."""
    
    class Meta:
        model = dict
    
    activity_id = factory.LazyFunction(lambda: f"ACTIVITY#{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    start_time = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    
    @factory.lazy_attribute
    def end_time(self) -> str:
        start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        duration = timedelta(minutes=random.randint(5, 120))
        return (start + duration).isoformat()
    
    activity_type = factory.fuzzy.FuzzyChoice([
        "walking", "running", "cycling", "swimming", "gym", "yoga", "sleeping"
    ])
    
    @factory.lazy_attribute
    def metrics(self) -> dict[str, Any]:
        base_metrics = {
            "duration_minutes": random.randint(5, 120),
            "calories_burned": random.randint(50, 500),
        }
        
        if self.activity_type in ["walking", "running", "cycling"]:
            base_metrics.update({
                "distance_km": round(random.uniform(0.5, 20), 2),
                "avg_speed_kmh": round(random.uniform(3, 25), 1),
                "steps": random.randint(100, 20000) if self.activity_type != "cycling" else 0,
            })
        
        if self.activity_type in ["running", "cycling", "gym"]:
            base_metrics.update({
                "avg_heart_rate": random.randint(100, 170),
                "max_heart_rate": random.randint(140, 190),
            })
        
        return base_metrics


class SleepDataFactory(factory.Factory):
    """Factory for generating sleep data."""
    
    class Meta:
        model = dict
    
    sleep_id = factory.LazyFunction(lambda: f"SLEEP#{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    
    @factory.lazy_attribute
    def bedtime(self) -> str:
        # Random bedtime between 9 PM and 1 AM
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        bedtime_hour = random.choice([21, 22, 23, 0, 1])
        if bedtime_hour < 12:
            today += timedelta(days=1)
        bedtime = today.replace(hour=bedtime_hour, minute=random.randint(0, 59))
        return bedtime.isoformat()
    
    @factory.lazy_attribute
    def wake_time(self) -> str:
        bedtime = datetime.fromisoformat(self.bedtime.replace("Z", "+00:00"))
        sleep_duration = timedelta(hours=random.uniform(5, 9))
        return (bedtime + sleep_duration).isoformat()
    
    @factory.lazy_attribute
    def sleep_stages(self) -> dict[str, float]:
        total_hours = random.uniform(5, 9)
        deep_percentage = random.uniform(0.15, 0.25)
        rem_percentage = random.uniform(0.20, 0.30)
        light_percentage = random.uniform(0.40, 0.55)
        awake_percentage = 1.0 - (deep_percentage + rem_percentage + light_percentage)
        
        return {
            "deep": round(total_hours * deep_percentage, 2),
            "rem": round(total_hours * rem_percentage, 2),
            "light": round(total_hours * light_percentage, 2),
            "awake": round(total_hours * awake_percentage, 2),
            "total": round(total_hours, 2),
        }
    
    sleep_efficiency = factory.fuzzy.FuzzyFloat(0.75, 0.95)
    interruptions = factory.fuzzy.FuzzyInteger(0, 5)
    
    @factory.lazy_attribute
    def heart_rate_data(self) -> dict[str, int]:
        return {
            "avg": random.randint(50, 70),
            "min": random.randint(40, 55),
            "max": random.randint(65, 85),
        }


class PATPredictionFactory(factory.Factory):
    """Factory for generating PAT model predictions."""
    
    class Meta:
        model = dict
    
    prediction_id = factory.LazyFunction(lambda: f"PREDICTION#{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    model_version = factory.fuzzy.FuzzyChoice(["1.0", "1.1", "2.0"])
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    
    @factory.lazy_attribute
    def predictions(self) -> dict[str, Any]:
        return {
            "depression_risk": round(random.uniform(0, 1), 3),
            "sleep_quality": round(random.uniform(0, 1), 3),
            "activity_level": random.choice(["low", "moderate", "high"]),
            "health_score": random.randint(60, 100),
        }
    
    @factory.lazy_attribute
    def confidence_scores(self) -> dict[str, float]:
        return {
            "depression_risk": round(random.uniform(0.7, 0.95), 3),
            "sleep_quality": round(random.uniform(0.8, 0.98), 3),
            "activity_level": round(random.uniform(0.85, 0.99), 3),
            "health_score": round(random.uniform(0.75, 0.95), 3),
        }
    
    @factory.lazy_attribute
    def features_used(self) -> list[str]:
        all_features = [
            "sleep_duration", "sleep_efficiency", "activity_counts",
            "heart_rate_variability", "step_count", "sedentary_time",
            "circadian_rhythm", "sleep_regularity", "weekend_difference"
        ]
        num_features = random.randint(5, len(all_features))
        return random.sample(all_features, num_features)


class AnalysisReportFactory(factory.Factory):
    """Factory for generating analysis reports."""
    
    class Meta:
        model = dict
    
    report_id = factory.LazyFunction(lambda: f"REPORT#{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"USER#{fake.uuid4()}")
    report_type = factory.fuzzy.FuzzyChoice([
        "weekly_summary", "monthly_trends", "health_insights", "activity_analysis"
    ])
    generated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    period_start = factory.LazyFunction(
        lambda: (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    )
    period_end = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    
    @factory.lazy_attribute
    def insights(self) -> list[dict[str, str]]:
        insights_pool = [
            {
                "type": "positive",
                "category": "sleep",
                "message": "Your sleep consistency has improved by 15% this week.",
            },
            {
                "type": "suggestion",
                "category": "activity",
                "message": "Try to increase your step count by 1000 steps daily.",
            },
            {
                "type": "warning",
                "category": "heart_rate",
                "message": "Your resting heart rate has been elevated recently.",
            },
            {
                "type": "achievement",
                "category": "goals",
                "message": "You've met your activity goal 6 out of 7 days!",
            },
        ]
        return random.sample(insights_pool, random.randint(2, 4))
    
    @factory.lazy_attribute
    def summary_stats(self) -> dict[str, Any]:
        return {
            "avg_sleep_hours": round(random.uniform(6, 8), 1),
            "avg_steps": random.randint(5000, 12000),
            "active_days": random.randint(4, 7),
            "goal_completion_rate": round(random.uniform(0.6, 0.95), 2),
        }


# Batch data generators


def generate_user_batch(count: int = 10) -> list[dict[str, Any]]:
    """Generate a batch of users."""
    return [UserFactory() for _ in range(count)]


def generate_health_metrics_batch(
    user_id: str, days: int = 7, metrics_per_day: int = 5
) -> list[dict[str, Any]]:
    """Generate health metrics for a user over multiple days."""
    metrics = []
    base_date = datetime.now(timezone.utc)
    
    for day in range(days):
        date = base_date - timedelta(days=day)
        for _ in range(metrics_per_day):
            metric = HealthMetricFactory(user_id=user_id)
            # Override timestamp to spread throughout the day
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            metric["timestamp"] = date.replace(hour=hour, minute=minute).isoformat()
            metrics.append(metric)
    
    return metrics


def generate_activity_timeline(
    user_id: str, days: int = 7
) -> list[dict[str, Any]]:
    """Generate a realistic activity timeline for a user."""
    activities = []
    base_date = datetime.now(timezone.utc)
    
    for day in range(days):
        date = base_date - timedelta(days=day)
        
        # Morning activity (50% chance)
        if random.random() > 0.5:
            morning_activity = ActivityDataFactory(
                user_id=user_id,
                activity_type=random.choice(["walking", "running", "yoga"]),
            )
            morning_time = date.replace(hour=random.randint(6, 9), minute=random.randint(0, 59))
            morning_activity["start_time"] = morning_time.isoformat()
            activities.append(morning_activity)
        
        # Afternoon activity (30% chance)
        if random.random() > 0.7:
            afternoon_activity = ActivityDataFactory(
                user_id=user_id,
                activity_type=random.choice(["walking", "gym", "cycling"]),
            )
            afternoon_time = date.replace(hour=random.randint(12, 17), minute=random.randint(0, 59))
            afternoon_activity["start_time"] = afternoon_time.isoformat()
            activities.append(afternoon_activity)
        
        # Evening activity (40% chance)
        if random.random() > 0.6:
            evening_activity = ActivityDataFactory(
                user_id=user_id,
                activity_type=random.choice(["walking", "gym", "swimming", "yoga"]),
            )
            evening_time = date.replace(hour=random.randint(18, 21), minute=random.randint(0, 59))
            evening_activity["start_time"] = evening_time.isoformat()
            activities.append(evening_activity)
        
        # Sleep data (always present)
        sleep_data = SleepDataFactory(user_id=user_id)
        sleep_time = date.replace(hour=22, minute=random.randint(0, 59))
        sleep_data["bedtime"] = sleep_time.isoformat()
        activities.append(sleep_data)
    
    return activities


def generate_test_dataset(num_users: int = 5) -> dict[str, Any]:
    """Generate a complete test dataset with users and their associated data."""
    dataset = {
        "users": [],
        "health_metrics": [],
        "activities": [],
        "predictions": [],
        "reports": [],
    }
    
    for _ in range(num_users):
        # Create user
        user = UserFactory()
        dataset["users"].append(user)
        
        # Generate health metrics
        metrics = generate_health_metrics_batch(user["user_id"], days=7, metrics_per_day=3)
        dataset["health_metrics"].extend(metrics)
        
        # Generate activity timeline
        activities = generate_activity_timeline(user["user_id"], days=7)
        dataset["activities"].extend(activities)
        
        # Generate predictions
        for _ in range(3):  # 3 predictions per user
            prediction = PATPredictionFactory(user_id=user["user_id"])
            dataset["predictions"].append(prediction)
        
        # Generate reports
        report = AnalysisReportFactory(user_id=user["user_id"])
        dataset["reports"].append(report)
    
    return dataset