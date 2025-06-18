#!/usr/bin/env python3
"""
🌱 Development Data Seeding Script
Creates realistic test data for local development
"""

import asyncio
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import httpx
import boto3
from faker import Faker

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
LOCALSTACK_ENDPOINT = "http://localhost:4566"
DYNAMODB_TABLE = "clarity-dev-health-data"
USERS_TABLE = "clarity-dev-users"

fake = Faker()


class DataSeeder:
    """Generate and seed realistic test data"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.dynamodb = boto3.resource(
            'dynamodb',
            endpoint_url=LOCALSTACK_ENDPOINT,
            region_name='us-east-1',
            aws_access_key_id='test',
            aws_secret_access_key='test'
        )
    
    async def seed_users(self, count: int = 10) -> List[Dict[str, Any]]:
        """Create test users"""
        print(f"🧑‍🤝‍🧑 Creating {count} test users...")
        
        users = []
        for i in range(count):
            user = {
                "user_id": f"testuser{i+1}@clarity.dev",
                "email": f"testuser{i+1}@clarity.dev",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "preferences": {
                    "units": random.choice(["metric", "imperial"]),
                    "timezone": fake.timezone(),
                    "notifications": random.choice([True, False])
                }
            }
            users.append(user)
            
            # Store in DynamoDB
            try:
                table = self.dynamodb.Table(USERS_TABLE)
                table.put_item(Item=user)
            except Exception as e:
                print(f"⚠️  Could not store user in DynamoDB: {e}")
        
        print(f"✅ Created {len(users)} users")
        return users
    
    def generate_health_data_points(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Generate realistic health data for a user"""
        data_points = []
        base_date = datetime.utcnow() - timedelta(days=days)
        
        # User-specific baselines for consistency
        base_heart_rate = random.randint(60, 80)
        base_steps = random.randint(6000, 12000)
        base_sleep = random.uniform(6.5, 8.5)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Generate multiple data points per day
            for hour in [8, 12, 16, 20]:  # Morning, noon, afternoon, evening
                timestamp = current_date.replace(hour=hour, minute=random.randint(0, 59))
                
                # Add realistic variations
                heart_rate = max(50, base_heart_rate + random.randint(-15, 25))
                steps = max(0, base_steps + random.randint(-3000, 5000))
                
                # Sleep data only at night
                sleep_hours = None
                if hour == 8:  # Morning sleep data
                    sleep_hours = max(4, base_sleep + random.uniform(-2, 2))
                
                data_point = {
                    "user_id": user_id,
                    "timestamp": timestamp.isoformat() + "Z",
                    "heart_rate": heart_rate,
                    "steps": steps,
                    "data_source": "development_seed",
                    "device_type": random.choice(["apple_watch", "fitbit", "garmin"]),
                    "activity_type": random.choice([
                        "walking", "running", "cycling", "swimming", 
                        "weight_training", "yoga", "rest"
                    ])
                }
                
                if sleep_hours:
                    data_point["sleep_hours"] = sleep_hours
                    data_point["sleep_quality"] = random.choice(["poor", "fair", "good", "excellent"])
                
                # Add some biometric data occasionally
                if random.random() < 0.3:
                    data_point["blood_pressure_systolic"] = random.randint(110, 140)
                    data_point["blood_pressure_diastolic"] = random.randint(70, 90)
                
                if random.random() < 0.2:
                    data_point["weight_kg"] = round(random.uniform(50, 100), 1)
                
                data_points.append(data_point)
        
        return data_points
    
    async def seed_health_data(self, users: List[Dict[str, Any]], days: int = 30):
        """Seed health data for all users"""
        print(f"📊 Generating {days} days of health data for {len(users)} users...")
        
        total_points = 0
        
        for user in users:
            user_id = user["user_id"]
            data_points = self.generate_health_data_points(user_id, days)
            
            # Store in DynamoDB
            try:
                table = self.dynamodb.Table(DYNAMODB_TABLE)
                with table.batch_writer() as batch:
                    for point in data_points:
                        batch.put_item(Item=point)
                
                total_points += len(data_points)
                print(f"  ✅ {user_id}: {len(data_points)} data points")
                
            except Exception as e:
                print(f"  ⚠️  {user_id}: Could not store data - {e}")
                
                # Try via API as fallback
                try:
                    for point in data_points[:10]:  # Limit for API
                        response = await self.http_client.post(
                            f"{API_BASE_URL}/health-data",
                            json=point,
                            timeout=10
                        )
                        if response.status_code not in [200, 201]:
                            print(f"    ⚠️  API error: {response.status_code}")
                            break
                    
                    print(f"  ✅ {user_id}: Stored via API (limited)")
                    
                except Exception as api_e:
                    print(f"  ❌ {user_id}: API also failed - {api_e}")
        
        print(f"✅ Generated {total_points} total health data points")
    
    async def seed_ml_insights(self, users: List[Dict[str, Any]]):
        """Generate sample ML insights and analysis results"""
        print("🧠 Generating sample ML insights...")
        
        insight_templates = [
            "Your sleep quality has improved by {improvement}% over the last week",
            "Your average heart rate during exercise is {heart_rate} BPM, which is {zone} for your age",
            "You've been consistently hitting your step goal {streak} days in a row",
            "Your stress levels appear elevated on {days} - consider relaxation techniques",
            "Your activity pattern suggests you're most energetic during {time_period}",
        ]
        
        for user in users[:5]:  # Generate insights for first 5 users
            user_id = user["user_id"]
            
            for _ in range(random.randint(2, 5)):
                template = random.choice(insight_templates)
                insight = template.format(
                    improvement=random.randint(5, 25),
                    heart_rate=random.randint(120, 160),
                    zone=random.choice(["optimal", "above average", "excellent"]),
                    streak=random.randint(3, 14),
                    days=random.choice(["weekdays", "weekends", "Mondays"]),
                    time_period=random.choice(["morning", "afternoon", "evening"])
                )
                
                insight_data = {
                    "user_id": user_id,
                    "insight_type": random.choice(["sleep", "activity", "heart_rate", "stress"]),
                    "content": insight,
                    "confidence_score": random.uniform(0.7, 0.95),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "source": "development_seed"
                }
                
                # Try to store via API
                try:
                    response = await self.http_client.post(
                        f"{API_BASE_URL}/insights",
                        json=insight_data,
                        timeout=10
                    )
                    if response.status_code in [200, 201]:
                        print(f"  ✅ Generated insight for {user_id}")
                    else:
                        print(f"  ⚠️  API returned {response.status_code} for insight")
                        
                except Exception as e:
                    print(f"  ⚠️  Could not store insight: {e}")
        
        print("✅ Generated sample ML insights")
    
    async def run_full_seed(self, user_count: int = 10, days: int = 30):
        """Run complete data seeding process"""
        print("🚀 Starting development data seeding...")
        print(f"  👥 Users: {user_count}")
        print(f"  📅 Days of data: {days}")
        print()
        
        try:
            # Create users
            users = await self.seed_users(user_count)
            
            # Generate health data
            await self.seed_health_data(users, days)
            
            # Generate ML insights
            await self.seed_ml_insights(users)
            
            print()
            print("🎉 Data seeding complete!")
            print("✅ Your development environment now has realistic test data")
            
        except Exception as e:
            print(f"❌ Error during seeding: {e}")
            raise
        
        finally:
            await self.http_client.aclose()


async def main():
    """Main seeding function"""
    seeder = DataSeeder()
    await seeder.run_full_seed(user_count=10, days=30)


if __name__ == "__main__":
    asyncio.run(main())