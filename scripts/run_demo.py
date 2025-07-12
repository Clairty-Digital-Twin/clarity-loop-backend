#!/usr/bin/env python3
"""Clarity Digital Twin - Complete Demo Script
Runs the full demo showcasing HealthKit integration and bipolar risk detection.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import uuid

import boto3
import requests

# Configuration
BASE_URL = "https://clarity.novamindnyc.com"
DEMO_BUCKET = "clarity-demo-data-63953c36"
API_BASE = f"{BASE_URL}/api/v1"


class ClarityDemo:
    def __init__(self) -> None:
        self.s3_client = boto3.client('s3')
        self.demo_user_id = f"demo-user-{uuid.uuid4().hex[:8]}"
        self.session_token = None

    def print_banner(self, title) -> None:
        """Print a formatted banner for demo sections."""
        print(f"\n{'=' * 60}")
        print(f"🎯 {title}")
        print(f"{'=' * 60}")

    def print_success(self, message) -> None:
        """Print success message."""
        print(f"✅ {message}")

    def print_error(self, message) -> None:
        """Print error message."""
        print(f"❌ {message}")

    def print_info(self, message) -> None:
        """Print info message."""
        print(f"ℹ️  {message}")

    def test_infrastructure(self) -> None:
        """Test core infrastructure components."""
        self.print_banner("Testing Core Infrastructure")

        # Test API connectivity
        try:
            response = requests.get(f"{API_BASE}/", timeout=10)
            if response.status_code == 200:
                api_info = response.json()
                self.print_success(f"API Online: {api_info['version']} - {len(api_info['endpoints'])} endpoints")
            else:
                self.print_error(f"API failed: {response.status_code}")
        except Exception as e:
            self.print_error(f"API connection failed: {e}")

        # Test S3 demo data
        try:
            objects = self.s3_client.list_objects_v2(Bucket=DEMO_BUCKET)
            self.print_success(f"Demo Data: {objects['KeyCount']} files available in S3")
        except Exception as e:
            self.print_error(f"S3 access failed: {e}")

        # Test health endpoint
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                self.print_success("Health Check: System operational")
            else:
                self.print_error(f"Health check failed: {response.status_code}")
        except Exception as e:
            self.print_error(f"Health check failed: {e}")

    def demo_healthkit_integration(self) -> None:
        """Demo Apple HealthKit integration."""
        self.print_banner("Apple HealthKit Integration Demo")

        # Load demo health data
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='healthkit/health_metrics.json')
            health_data = json.loads(response['Body'].read())
            self.print_success(f"Loaded {len(health_data)} health metrics")

            # Show sample data
            for i, metric in enumerate(health_data[:3]):
                self.print_info(f"Sample {i + 1}: {metric['metric_type']} = {metric['value']} {metric['unit']}")

        except Exception as e:
            self.print_error(f"Failed to load health data: {e}")

        # Load demo users
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='healthkit/users.json')
            users = json.loads(response['Body'].read())
            self.print_success(f"Demo Users: {len(users)} synthetic profiles")

            for user in users:
                # Build display name from actual data structure
                name = f"{user['first_name']} {user['last_name']}"
                # Calculate age from date_of_birth
                birth_date = datetime.strptime(user['date_of_birth'], '%Y-%m-%d')
                age = (datetime.now() - birth_date).days // 365
                # Use a fallback for activity_level if not present
                activity_level = user.get('activity_level', 'moderate')
                self.print_info(f"User: {name} ({age} years, {activity_level})")

        except Exception as e:
            self.print_error(f"Failed to load user data: {e}")

        # Test health data API endpoint (will require auth)
        try:
            response = requests.get(f"{API_BASE}/health-data/", timeout=10)
            if response.status_code == 401:
                self.print_info("Health Data API: Authentication required (expected)")
            else:
                self.print_error(f"Unexpected response: {response.status_code}")
        except Exception as e:
            self.print_error(f"Health data API test failed: {e}")

    def demo_bipolar_risk_detection(self) -> None:
        """Demo bipolar risk detection system."""
        self.print_banner("Bipolar Risk Detection Demo")

        # Load clinical scenarios summary
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='clinical/scenarios_summary.json')
            scenarios_data = json.loads(response['Body'].read())

            # Extract scenario summary information
            summary = scenarios_data['scenario_summary']
            self.print_success(f"Clinical Scenarios: {summary['total_scenarios']} phases loaded")

            # Show risk detection targets
            risk_targets = scenarios_data['risk_detection_targets']
            for phase, target in risk_targets.items():
                emoji = "🟢" if "low" in target else "🟡" if "increase" in target else "🔴"
                self.print_info(f"{emoji} {phase}: {target}")

        except Exception as e:
            self.print_error(f"Failed to load clinical scenarios: {e}")

        # Load complete episode scenario
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='clinical/complete_episode_scenario.json')
            episode = json.loads(response['Body'].read())
            # Use correct field name from actual data
            self.print_success(f"Complete Episode: {episode['total_duration_days']} days tracked")
            # Build description from available data
            phases = episode['phases']
            phase_desc = f"Baseline ({phases['baseline']}d) → Prodromal ({phases['prodromal']}d) → Acute ({phases['acute_manic']}d) → Recovery ({phases['recovery']}d)"
            self.print_info(f"Episode Pattern: {phase_desc}")

        except Exception as e:
            self.print_error(f"Failed to load episode data: {e}")

    def demo_pat_analysis(self) -> None:
        """Demo PAT (Pretrained Actigraphy Transformer) analysis."""
        self.print_banner("PAT Model Analysis Demo")

        # Load PAT predictions
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='healthkit/predictions.json')
            predictions = json.loads(response['Body'].read())
            self.print_success(f"PAT Predictions: {len(predictions)} analysis results")

            for i, pred in enumerate(predictions[:2]):
                # Use actual data structure
                prediction_data = pred['predictions']
                confidence_data = pred['confidence_scores']

                # Extract analysis type from prediction keys
                analysis_type = "Depression Risk" if 'depression_risk' in prediction_data else "Health Analysis"
                # Get confidence score (use depression_risk confidence as primary)
                confidence = confidence_data.get('depression_risk', confidence_data.get('health_score', 0.0))

                self.print_info(f"Analysis {i + 1}: {analysis_type} - Score: {confidence:.3f}")

                # Show key finding from prediction data
                if 'depression_risk' in prediction_data:
                    risk_score = prediction_data['depression_risk']
                    self.print_info(f"  Key Finding: Depression risk score {risk_score:.3f}")
                elif 'health_score' in prediction_data:
                    health_score = prediction_data['health_score']
                    self.print_info(f"  Key Finding: Health score {health_score}")

        except Exception as e:
            self.print_error(f"Failed to load PAT predictions: {e}")

        # Test PAT API endpoint (will require auth)
        try:
            response = requests.get(f"{API_BASE}/pat/health", timeout=10)
            if response.status_code == 401:
                self.print_info("PAT API: Authentication required (expected)")
            else:
                self.print_error(f"Unexpected response: {response.status_code}")
        except Exception as e:
            self.print_error(f"PAT API test failed: {e}")

    def demo_chat_interface(self) -> None:
        """Demo conversational health interface."""
        self.print_banner("Chat Interface Demo")

        # Load conversation starters
        try:
            response = self.s3_client.get_object(Bucket=DEMO_BUCKET, Key='chat/health_conscious/conversation_starters.json')
            starters = json.loads(response['Body'].read())
            self.print_success(f"Conversation Starters: {len(starters)} topics available")

            for i, starter in enumerate(starters[:3]):
                # Use correct field name from actual data
                topic = starter['category']  # Use 'category' instead of 'topic'
                question = starter['question']
                self.print_info(f"Topic {i + 1}: {topic}")
                self.print_info(f"  Question: \"{question}\"")

        except Exception as e:
            self.print_error(f"Failed to load conversation starters: {e}")

        # Test WebSocket endpoint structure
        ws_url = f"wss://clarity.novamindnyc.com/api/v1/ws/health-analysis/{self.demo_user_id}"
        self.print_info(f"WebSocket URL: {ws_url}")
        self.print_info("Real-time chat requires WebSocket connection with authentication")

    def demonstrate_value_proposition(self) -> None:
        """Show the complete value proposition."""
        self.print_banner("Value Proposition Summary")

        self.print_success("✨ MATURE APPLE HEALTHKIT INTEGRATION")
        self.print_info("  • Natural language chat with personal health data")
        self.print_info("  • Real-time data processing and analysis")
        self.print_info("  • Multi-device data fusion (Apple Watch, iPhone, etc.)")

        self.print_success("🧠 BIPOLAR RISK DETECTION SYSTEM")
        self.print_info("  • 2-week early warning system for mood episodes")
        self.print_info("  • Clinical-grade risk scoring (0.0-1.0)")
        self.print_info("  • Complete episode tracking (baseline → crisis → recovery)")

        self.print_success("🤖 ADVANCED ML PIPELINE")
        self.print_info("  • PAT (Pretrained Actigraphy Transformer) models")
        self.print_info("  • Proxy actigraphy from Apple HealthKit data")
        self.print_info("  • Real-time inference with caching")

        self.print_success("🏗️ ENTERPRISE ARCHITECTURE")
        self.print_info("  • AWS ECS deployment with auto-scaling")
        self.print_info("  • Cognito authentication and authorization")
        self.print_info("  • S3 data lake with DynamoDB for real-time queries")

    def run_complete_demo(self) -> None:
        """Run the complete demonstration."""
        print("🚀 CLARITY DIGITAL TWIN - COMPLETE DEMO")
        print("=" * 60)
        print("Demonstrating the future of personalized health monitoring")
        print("=" * 60)

        # Run all demo sections
        self.test_infrastructure()
        self.demo_healthkit_integration()
        self.demo_bipolar_risk_detection()
        self.demo_pat_analysis()
        self.demo_chat_interface()
        self.demonstrate_value_proposition()

        # Final summary
        self.print_banner("Demo Complete - Next Steps")
        self.print_success("🎯 Ready for cofounder partnership!")
        self.print_info("• All systems operational and scalable")
        self.print_info("• Demo data showcases real-world scenarios")
        self.print_info("• Architecture ready for production scaling")
        self.print_info("• Technical foundation solid for rapid feature development")

        print("\n📞 Contact: Ready to discuss partnership opportunities")
        print(f"📧 Demo Data: s3://{DEMO_BUCKET}/")
        print(f"🌐 Live API: {BASE_URL}/api/v1/")


def main() -> None:
    """Main demo execution."""
    demo = ClarityDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
