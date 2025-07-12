#!/usr/bin/env python3
"""Verify Vertex AI Configuration Script

This script performs comprehensive testing of the Vertex AI/Gemini integration
to ensure all configuration issues have been resolved.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.services.gcp_credentials import get_gcp_credentials_manager, initialize_gcp_credentials
from clarity.core.config_aws import get_settings
from clarity.ml.gemini_service import GeminiService
from clarity.api.v1.gemini_insights import get_gemini_service as get_gemini_insights_service
from clarity.api.v1.websocket.chat_handler import get_gemini_service as get_chat_service
from clarity.services.messaging.insight_subscriber import InsightSubscriber

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gcp_credentials():
    """Test GCP credentials configuration."""
    print("\n🔍 Testing GCP Credentials Configuration...")
    
    try:
        # Initialize credentials
        initialize_gcp_credentials()
        
        # Get credentials manager
        manager = get_gcp_credentials_manager()
        
        # Check credentials path
        credentials_path = manager.get_credentials_path()
        print(f"✅ Credentials path: {credentials_path}")
        
        # Check project ID
        project_id = manager.get_project_id()
        print(f"✅ Project ID: {project_id}")
        
        if project_id == "clarity-loop-backend":
            print("✅ Project ID is correct!")
        else:
            print(f"❌ Project ID is incorrect. Expected: clarity-loop-backend, Got: {project_id}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ GCP credentials test failed: {e}")
        return False

def test_settings_configuration():
    """Test settings configuration."""
    print("\n🔍 Testing Settings Configuration...")
    
    try:
        settings = get_settings()
        
        # Check if gcp_project_id is available
        if hasattr(settings, 'gcp_project_id'):
            print(f"✅ Settings has gcp_project_id: {settings.gcp_project_id}")
        else:
            print("❌ Settings missing gcp_project_id")
            return False
            
        # Check vertex_ai_location
        if hasattr(settings, 'vertex_ai_location'):
            print(f"✅ Settings has vertex_ai_location: {settings.vertex_ai_location}")
        else:
            print("❌ Settings missing vertex_ai_location")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

async def test_gemini_service_direct():
    """Test direct GeminiService initialization."""
    print("\n🔍 Testing Direct GeminiService Initialization...")
    
    try:
        # Test with correct project ID
        service = GeminiService(
            project_id="clarity-loop-backend",
            testing=True  # Use testing mode to avoid actual API calls
        )
        
        print(f"✅ GeminiService created with project_id: {service.project_id}")
        print(f"✅ Location: {service.location}")
        print(f"✅ Testing mode: {service.testing}")
        
        # Test initialization
        await service.initialize()
        print("✅ GeminiService initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct GeminiService test failed: {e}")
        return False

def test_gemini_insights_service():
    """Test GeminiService from insights module."""
    print("\n🔍 Testing Gemini Insights Service...")
    
    try:
        # Mock the dependencies
        from clarity.auth.mock_auth import MockAuthProvider
        from clarity.core.config_aws import Settings
        
        class MockConfigProvider:
            def is_development(self):
                return True
                
        from clarity.api.v1.gemini_insights import set_dependencies
        
        # Set up dependencies
        set_dependencies(MockAuthProvider(), MockConfigProvider())
        
        # Get the service
        service = get_gemini_insights_service()
        
        print(f"✅ Gemini Insights service created with project_id: {service.project_id}")
        
        if service.project_id == "clarity-loop-backend":
            print("✅ Gemini Insights service has correct project ID!")
        else:
            print(f"❌ Gemini Insights service has wrong project ID: {service.project_id}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Gemini Insights service test failed: {e}")
        return False

def test_chat_service():
    """Test GeminiService from chat handler."""
    print("\n🔍 Testing Chat Handler Service...")
    
    try:
        service = get_chat_service()
        
        print(f"✅ Chat service created with project_id: {service.project_id}")
        
        if service.project_id == "clarity-loop-backend":
            print("✅ Chat service has correct project ID!")
        else:
            print(f"❌ Chat service has wrong project ID: {service.project_id}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Chat service test failed: {e}")
        return False

def test_insight_subscriber():
    """Test InsightSubscriber service."""
    print("\n🔍 Testing Insight Subscriber Service...")
    
    try:
        # Mock the storage client to avoid actual GCP calls
        from unittest.mock import Mock
        
        subscriber = InsightSubscriber()
        
        print(f"✅ Insight subscriber created with project_id: {subscriber.gemini_service.project_id}")
        
        if subscriber.gemini_service.project_id == "clarity-loop-backend":
            print("✅ Insight subscriber has correct project ID!")
        else:
            print(f"❌ Insight subscriber has wrong project ID: {subscriber.gemini_service.project_id}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Insight subscriber test failed: {e}")
        return False

def check_environment_variables():
    """Check required environment variables."""
    print("\n🔍 Checking Environment Variables...")
    
    required_vars = {
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": "GCP service account JSON",
        "GCP_PROJECT_ID": "GCP project ID (optional)",
        "VERTEX_AI_LOCATION": "Vertex AI location (optional)"
    }
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if var == "GOOGLE_APPLICATION_CREDENTIALS_JSON":
                print(f"✅ {var}: [REDACTED - {len(value)} characters]")
            else:
                print(f"✅ {var}: {value}")
        else:
            if var == "GOOGLE_APPLICATION_CREDENTIALS_JSON":
                print(f"⚠️  {var}: Not set (required for production)")
            else:
                print(f"⚠️  {var}: Not set (using default)")

async def run_all_tests():
    """Run all verification tests."""
    print("🚀 Starting Vertex AI Configuration Verification...")
    
    # Track results
    results = []
    
    # Test GCP credentials
    results.append(("GCP Credentials", test_gcp_credentials()))
    
    # Test settings
    results.append(("Settings Configuration", test_settings_configuration()))
    
    # Test environment variables
    check_environment_variables()
    
    # Test direct GeminiService
    results.append(("Direct GeminiService", await test_gemini_service_direct()))
    
    # Test Gemini Insights service
    results.append(("Gemini Insights Service", test_gemini_insights_service()))
    
    # Test Chat service
    results.append(("Chat Service", test_chat_service()))
    
    # Test Insight Subscriber
    results.append(("Insight Subscriber", test_insight_subscriber()))
    
    # Print summary
    print("\n📊 VERIFICATION RESULTS:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print("=" * 50)
    print(f"Total: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Vertex AI configuration is working correctly.")
        return True
    else:
        print(f"\n❌ {failed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 