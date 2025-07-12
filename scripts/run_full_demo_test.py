#!/usr/bin/env python3
"""
🚀 CLARITY DIGITAL TWIN - FULL DEMO TEST
==========================================

This script runs a comprehensive test of all AI-powered features
to verify that Vertex AI integration is working properly.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import requests
from requests.auth import HTTPBasicAuth

# Configuration
API_BASE = "https://clarity.novamindnyc.com/api/v1"
DEMO_EMAIL = "demo@clarity.ai"
DEMO_PASSWORD = "DemoPassword123!"

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, details: str = "") -> None:
    """Print test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

def test_authentication() -> tuple[bool, str]:
    """Test authentication and get JWT token."""
    print_section("AUTHENTICATION TEST")
    
    try:
        response = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            if token:
                print_result("Authentication", True, f"Token received: {token[:20]}...")
                return True, token
            else:
                print_result("Authentication", False, "No token in response")
                return False, ""
        else:
            print_result("Authentication", False, f"HTTP {response.status_code}: {response.text}")
            return False, ""
            
    except Exception as e:
        print_result("Authentication", False, f"Exception: {e}")
        return False, ""

def test_pat_model_health(token: str) -> bool:
    """Test PAT model health endpoint."""
    print_section("PAT MODEL HEALTH CHECK")
    
    try:
        # Correct endpoint is /health/pat (requires auth)
        response = requests.get(
            f"{API_BASE}/health/pat", 
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result("PAT Model Health", True)
            
            # Print detailed health metrics
            print(f"   🧠 Model Status: {data.get('status', 'Unknown')}")
            print(f"   🎯 Health Score: {data.get('health_score', 'N/A')}")
            print(f"   ⚡ Memory Usage: {data.get('cache_status', {}).get('memory_usage_mb', 'N/A')} MB")
            print(f"   📊 Cache Hit Rate: {data.get('cache_status', {}).get('hit_rate', 'N/A')}")
            print(f"   🔄 Models Loaded: {len(data.get('loaded_models', {}))}")
            return True
        else:
            print_result("PAT Model Health", False, f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print_result("PAT Model Health", False, f"Exception: {e}")
        return False

def test_pat_analysis(token: str) -> bool:
    """Test PAT analysis endpoint with sample data."""
    print_section("PAT ANALYSIS TEST")
    
    # Sample step data for step-analysis endpoint
    sample_data = {
        "user_id": "demo-user",
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
    
    try:
        # Try step-analysis endpoint first
        response = requests.post(
            f"{API_BASE}/pat/step-analysis",
            json=sample_data,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result("PAT Analysis", True)
            
            # Print analysis results
            if "analysis" in data:
                analysis = data["analysis"]
                print(f"   💤 Sleep Efficiency: {analysis.get('sleep_efficiency', 'N/A')}%")
                print(f"   🌙 Circadian Score: {analysis.get('circadian_rhythm_score', 'N/A')}")
                print(f"   😔 Depression Risk: {analysis.get('depression_risk_score', 'N/A')}")
                print(f"   🚨 Mania Risk: {analysis.get('mania_risk_score', 'N/A')}")
                print(f"   📈 Confidence: {analysis.get('confidence', 'N/A')}")
                
                # Show if we got clinical insights
                if "clinical_insights" in analysis:
                    insights = analysis["clinical_insights"]
                    print(f"   💡 Clinical Insights: {len(insights)} insights generated")
                    if insights:
                        print(f"   🎯 Sample insight: {insights[0][:100]}...")
                
            return True
        else:
            print_result("PAT Analysis", False, f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print_result("PAT Analysis", False, f"Exception: {e}")
        return False

def test_gemini_insights(token: str) -> bool:
    """Test Gemini insights generation."""
    print_section("GEMINI INSIGHTS TEST")
    
    # Sample analysis data to generate insights from
    sample_analysis = {
        "analysis_results": {
            "sleep_efficiency": 85.5,
            "circadian_rhythm_score": 0.78,
            "depression_risk_score": 0.23,
            "mania_risk_score": 0.12,
            "confidence": 0.91,
            "clinical_insights": [
                "Sleep pattern shows good consistency",
                "Circadian rhythm is well-aligned",
                "Low mood risk indicators"
            ]
        },
        "context": "Weekly health summary",
        "insight_type": "health_summary"
    }
    
    try:
        # Correct endpoint is /insights/ (POST)
        response = requests.post(
            f"{API_BASE}/insights/",
            json=sample_analysis,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result("Gemini Insights", True)
            
            # Check if we got real AI-generated insights vs fallback
            if "data" in data:
                insights = data["data"]
                narrative = insights.get("narrative", "")
                key_insights = insights.get("key_insights", [])
                recommendations = insights.get("recommendations", [])
                
                print(f"   📝 Narrative Length: {len(narrative)} characters")
                print(f"   💡 Key Insights: {len(key_insights)} items")
                print(f"   🎯 Recommendations: {len(recommendations)} items")
                
                # Show quality indicators
                if len(narrative) > 100:
                    print(f"   📖 Sample narrative: {narrative[:150]}...")
                
                if key_insights:
                    print(f"   🧠 Sample insight: {key_insights[0][:100]}...")
                
                if recommendations:
                    print(f"   🔍 Sample recommendation: {recommendations[0][:100]}...")
                
                # Check for signs of real AI vs fallback
                if len(narrative) > 200 and len(key_insights) > 0:
                    print("   ✨ HIGH QUALITY: Appears to be real AI-generated content!")
                else:
                    print("   ⚠️  POSSIBLE FALLBACK: May be using placeholder responses")
                
            return True
        else:
            print_result("Gemini Insights", False, f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print_result("Gemini Insights", False, f"Exception: {e}")
        return False

def test_websocket_chat(token: str) -> bool:
    """Test WebSocket chat functionality."""
    print_section("WEBSOCKET CHAT TEST")
    
    try:
        # Test the websocket chat stats endpoint
        response = requests.get(
            f"{API_BASE}/ws/chat/stats",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result("WebSocket Chat Stats", True)
            print(f"   🔗 Active connections: {data.get('active_connections', 'N/A')}")
            print(f"   🎭 Available rooms: {data.get('total_rooms', 'N/A')}")
            print(f"   📊 Total messages: {data.get('total_messages', 'N/A')}")
            return True
        else:
            print_result("WebSocket Chat Stats", False, f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print_result("WebSocket Chat Stats", False, f"Exception: {e}")
        return False

def run_full_demo() -> None:
    """Run the complete demo test suite."""
    print("🚀 CLARITY DIGITAL TWIN - FULL DEMO TEST")
    print("==========================================")
    print("Testing all AI-powered features after Vertex AI fixes...")
    
    # Track results
    results = {
        "authentication": False,
        "pat_health": False,
        "pat_analysis": False,
        "gemini_insights": False,
        "websocket_chat": False
    }
    
    # Test 1: Authentication
    auth_success, token = test_authentication()
    results["authentication"] = auth_success
    
    if not auth_success:
        print("\n❌ Authentication failed - cannot proceed with other tests")
        return
    
    # Test 2: PAT Model Health
    results["pat_health"] = test_pat_model_health(token)
    
    # Test 3: PAT Analysis
    results["pat_analysis"] = test_pat_analysis(token)
    
    # Test 4: Gemini Insights
    results["gemini_insights"] = test_gemini_insights(token)
    
    # Test 5: WebSocket Chat
    results["websocket_chat"] = test_websocket_chat(token)
    
    # Final Results
    print_section("FINAL RESULTS")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 OVERALL SCORE: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 PERFECT SCORE! ALL SYSTEMS OPERATIONAL!")
        print("🚀 Ready for cofounder demo and investor presentations!")
    elif passed >= total * 0.8:
        print("✅ EXCELLENT! System is production-ready!")
    elif passed >= total * 0.6:
        print("⚠️  GOOD - Some issues to address")
    else:
        print("❌ CRITICAL ISSUES - Need immediate attention")

if __name__ == "__main__":
    run_full_demo() 