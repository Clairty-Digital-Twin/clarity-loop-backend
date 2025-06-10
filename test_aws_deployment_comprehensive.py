#!/usr/bin/env python3
"""Comprehensive AWS Deployment Test Suite for CLARITY Backend.

This script tests all endpoints on the deployed AWS backend to verify functionality.
Tests include health checks, authentication, data storage/retrieval, insights, and profile management.

Usage:
    python test_aws_deployment_comprehensive.py [--base-url URL] [--api-key KEY]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Constants
DEFAULT_BASE_URL = "http://***REMOVED***"
DEFAULT_API_KEY = "production-api-key-change-me"
DEFAULT_TIMEOUT = 30


class AWSBackendTester:
    """Comprehensive test suite for AWS deployed CLARITY backend."""

    def __init__(self, base_url: str, api_key: str):
        """Initialize the tester with base URL and API key."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[Dict[str, Any]] = []
        self.test_user_id = f"test_user_{uuid4().hex[:8]}"
        self.auth_token: Optional[str] = None

    async def __aenter__(self):
        """Enter async context."""
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session:
            await self.session.close()

    def _print_header(self, title: str):
        """Print a section header."""
        print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 {title}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

    def _print_success(self, message: str):
        """Print success message."""
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

    def _print_error(self, message: str):
        """Print error message."""
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

    def _print_warning(self, message: str):
        """Print warning message."""
        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

    def _print_info(self, message: str):
        """Print info message."""
        print(f"{Fore.BLUE}ℹ️  {message}{Style.RESET_ALL}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[int, Dict[str, Any], float]:
        """Make an HTTP request and return status, response, and time."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        url = f"{self.base_url}{endpoint}"
        
        # Set up headers
        request_headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        # Add auth token if available
        if self.auth_token and "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {self.auth_token}"

        start_time = time.time()
        
        try:
            async with self.session.request(
                method, url, headers=request_headers, **kwargs
            ) as response:
                response_time = time.time() - start_time
                
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    data = {"text": await response.text()}

                self.results.append({
                    "endpoint": endpoint,
                    "method": method,
                    "status": response.status,
                    "response_time": response_time,
                    "success": 200 <= response.status < 300
                })

                return response.status, data, response_time

        except Exception as e:
            response_time = time.time() - start_time
            self.results.append({
                "endpoint": endpoint,
                "method": method,
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e)
            })
            return 0, {"error": str(e)}, response_time

    async def test_health_endpoints(self):
        """Test all health check endpoints."""
        self._print_header("HEALTH CHECK ENDPOINTS")

        health_endpoints = [
            ("/health", "Root Health Check"),
            ("/api/v1/auth/health", "Authentication Service"),
            ("/api/v1/health-data/health", "Health Data Service"),
            ("/api/v1/pat/health", "PAT Analysis Service"),
            ("/api/v1/insights/health", "Gemini Insights Service"),
        ]

        all_healthy = True
        for endpoint, description in health_endpoints:
            status, response, response_time = await self._make_request("GET", endpoint)
            
            if 200 <= status < 300:
                self._print_success(
                    f"{description}: {response.get('status', 'healthy')} "
                    f"(Response time: {response_time:.2f}s)"
                )
                
                # Print additional details if available
                if "database" in response:
                    self._print_info(f"  Database: {response['database']}")
                if "authentication" in response:
                    self._print_info(f"  Authentication: {response['authentication']}")
            else:
                self._print_error(
                    f"{description}: Status {status} "
                    f"(Response time: {response_time:.2f}s)"
                )
                if response.get("error"):
                    self._print_info(f"  Error: {response['error']}")
                all_healthy = False

        return all_healthy

    async def test_authentication(self):
        """Test authentication endpoints."""
        self._print_header("AUTHENTICATION ENDPOINTS")

        # Test user registration
        test_email = f"test_{uuid4().hex[:8]}@clarity.health"
        test_password = "TestPassword123!"
        
        register_data = {
            "email": test_email,
            "password": test_password,
            "profile": {
                "first_name": "Test",
                "last_name": "User",
                "date_of_birth": "1990-01-01"
            }
        }

        self._print_info(f"Testing registration with email: {test_email}")
        status, response, response_time = await self._make_request(
            "POST", "/api/v1/auth/register", json=register_data
        )

        if 200 <= status < 300:
            self._print_success(f"User registration successful (Response time: {response_time:.2f}s)")
            if "user_id" in response:
                self.test_user_id = response["user_id"]
                self._print_info(f"  User ID: {self.test_user_id}")
        else:
            self._print_warning(
                f"User registration returned status {status} "
                f"(Response time: {response_time:.2f}s)"
            )
            if response.get("detail"):
                self._print_info(f"  Detail: {response['detail']}")

        # Test login
        login_data = {"email": test_email, "password": test_password}
        
        self._print_info("Testing login...")
        status, response, response_time = await self._make_request(
            "POST", "/api/v1/auth/login", json=login_data
        )

        if 200 <= status < 300:
            self._print_success(f"User login successful (Response time: {response_time:.2f}s)")
            if "access_token" in response:
                self.auth_token = response["access_token"]
                self._print_info("  Access token received")
            if "user" in response:
                self._print_info(f"  User email: {response['user'].get('email')}")
        else:
            self._print_warning(f"User login returned status {status}")
            # Try with mock credentials as fallback
            self._print_info("Attempting login with mock credentials...")
            self.auth_token = "mock-jwt-token"

        # Test get current user
        if self.auth_token:
            self._print_info("Testing get current user info...")
            status, response, response_time = await self._make_request(
                "GET", "/api/v1/auth/me"
            )

            if 200 <= status < 300:
                self._print_success(
                    f"Retrieved current user info (Response time: {response_time:.2f}s)"
                )
                if "user" in response:
                    self._print_info(f"  User ID: {response['user'].get('user_id')}")
                    self._print_info(f"  Email: {response['user'].get('email')}")
            else:
                self._print_warning(f"Get user info returned status {status}")

        return bool(self.auth_token)

    async def test_health_data_storage(self):
        """Test health data upload and retrieval."""
        self._print_header("HEALTH DATA STORAGE")

        if not self.auth_token:
            self._print_warning("Skipping health data tests - no auth token available")
            return False

        # Upload health data
        health_data = {
            "user_id": self.test_user_id,
            "data_type": "heart_rate",
            "measurements": [
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "value": 72.5,
                    "unit": "bpm"
                },
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "value": 75.0,
                    "unit": "bpm"
                }
            ],
            "source": "apple_watch",
            "device_info": {
                "model": "Apple Watch Series 9",
                "os_version": "10.0"
            }
        }

        self._print_info("Uploading health data...")
        status, response, response_time = await self._make_request(
            "POST", "/api/v1/health-data/upload", json=health_data
        )

        processing_id = None
        if 200 <= status < 300:
            self._print_success(f"Health data uploaded (Response time: {response_time:.2f}s)")
            if "processing_id" in response:
                processing_id = response["processing_id"]
                self._print_info(f"  Processing ID: {processing_id}")
            if "status" in response:
                self._print_info(f"  Status: {response['status']}")
        else:
            self._print_error(f"Health data upload failed with status {status}")
            if response.get("detail"):
                self._print_info(f"  Detail: {response['detail']}")
            return False

        # Check processing status
        if processing_id:
            await asyncio.sleep(1)  # Wait a bit for processing
            
            self._print_info("Checking processing status...")
            status, response, response_time = await self._make_request(
                "GET", f"/api/v1/health-data/processing/{processing_id}"
            )

            if 200 <= status < 300:
                self._print_success(
                    f"Retrieved processing status (Response time: {response_time:.2f}s)"
                )
                if "status" in response:
                    self._print_info(f"  Processing status: {response['status']}")
            else:
                self._print_warning(f"Processing status check returned {status}")

        # Query health data
        self._print_info("Querying health data...")
        query_params = {
            "data_type": "heart_rate",
            "limit": 10
        }
        
        status, response, response_time = await self._make_request(
            "GET", "/api/v1/health-data/", params=query_params
        )

        if 200 <= status < 300:
            self._print_success(f"Health data retrieved (Response time: {response_time:.2f}s)")
            if "data" in response:
                self._print_info(f"  Retrieved {len(response['data'])} records")
            if "pagination" in response:
                self._print_info(f"  Has next page: {response['pagination'].get('has_next')}")
        else:
            self._print_error(f"Health data query failed with status {status}")

        return True

    async def test_pat_analysis(self):
        """Test PAT sleep analysis endpoints."""
        self._print_header("PAT SLEEP ANALYSIS")

        if not self.auth_token:
            self._print_warning("Skipping PAT analysis tests - no auth token available")
            return False

        # Sample actigraphy data
        actigraphy_data = {
            "user_id": self.test_user_id,
            "data": [
                {"timestamp": "2025-01-10T22:00:00Z", "activity_level": 0.1},
                {"timestamp": "2025-01-10T22:30:00Z", "activity_level": 0.05},
                {"timestamp": "2025-01-10T23:00:00Z", "activity_level": 0.02},
                {"timestamp": "2025-01-10T23:30:00Z", "activity_level": 0.01},
                {"timestamp": "2025-01-11T00:00:00Z", "activity_level": 0.0},
                {"timestamp": "2025-01-11T06:00:00Z", "activity_level": 0.0},
                {"timestamp": "2025-01-11T06:30:00Z", "activity_level": 0.3},
                {"timestamp": "2025-01-11T07:00:00Z", "activity_level": 0.8}
            ],
            "analysis_type": "sleep_stages"
        }

        self._print_info("Requesting PAT sleep analysis...")
        status, response, response_time = await self._make_request(
            "POST", "/api/v1/pat/analyze", json=actigraphy_data
        )

        if 200 <= status < 300:
            self._print_success(f"PAT analysis completed (Response time: {response_time:.2f}s)")
            if "analysis" in response:
                self._print_info("  Analysis results received")
                if "sleep_metrics" in response.get("analysis", {}):
                    metrics = response["analysis"]["sleep_metrics"]
                    self._print_info(f"  Sleep quality: {metrics.get('sleep_quality')}")
                    self._print_info(f"  Total sleep time: {metrics.get('total_sleep_time')}")
        else:
            self._print_warning(f"PAT analysis returned status {status}")
            if response.get("detail"):
                self._print_info(f"  Detail: {response['detail']}")

        return 200 <= status < 300

    async def test_gemini_insights(self):
        """Test Gemini AI insights generation."""
        self._print_header("GEMINI AI INSIGHTS")

        if not self.auth_token:
            self._print_warning("Skipping Gemini insights tests - no auth token available")
            return False

        # Generate insights request
        insight_request = {
            "analysis_results": {
                "heart_rate_avg": 72,
                "sleep_quality": 0.85,
                "activity_level": "moderate",
                "stress_indicators": ["elevated_hr_variability"]
            },
            "context": "Weekly health summary",
            "insight_type": "comprehensive",
            "include_recommendations": True
        }

        self._print_info("Generating AI health insights...")
        status, response, response_time = await self._make_request(
            "POST", "/api/v1/insights/generate", json=insight_request
        )

        insight_id = None
        if 200 <= status < 300:
            self._print_success(f"AI insights generated (Response time: {response_time:.2f}s)")
            if "data" in response:
                data = response["data"]
                if "narrative" in data:
                    self._print_info(f"  Narrative preview: {data['narrative'][:100]}...")
                if "key_insights" in data:
                    self._print_info(f"  Key insights count: {len(data['key_insights'])}")
                if "recommendations" in data:
                    self._print_info(f"  Recommendations count: {len(data['recommendations'])}")
                if "confidence_score" in data:
                    self._print_info(f"  Confidence score: {data['confidence_score']}")
        else:
            self._print_warning(f"Insights generation returned status {status}")
            if response.get("detail"):
                self._print_info(f"  Detail: {response['detail']}")

        # Test service status
        self._print_info("Checking Gemini service status...")
        status, response, response_time = await self._make_request(
            "GET", "/api/v1/insights/status"
        )

        if 200 <= status < 300:
            self._print_success(f"Service status retrieved (Response time: {response_time:.2f}s)")
            if "data" in response:
                data = response["data"]
                self._print_info(f"  Service status: {data.get('status')}")
                if "model" in data:
                    self._print_info(f"  Model: {data['model'].get('model_name')}")
                    self._print_info(f"  Initialized: {data['model'].get('initialized')}")
        else:
            self._print_warning(f"Service status check returned {status}")

        return True

    async def test_api_documentation(self):
        """Test API documentation endpoints."""
        self._print_header("API DOCUMENTATION")

        docs_endpoints = [
            ("/docs", "Swagger UI"),
            ("/redoc", "ReDoc"),
            ("/openapi.json", "OpenAPI Schema")
        ]

        all_accessible = True
        for endpoint, description in docs_endpoints:
            status, _, response_time = await self._make_request("GET", endpoint)
            
            if 200 <= status < 300:
                self._print_success(
                    f"{description}: Accessible (Response time: {response_time:.2f}s)"
                )
            else:
                self._print_error(f"{description}: Not accessible (Status {status})")
                all_accessible = False

        return all_accessible

    async def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        self._print_header("METRICS ENDPOINT")

        status, response, response_time = await self._make_request("GET", "/metrics")
        
        if 200 <= status < 300:
            self._print_success(f"Metrics endpoint accessible (Response time: {response_time:.2f}s)")
            # Check if response looks like Prometheus metrics
            if "text" in response and "# HELP" in response["text"]:
                self._print_info("  Prometheus metrics format confirmed")
                # Count metric lines
                metric_lines = [
                    line for line in response["text"].split("\n") 
                    if line and not line.startswith("#")
                ]
                self._print_info(f"  Total metrics: {len(metric_lines)}")
        else:
            self._print_error(f"Metrics endpoint returned status {status}")

        return 200 <= status < 300

    def print_summary(self):
        """Print test summary."""
        self._print_header("TEST SUMMARY")

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        # Calculate average response time
        response_times = [r["response_time"] for r in self.results if r["response_time"] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        print(f"\n{Fore.CYAN}📊 Test Results:{Style.RESET_ALL}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {Fore.GREEN}{successful_tests}{Style.RESET_ALL}")
        print(f"  Failed: {Fore.RED}{failed_tests}{Style.RESET_ALL}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Response Time: {avg_response_time:.2f}s")

        # List failed tests
        if failed_tests > 0:
            print(f"\n{Fore.RED}Failed Tests:{Style.RESET_ALL}")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['method']} {result['endpoint']} (Status: {result['status']})")

        # Overall status
        print(f"\n{Fore.CYAN}Overall Status:{Style.RESET_ALL}")
        if success_rate >= 90:
            self._print_success(f"EXCELLENT - Backend is fully operational ({success_rate:.1f}%)")
        elif success_rate >= 70:
            self._print_warning(f"GOOD - Backend is mostly operational ({success_rate:.1f}%)")
        else:
            self._print_error(f"NEEDS ATTENTION - Backend has issues ({success_rate:.1f}%)")

    async def run_all_tests(self):
        """Run all test suites."""
        print(f"{Fore.MAGENTA}{Style.BRIGHT}")
        print("🚀 CLARITY BACKEND AWS DEPLOYMENT TEST SUITE 🚀")
        print(f"Testing backend at: {self.base_url}")
        print(f"API Key: {'*' * 10}{self.api_key[-10:]}")
        print(f"{Style.RESET_ALL}")

        # Run all test suites
        await self.test_health_endpoints()
        await self.test_api_documentation()
        await self.test_authentication()
        await self.test_health_data_storage()
        await self.test_pat_analysis()
        await self.test_gemini_insights()
        await self.test_metrics_endpoint()

        # Print summary
        self.print_summary()


async def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(
        description="Test CLARITY backend AWS deployment"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the backend (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for authentication"
    )
    
    args = parser.parse_args()

    try:
        async with AWSBackendTester(args.base_url, args.api_key) as tester:
            await tester.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Test suite failed: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import aiohttp
        import colorama
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "colorama"])
        import aiohttp
        import colorama

    asyncio.run(main())