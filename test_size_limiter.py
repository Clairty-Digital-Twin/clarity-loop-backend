#!/usr/bin/env python3
"""Test script for request size limiter middleware."""

import requests
import json
import sys

def test_large_payload():
    """Test request size limiter with actual large payload."""
    # Create exactly 6MB of JSON data  
    large_data = {'attack_payload': 'x' * (6 * 1024 * 1024)}
    json_data = json.dumps(large_data)
    
    print(f'Payload size: {len(json_data.encode())} bytes ({len(json_data.encode())/(1024*1024):.1f} MB)')
    
    try:
        response = requests.post('http://localhost:8000/health', 
                               json=large_data, 
                               timeout=10)
        print(f'Response: {response.status_code}')
        print(f'Response body: {response.text[:200]}...')
        
        if response.status_code == 413:
            print('✅ REQUEST SIZE LIMITER WORKING - 413 Payload Too Large')
        else:
            print(f'❌ Expected 413, got {response.status_code}')
            
    except requests.exceptions.Timeout:
        print('⏰ Request timed out (may indicate server hang)')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')

def test_small_payload():
    """Test that small payloads still work."""
    small_data = {'test': 'small payload'}
    
    try:
        response = requests.post('http://localhost:8000/health', 
                               json=small_data, 
                               timeout=5)
        print(f'Small payload test: {response.status_code}')
        
    except Exception as e:
        print(f'Small payload failed: {e}')

if __name__ == "__main__":
    print("🔒 Testing Request Size Limiter...")
    print("\n1. Testing small payload:")
    test_small_payload()
    
    print("\n2. Testing large payload (6MB):")
    test_large_payload() 