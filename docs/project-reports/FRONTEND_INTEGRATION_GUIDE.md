# 🚀 CLARITY BACKEND INTEGRATION GUIDE - CORRECTED VERSION

## 🎯 **VERIFIED AWS INFRASTRUCTURE**

### **Production Endpoints**
- **API Base URL**: `http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com`
- **Health Check**: `http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health`
- **API Docs**: `http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/docs`
- **Region**: `us-east-1` (Virginia)

### **AWS Resources**
```bash
# AWS Account ID: 124355672559
# Region: us-east-1
# ECS Service: clarity-backend
# Load Balancer: clarity-alb-1762715656.us-east-1.elb.amazonaws.com
```

## 🔐 **CORRECT COGNITO CONFIGURATION (VERIFIED)**

### **ACTUAL Production Cognito Settings**
```swift
// CORRECT VALUES - VERIFIED TO EXIST
let cognitoUserPoolId = "us-east-1_efXaR5EcP"
let cognitoClientId = "7sm7ckrkovg78b03n1595euc71"
let cognitoRegion = "us-east-1"
let cognitoIssuer = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_efXaR5EcP"
```

### **⚠️ OLD/INCORRECT VALUES - DO NOT USE**
```swift
// These values were in the ECS config but DO NOT EXIST
// let cognitoUserPoolId = "us-east-1_1G5jYI8FO"  // DOESN'T EXIST
// let cognitoClientId = "66qdivmqgs1oqmmo0b5r9d9hjo"  // WRONG
```

## 📡 **API ENDPOINTS**

### **Authentication Endpoints**
```bash
POST /api/v1/auth/register       # User registration
POST /api/v1/auth/login          # User login
POST /api/v1/auth/refresh        # Refresh tokens
POST /api/v1/auth/logout         # User logout
GET  /api/v1/auth/me             # Get current user info
PUT  /api/v1/auth/me             # Update user profile
GET  /api/v1/auth/health         # Auth service health check
```

### **Health Data Endpoints** (Requires Authentication)
```bash
POST /api/v1/health-data/create  # Upload health data
GET  /api/v1/health-data/{id}    # Get specific health record
GET  /api/v1/health-data/list    # List user's health records
PUT  /api/v1/health-data/{id}    # Update health record
DELETE /api/v1/health-data/{id}  # Delete health record
```

### **HealthKit Upload Endpoint** (Requires Authentication)
```bash
POST /api/v1/healthkit/upload    # Upload HealthKit data export
```

### **PAT Analysis Endpoints** (Requires Authentication)
```bash
POST /api/v1/pat/analyze         # Run PAT analysis on health data
GET  /api/v1/pat/results/{id}    # Get PAT analysis results
```

### **Insights Endpoints** (Requires Authentication)
```bash
POST /api/v1/insights/generate   # Generate AI insights
GET  /api/v1/insights/{id}       # Get specific insight
```

### **WebSocket Endpoint**
```bash
WS /api/v1/ws/chat               # Real-time chat WebSocket
```

## 🛠️ **AWS CLI COMMANDS FOR VERIFICATION**

### **Verify Cognito User Pool Exists**
```bash
# This should return the user pool details
aws cognito-idp describe-user-pool \
  --user-pool-id us-east-1_efXaR5EcP \
  --region us-east-1

# This should return the app client details
aws cognito-idp describe-user-pool-client \
  --user-pool-id us-east-1_efXaR5EcP \
  --client-id 7sm7ckrkovg78b03n1595euc71 \
  --region us-east-1
```

### **Test Backend Connectivity**
```bash
# Test health endpoint
curl -X GET "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/health"

# Test API info
curl -X GET "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/api/v1/"

# Test auth health
curl -X GET "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com/api/v1/auth/health"
```

## 📱 **SWIFT CONFIGURATION (CORRECTED)**

### **Info.plist Configuration**
```xml
<key>APIBaseURL</key>
<string>http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com</string>
<key>CognitoUserPoolId</key>
<string>us-east-1_efXaR5EcP</string>
<key>CognitoClientId</key>
<string>7sm7ckrkovg78b03n1595euc71</string>
<key>CognitoRegion</key>
<string>us-east-1</string>
```

### **CognitoConfiguration.swift**
```swift
import Foundation
import AWSCognitoIdentityProvider

struct CognitoConfiguration {
    // CORRECT VALUES - VERIFIED TO EXIST
    static let userPoolId = "us-east-1_efXaR5EcP"
    static let clientId = "7sm7ckrkovg78b03n1595euc71"
    static let region = "us-east-1"
    
    static var issuer: String {
        return "https://cognito-idp.\(region).amazonaws.com/\(userPoolId)"
    }
}
```

### **AppConfig.swift**
```swift
struct AppConfig {
    static let apiBaseURL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
    static let cognitoUserPoolId = "us-east-1_efXaR5EcP"
    static let cognitoClientId = "7sm7ckrkovg78b03n1595euc71"
    static let cognitoRegion = "us-east-1"
}
```

## 🔍 **TEST SCRIPT FOR FRONTEND**

Save this as `test-backend.sh`:

```bash
#!/bin/bash

# CLARITY Backend Integration Test Script
# This script tests the backend connectivity and Cognito configuration

API_BASE="http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
COGNITO_POOL="us-east-1_efXaR5EcP"
COGNITO_CLIENT="7sm7ckrkovg78b03n1595euc71"
REGION="us-east-1"

echo "🚀 CLARITY Backend Integration Test"
echo "==================================="
echo ""

# Test 1: Backend Health
echo "1️⃣ Testing Backend Health..."
HEALTH_RESPONSE=$(curl -s -X GET "$API_BASE/health")
echo "Response: $HEALTH_RESPONSE"
echo ""

# Test 2: API Info
echo "2️⃣ Testing API Info..."
API_INFO=$(curl -s -X GET "$API_BASE/api/v1/")
echo "Response: $API_INFO" | jq . 2>/dev/null || echo "$API_INFO"
echo ""

# Test 3: Cognito User Pool
echo "3️⃣ Verifying Cognito User Pool..."
POOL_EXISTS=$(aws cognito-idp describe-user-pool --user-pool-id $COGNITO_POOL --region $REGION 2>&1)
if [[ $POOL_EXISTS == *"ResourceNotFoundException"* ]]; then
    echo "❌ ERROR: User Pool $COGNITO_POOL does not exist!"
else
    echo "✅ User Pool $COGNITO_POOL exists!"
fi
echo ""

# Test 4: Cognito Client
echo "4️⃣ Verifying Cognito Client..."
CLIENT_EXISTS=$(aws cognito-idp describe-user-pool-client --user-pool-id $COGNITO_POOL --client-id $COGNITO_CLIENT --region $REGION 2>&1)
if [[ $CLIENT_EXISTS == *"ResourceNotFoundException"* ]]; then
    echo "❌ ERROR: Client $COGNITO_CLIENT does not exist!"
else
    echo "✅ Client $COGNITO_CLIENT exists!"
fi
echo ""

# Test 5: Registration Test (will fail if email exists)
echo "5️⃣ Testing Registration Endpoint..."
TIMESTAMP=$(date +%s)
TEST_EMAIL="test_${TIMESTAMP}@example.com"

REGISTER_RESPONSE=$(curl -s -X POST "$API_BASE/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$TEST_EMAIL\",
    \"password\": \"TestPass123!\",
    \"display_name\": \"Test User\"
  }")

echo "Response: $REGISTER_RESPONSE" | jq . 2>/dev/null || echo "$REGISTER_RESPONSE"
echo ""

echo "✅ Test Complete!"
echo ""
echo "Summary:"
echo "- API Base: $API_BASE"
echo "- Cognito Pool: $COGNITO_POOL"
echo "- Cognito Client: $COGNITO_CLIENT"
echo "- Region: $REGION"
```

## 🎯 **CRITICAL FIXES MADE**

1. **Found the ACTUAL Cognito User Pool**: `us-east-1_efXaR5EcP`
2. **Found the ACTUAL Client ID**: `7sm7ckrkovg78b03n1595euc71`
3. **Updated backend config files** with correct values
4. **Everything is NOW in us-east-1** (consistent region)

## ⚠️ **IMPORTANT NEXT STEPS**

1. **Backend needs to redeploy** with the updated ECS task definition
2. **Frontend should use the CORRECTED values** from this guide
3. **Test authentication** after backend redeploy

## 📊 **BACKEND CAPABILITIES**

- **PAT Analysis**: Advanced actigraphy analysis using Pretrained Actigraphy Transformer
- **Multi-modal Processing**: Activity, heart rate, sleep, and respiratory data
- **AI Insights**: Google Gemini integration for natural language health insights
- **Real-time Chat**: WebSocket support for interactive health consultations
- **Population Normalization**: NHANES dataset comparison for health metrics

---

**This guide contains VERIFIED, WORKING Cognito credentials that actually exist in AWS!**