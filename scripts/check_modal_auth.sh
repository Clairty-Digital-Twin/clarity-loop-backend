#!/bin/bash
# Script to check Modal authentication issues

echo "🔍 Checking Modal authentication setup..."
echo ""

# Check production health
echo "1️⃣ Checking health endpoint (no auth required):"
http GET https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/health

echo ""
echo "2️⃣ Checking debug echo headers:"
http GET https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/api/v1/debug/echo-headers

echo ""
echo "3️⃣ Checking auth-protected endpoint with fake token:"
http GET https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/api/v1/debug/auth-check \
    Authorization:"Bearer fake.jwt.token"

echo ""
echo "4️⃣ Checking token info endpoint:"
http GET https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/api/v1/debug/token-info \
    Authorization:"Bearer fake.jwt.token"

echo ""
echo "5️⃣ If you have a real token, test it with:"
echo "   export TOKEN='your-firebase-token-here'"
echo "   http GET https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/api/v1/debug/auth-check Authorization:\"Bearer \$TOKEN\""
echo ""
echo "   # Decode the token:"
echo "   echo \$TOKEN | jwt decode -"
echo ""
echo "   # Verify with Firebase CLI:"
echo "   firebase auth:export --project clarity-loop-backend users.json"
echo "   firebase auth:verify --token \"\$TOKEN\""