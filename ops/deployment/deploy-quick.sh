#!/bin/bash
# Quick deployment script for CLARITY backend
# Usage: ./deploy-quick.sh

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 CLARITY Quick Deploy${NC}"
echo -e "${BLUE}=====================${NC}"

# Configuration
ECR_REPO="124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend"
TAG=$(git rev-parse --short HEAD)

# Check if we have uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${RED}❌ You have uncommitted changes. Please commit first.${NC}"
    exit 1
fi

echo -e "${BLUE}Building image with tag: ${TAG}${NC}"

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO

# Build and push
echo -e "${BLUE}Building for linux/amd64...${NC}"
docker build --platform linux/amd64 -t $ECR_REPO:$TAG -t $ECR_REPO:latest .

echo -e "${BLUE}Pushing to ECR...${NC}"
docker push $ECR_REPO:$TAG
docker push $ECR_REPO:latest

# Deploy
echo -e "${BLUE}Deploying to ECS...${NC}"
TAG=$TAG ops/deploy.sh

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}Image: $ECR_REPO:$TAG${NC}"