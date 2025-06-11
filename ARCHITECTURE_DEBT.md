# CRITICAL ARCHITECTURE DEBT - Google Cloud AI in AWS Deployment

## The Problem
We're using Google Cloud AI services (Vertex AI, Gemini) while deploying to AWS. This creates:
- Confusion about which API keys to use (personal Gemini vs Vertex AI)
- Cross-cloud dependencies
- Unclear billing and security boundaries

## Current State
- Gemini service expects both `vertexai` and `google-generativeai` modules
- Using personal Gemini API key stored in AWS Secrets Manager
- Deployed on AWS ECS but calling Google Cloud services

## TODO (After 60+ endpoints are working)
1. Decide: Keep Google AI or migrate to AWS Bedrock?
2. If keeping Google AI: Document clear separation
3. If migrating: Replace with AWS native AI services
4. Clean up dependencies

## Don't touch this until deployment is stable!