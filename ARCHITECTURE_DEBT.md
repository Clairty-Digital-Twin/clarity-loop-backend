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

## Architecture Decision (June 11, 2024)
After successful AWS migration with all 60+ endpoints working, we've decided to:

### Keep Gemini API (for now)
- **Rationale**: The Gemini API integration is working well and stable
- **Performance**: No latency issues with cross-cloud calls
- **Cost**: Personal API key keeps costs manageable during development
- **Simplicity**: Avoiding another migration reduces complexity

### Why Not AWS Bedrock?
- Would require rewriting the entire ML service layer
- Bedrock's Anthropic Claude models have different API structure
- Current Gemini integration provides needed functionality
- Can revisit when scaling becomes an issue

### Future Considerations
1. **When to migrate**: If latency becomes an issue or costs escalate
2. **Migration path**: AWS Bedrock with Claude models
3. **Hybrid approach**: Keep Gemini for development, Bedrock for production
4. **Clean architecture**: Current separation makes future migration easier

## Migration Completion Report
- ✅ Firebase Auth → AWS Cognito (fully migrated)
- ✅ Firestore → DynamoDB (fully migrated)
- ✅ All 60+ API endpoints working
- ✅ WebSocket support added
- ✅ ECS deployment stable
- ⚠️  Gemini API remains (architectural decision)

## Don't touch this until deployment is stable!