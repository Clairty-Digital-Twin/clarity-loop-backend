# Clarity Loop Backend Documentation

This directory contains comprehensive documentation for the Clarity Loop Backend project.

## Documentation Structure

- **api/** - API specifications, endpoints documentation, and data schemas
- **architecture/** - System architecture diagrams, component interactions, and design decisions
- **development/** - Development guides, setup instructions, and best practices
- **integrations/** - Third-party service integrations (Firebase, HealthKit, etc.)
- **literature/** - Research papers and literature references

## Key Documents

### Implementation Guides

- [Implementation Blueprint](./blueprint.md) - Comprehensive end-to-end implementation plan
- [Quick Start Guide](./quickstart.md) - Getting started with the project

### Core Technical Documentation

- [ML API Endpoints](./api/ml-endpoints.md) - Machine learning API specifications with PAT integration
- [PAT Model Deployment](./development/model-deployment.md) - **Pretrained Actigraphy Transformer deployment guide**
- [ML Pipeline Development](./development/ml-pipeline.md) - ML service implementation details
- [HealthKit Integration](./integrations/healthkit.md) - Apple HealthKit data processing

### API Documentation

- [Authentication](./api/authentication.md) - Firebase Auth integration
- [Health Data APIs](./api/health-data.md) - Health data endpoints
- [Insights APIs](./api/insights.md) - AI-powered insights endpoints
- [User Management](./api/user-management.md) - User account management

### Architecture & Design

- [Data Models](./architecture/data-models.md) - Database schemas and data structures
- [System Architecture](./architecture/) - Overall system design

## Pretrained Actigraphy Transformer (PAT)

The PAT integration is a core component of our ML pipeline:

- **Model Weights**: Located in `/models/` directory (PAT-L, PAT-M, PAT-S variants)
- **Research Source**: Dartmouth College (29,307 participants, NHANES 2003-2014)
- **Paper**: "AI Foundation Models for Wearable Movement Data in Mental Health Research" (arXiv:2411.15240)
- **Implementation**: [Model Deployment Guide](./development/model-deployment.md)

## External References

- [External Links & Resources](./external-links.md) - Useful external documentation and tools
