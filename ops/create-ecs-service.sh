#!/bin/bash

aws ecs create-service \
  --cluster clarity-backend-cluster \
  --service-name clarity-backend-service \
  --task-definition clarity-backend:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0f5578435b4b48bf2,subnet-09e851182f425a48e],securityGroups=[sg-00feb3a1ae8f40c1e],assignPublicIp=ENABLED}" \
  --region us-east-1