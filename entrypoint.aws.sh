#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# AWS ECS specific environment setup
echo "Starting CLARITY backend on AWS ECS..."

# Use the module specified in environment or default to NUCLEAR AWS main
MAIN_MODULE=${CLARITY_MAIN_MODULE:-clarity.main_aws_nuclear:app}

# Use the AWS-specific Gunicorn configuration
exec gunicorn -c /app/gunicorn.aws.conf.py $MAIN_MODULE