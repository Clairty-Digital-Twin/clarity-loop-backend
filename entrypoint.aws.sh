#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# AWS ECS specific environment setup
echo "Starting CLARITY backend on AWS ECS..."

# Use the AWS-specific Gunicorn configuration
exec gunicorn -c /app/gunicorn.aws.conf.py clarity.main:app