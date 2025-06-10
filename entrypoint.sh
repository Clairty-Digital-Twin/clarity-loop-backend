#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Start Gunicorn
# The -c flag specifies the Gunicorn configuration file.
# The application is specified as 'module:app_variable'.
# We set the working directory to /app/src to ensure that all
# relative imports within the 'clarity' module work correctly.
exec gunicorn -c /app/gunicorn.conf.py --chdir /app/src clarity.main:app 