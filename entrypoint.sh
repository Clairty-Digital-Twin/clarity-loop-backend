#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Start Gunicorn
# The -c flag specifies the Gunicorn configuration file.
# The application is specified as 'module:app_variable'.
# Since clarity is installed as a package, we don't need to change directories.
exec gunicorn -c /app/gunicorn.conf.py clarity.main:app 