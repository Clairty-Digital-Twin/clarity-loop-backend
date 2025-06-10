import os

# Gunicorn configuration for Google Cloud Run
# See: https://cloud.google.com/run/docs/configuring/services/build-and-deploy#container-image

# The address to bind to. Gunicorn will listen on all available network interfaces.
# The port is dynamically set by the 'PORT' environment variable provided by Cloud Run.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# The number of worker processes. A good starting point is 2-4 workers per vCPU core.
# For a typical Cloud Run instance with 1-2 vCPUs, 2 workers is a safe default.
# You can adjust this based on your application's load and performance characteristics.
workers = int(os.environ.get("WEB_CONCURRENCY", 2))

# The type of worker to use. For FastAPI, we use Uvicorn's worker class.
# This allows Gunicorn to manage Uvicorn processes, combining Gunicorn's
# robust process management with Uvicorn's high performance.
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout for workers in seconds. A worker that is silent for this long
# is killed and restarted. This helps prevent frozen workers from holding
# up requests.
timeout = 120

# The maximum number of simultaneous clients. This setting is for Gunicorn's
# synchronous workers and does not directly apply to the Uvicorn worker class,
# but it's good practice to set it.
worker_connections = 1000

# The number of threads per worker. For async workers like Uvicorn, this is
# typically kept at 1, as concurrency is handled by asyncio.
threads = 1

# Restart workers when they exit. This ensures that if a worker crashes for
# any reason, Gunicorn will automatically start a new one, improving fault tolerance.
# Set to True for production deployments.
reload = os.environ.get("GUNICORN_RELOAD", "false").lower() == "true"

# Logging configuration.
# Use '-' to log to stdout, which is standard for containerized applications.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower() 