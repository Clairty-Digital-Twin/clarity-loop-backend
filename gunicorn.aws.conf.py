import multiprocessing
import os

# Gunicorn configuration optimized for AWS ECS/Fargate
# Designed for containerized deployment with proper health checks

# Bind to all interfaces on the port specified by environment variable
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Workers configuration
# AWS ECS task CPU units: 256 (.25 vCPU), 512 (.5 vCPU), 1024 (1 vCPU), etc.
# Recommended: 2-4 workers per vCPU
cpu_units = int(os.environ.get("AWS_ECS_CPU_UNITS", "1024"))
vcpus = cpu_units / 1024
workers = int(os.environ.get("WEB_CONCURRENCY", max(2, min(4, int(vcpus * 2)))))

# Use Uvicorn worker for FastAPI async support
worker_class = "uvicorn.workers.UvicornWorker"

# Worker timeout - AWS ALB default timeout is 60s, so we set higher
timeout = int(os.environ.get("WORKER_TIMEOUT", "120"))

# Connection handling
worker_connections = 1000
keepalive = 65  # Slightly higher than ALB idle timeout (60s)

# Thread configuration (not used by async workers but good to set)
threads = 1

# Restart workers periodically to prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Preload application for better memory efficiency
preload_app = os.environ.get("GUNICORN_PRELOAD", "true").lower() == "true"

# Graceful shutdown timeout
graceful_timeout = 30

# Logging configuration
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# StatsD integration for CloudWatch metrics (optional)
if os.environ.get("STATSD_HOST"):
    statsd_host = (
        f"{os.environ.get('STATSD_HOST')}:{os.environ.get('STATSD_PORT', '8125')}"
    )
    statsd_prefix = "clarity.backend"

# Custom settings for the application
raw_env = [
    f"ENVIRONMENT={os.environ.get('ENVIRONMENT', 'production')}",
    f"AWS_REGION={os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')}",
]


# Worker lifecycle hooks for monitoring
def worker_int(worker):
    """Called just before a worker is killed."""
    worker.log.info("Worker received INT or QUIT signal")


def worker_abort(worker):
    """Called when a worker is killed by timeout."""
    worker.log.error(f"Worker timeout, aborting: {worker.pid}")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"Forking worker: {worker}")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned: {worker.pid}")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready. Spawning workers")


def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down server")
