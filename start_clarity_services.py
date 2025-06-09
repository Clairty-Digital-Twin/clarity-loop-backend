#!/usr/bin/env python3
"""CLARITY Platform Service Launcher.

Launches all CLARITY microservices for complete health data processing:
- Main API (FastAPI)
- Analysis Service (Pub/Sub subscriber)
- Insight Service (Gemini AI)

For production, each service would run in separate containers.
For development, this script runs them all locally.
"""

import asyncio
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configurations
SERVICES = {
    "main_api": {
        "command": [
            sys.executable, "-m", "uvicorn",
            "src.clarity.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ],
        "env": {"SERVICE_NAME": "main_api"},
        "description": "Main CLARITY API Server"
    },
    "analysis_service": {
        "command": [
            sys.executable, "-m", "uvicorn",
            "clarity.entrypoints.analysis_service:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ],
        "env": {"SERVICE_NAME": "analysis_service"},
        "description": "Health Data Analysis Service"
    },
    "insight_service": {
        "command": [
            sys.executable, "-m", "uvicorn",
            "clarity.entrypoints.insight_service:app",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--reload"
        ],
        "env": {"SERVICE_NAME": "insight_service"},
        "description": "AI Insight Generation Service"
    }
}


def run_service(service_name: str, config: dict) -> None:
    """Run a single service in a separate process."""
    logger.info(f"Starting {config['description']}...")

    # Set environment variables
    env = os.environ.copy()
    env.update(config["env"])

    try:
        # Run the service
        process = subprocess.run(
            config["command"],
            env=env,
            cwd=Path(__file__).parent,
            check=False
        )

        if process.returncode != 0:
            logger.error(f"Service {service_name} exited with code {process.returncode}")

    except KeyboardInterrupt:
        logger.info(f"Service {service_name} stopped by user")
    except Exception as e:
        logger.error(f"Error running service {service_name}: {e}")


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check if we're in the right directory by looking for pyproject.toml
    if not Path("pyproject.toml").exists():
        logger.error("pyproject.toml not found. Run this script from the project root.")
        return False

    # Check if the main application entrypoint exists
    if not Path("src/clarity/main.py").exists():
        logger.error("src/clarity/main.py not found. The application entrypoint is missing.")
        return False

    # Check if src directory exists
    if not Path("src").exists():
        logger.error("src directory not found.")
        return False

    # Check Python path
    current_dir = Path.cwd()
    src_dir = current_dir / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        logger.info(f"Added {src_dir} to Python path")

    logger.info("✅ Prerequisites check passed")
    return True


def show_status() -> None:
    """Show service status and endpoints."""
    print("\\n" + "=" * 60)
    print("🚀 CLARITY PLATFORM SERVICES STARTED")
    print("=" * 60)
    print("\\n📍 Service Endpoints:")
    print("   • Main API:        http://localhost:8000")
    print("   • API Docs:        http://localhost:8000/docs")
    print("   • Health Check:    http://localhost:8000/health")
    print("   • Analysis Service: http://localhost:8001")
    print("   • Insight Service:  http://localhost:8002")

    print("\\n🔬 Key Features:")
    print("   • HealthKit Data Upload & Processing")
    print("   • Multi-modal Signal Processing (Cardio + Respiratory)")
    print("   • PAT Model Activity Analysis")
    print("   • Transformer-based Feature Fusion")
    print("   • Gemini AI Insight Generation")
    print("   • Pub/Sub Event-driven Architecture")

    print("\\n💡 Test Endpoints:")
    print("   • POST /api/v1/health-data/upload")
    print("   • GET  /api/v1/health-data/query")
    print("   • GET  /api/v1/health-data/insights/{user_id}")

    print("\\n⚡ Environment:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • Working Dir: {Path.cwd()}")
    print(f"   • Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")

    print("\\n🛑 To stop all services: Press Ctrl+C")
    print("=" * 60 + "\\n")


def main() -> None:
    """Main entry point for service launcher."""
    logger.info("🚀 Starting CLARITY Platform Services...")

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Create process pool
    processes = []

    try:
        # Start each service in a separate process
        for service_name, config in SERVICES.items():
            process = multiprocessing.Process(
                target=run_service,
                args=(service_name, config),
                name=f"clarity-{service_name}"
            )
            process.start()
            processes.append(process)

            # Give each service time to start
            time.sleep(2)

        # Wait a bit for all services to start
        time.sleep(3)

        # Show status
        show_status()

        # Wait for all processes
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        logger.info("\\n🛑 Shutting down all services...")

        # Terminate all processes
        for process in processes:
            if process.is_alive():
                logger.info(f"Stopping {process.name}...")
                process.terminate()
                process.join(timeout=5)

                if process.is_alive():
                    logger.warning(f"Force killing {process.name}...")
                    process.kill()

        logger.info("✅ All services stopped")

    except Exception as e:
        logger.error(f"Error starting services: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)
    main()
