#!/usr/bin/env python3
"""
🛠️ Clarity Development CLI Tool
The ultimate developer experience tool for Clarity Loop Backend

Commands:
  start        - Start development environment with hot-reload
  stop         - Stop all development services
  restart      - Restart development environment
  logs         - Stream logs with filtering
  db:seed      - Seed database with test data
  db:reset     - Reset database to clean state
  test:watch   - Run tests on file change
  api:explore  - Open interactive API explorer
  perf:profile - Start performance profiling
  health       - Check all services health
  shell        - Start interactive Python shell
  clean        - Clean up development artifacts
"""

import asyncio
import json
import subprocess
import sys
import time
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any
import webbrowser
import signal
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import print as rprint
import docker
import requests
import httpx

# Initialize CLI
app = typer.Typer(
    name="clarity-dev",
    help="🚀 Ultimate Clarity Development Environment CLI",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.dev.yml"
SERVICES_CONFIG = {
    "clarity-backend": {"port": 8000, "health": "/health"},
    "localstack": {"port": 4566, "health": "/_localstack/health"},
    "postgres-dev": {"port": 5432},
    "redis": {"port": 6379},
    "swagger-ui": {"port": 8001},
    "pgadmin": {"port": 8002},
    "redis-commander": {"port": 8003},
    "filebrowser": {"port": 8004},
    "mailhog": {"port": 8025},
    "grafana": {"port": 3000},
    "prometheus": {"port": 9090},
    "jupyter": {"port": 8888},
}


class DevEnvironment:
    """Manage the development environment"""
    
    def __init__(self):
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception:
            console.print("⚠️  Docker not available", style="yellow")
    
    def is_running(self) -> bool:
        """Check if development environment is running"""
        if not self.docker_client:
            return False
        
        try:
            containers = self.docker_client.containers.list(
                filters={"label": "com.docker.compose.project=clarity-loop-backend"}
            )
            return len(containers) > 0
        except Exception:
            return False
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        status = {}
        
        for service, config in SERVICES_CONFIG.items():
            port = config["port"]
            health_endpoint = config.get("health")
            
            # Check if port is accessible
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    port_status = "✅"
            except:
                port_status = "❌"
                status[service] = {
                    "status": "down",
                    "port": port,
                    "health": "❌",
                    "url": f"http://localhost:{port}"
                }
                continue
            
            # Check health endpoint if available
            health_status = "N/A"
            if health_endpoint:
                try:
                    response = requests.get(
                        f"http://localhost:{port}{health_endpoint}",
                        timeout=2
                    )
                    health_status = "✅" if response.status_code == 200 else "⚠️"
                except:
                    health_status = "❌"
            
            status[service] = {
                "status": "up" if port_status == "✅" else "down",
                "port": port,
                "health": health_status,
                "url": f"http://localhost:{port}"
            }
        
        return status


# Initialize environment manager
dev_env = DevEnvironment()


@app.command()
def start(
    build: bool = typer.Option(False, "--build", help="Force rebuild containers"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in background"),
):
    """🚀 Start development environment with hot-reload"""
    
    with console.status("[bold green]Starting development environment..."):
        console.print(Panel.fit(
            "🚀 Starting Ultimate Development Environment\n"
            "Hot-reload enabled • Zero AWS dependencies • Pure flow state",
            style="bold blue"
        ))
        
        # Build Docker Compose command
        cmd = ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "up"]
        
        if build:
            cmd.append("--build")
        
        if detach:
            cmd.append("-d")
        
        try:
            # Start services
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                console.print("✅ Development environment started successfully!", style="green")
                
                if detach:
                    # Show dashboard after services start
                    time.sleep(5)
                    dashboard()
            else:
                console.print("❌ Failed to start development environment", style="red")
                sys.exit(1)
                
        except KeyboardInterrupt:
            console.print("\n🛑 Shutting down development environment...", style="yellow")
            stop()


@app.command()
def stop():
    """🛑 Stop all development services"""
    
    with console.status("[bold yellow]Stopping development environment..."):
        cmd = ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down"]
        
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                console.print("✅ Development environment stopped", style="green")
            else:
                console.print("⚠️  Some services may still be running", style="yellow")
                
        except Exception as e:
            console.print(f"❌ Error stopping services: {e}", style="red")


@app.command()
def restart(build: bool = typer.Option(False, "--build", help="Force rebuild containers")):
    """🔄 Restart development environment"""
    console.print("🔄 Restarting development environment...", style="blue")
    stop()
    time.sleep(2)
    start(build=build, detach=True)


@app.command()
def logs(
    service: Optional[str] = typer.Argument(None, help="Service to show logs for"),
    follow: bool = typer.Option(True, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(100, "--tail", help="Number of lines to show"),
):
    """📋 Stream logs with filtering"""
    
    cmd = ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "logs"]
    
    if follow:
        cmd.append("--follow")
    
    cmd.extend(["--tail", str(lines)])
    
    if service:
        cmd.append(service)
    
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        console.print("\n🛑 Stopped following logs", style="yellow")


@app.command("db:seed")
def db_seed():
    """🌱 Seed database with test data"""
    
    console.print("🌱 Seeding database with test data...", style="blue")
    
    # Check if services are running
    if not dev_env.is_running():
        console.print("❌ Development environment is not running. Start it first with: clarity-dev start", style="red")
        return
    
    seed_script = PROJECT_ROOT / "dev-scripts" / "seed-data.py"
    
    if seed_script.exists():
        try:
            result = subprocess.run([sys.executable, str(seed_script)], cwd=PROJECT_ROOT)
            if result.returncode == 0:
                console.print("✅ Database seeded successfully!", style="green")
            else:
                console.print("❌ Failed to seed database", style="red")
        except Exception as e:
            console.print(f"❌ Error seeding database: {e}", style="red")
    else:
        console.print("⚠️  Seed script not found, creating sample data via API...", style="yellow")
        
        # Create sample data via API
        try:
            sample_data = {
                "user_id": "testuser@clarity.dev",
                "heart_rate": 72,
                "steps": 8500,
                "sleep_hours": 7.5,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            response = requests.post(
                "http://localhost:8000/api/v1/health-data",
                json=sample_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                console.print("✅ Sample data created successfully!", style="green")
            else:
                console.print(f"⚠️  API returned status {response.status_code}", style="yellow")
                
        except Exception as e:
            console.print(f"❌ Error creating sample data: {e}", style="red")


@app.command("db:reset")
def db_reset():
    """🔄 Reset database to clean state"""
    
    confirm = typer.confirm("⚠️  This will delete all data. Are you sure?")
    if not confirm:
        console.print("Operation cancelled", style="yellow")
        return
    
    console.print("🔄 Resetting database...", style="blue")
    
    # Stop services, remove volumes, restart
    cmd_down = ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down", "-v"]
    cmd_up = ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d"]
    
    try:
        subprocess.run(cmd_down, cwd=PROJECT_ROOT)
        console.print("🗑️  Removed all data volumes", style="yellow")
        
        subprocess.run(cmd_up, cwd=PROJECT_ROOT)
        console.print("✅ Database reset complete!", style="green")
        
        # Re-seed data
        time.sleep(10)  # Wait for services to start
        db_seed()
        
    except Exception as e:
        console.print(f"❌ Error resetting database: {e}", style="red")


@app.command("test:watch")
def test_watch():
    """🧪 Run tests on file change"""
    
    console.print("🧪 Starting test watch mode...", style="blue")
    console.print("Tests will run automatically when files change", style="dim")
    
    try:
        # Use pytest-watch if available
        result = subprocess.run(["ptw", "--runner", "pytest -v --tb=short"])
    except FileNotFoundError:
        # Fallback to basic pytest
        console.print("⚠️  pytest-watch not found, running tests once", style="yellow")
        subprocess.run(["pytest", "-v"])


@app.command("api:explore")
def api_explore():
    """🔍 Open interactive API explorer"""
    
    console.print("🔍 Opening API explorer...", style="blue")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            raise Exception("Backend not healthy")
    except Exception:
        console.print("❌ Backend is not running. Start it first with: clarity-dev start", style="red")
        return
    
    # Open different API exploration tools
    urls = [
        ("📚 FastAPI Docs", "http://localhost:8000/docs"),
        ("🔧 ReDoc", "http://localhost:8000/redoc"),
        ("📊 Swagger UI", "http://localhost:8001"),
    ]
    
    console.print("🌐 Opening API exploration tools:", style="green")
    for name, url in urls:
        console.print(f"  {name}: {url}")
        webbrowser.open(url)


@app.command("perf:profile")
def perf_profile():
    """⚡ Start performance profiling"""
    
    console.print("⚡ Starting performance profiler...", style="blue")
    
    # Run py-spy profiler if available
    try:
        # Find Python process
        result = subprocess.run(
            ["pgrep", "-f", "uvicorn"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]
            console.print(f"📊 Profiling process {pid} for 30 seconds...", style="blue")
            
            profile_output = PROJECT_ROOT / "dev-data" / "profile.svg"
            subprocess.run([
                "py-spy", "record",
                "-o", str(profile_output),
                "-d", "30",
                "-p", pid
            ])
            
            console.print(f"✅ Profile saved to {profile_output}", style="green")
            console.print("🌐 Opening profile in browser...", style="blue")
            webbrowser.open(f"file://{profile_output.absolute()}")
            
        else:
            console.print("❌ No uvicorn process found", style="red")
            
    except FileNotFoundError:
        console.print("⚠️  py-spy not found. Install with: pip install py-spy", style="yellow")


@app.command()
def health():
    """❤️ Check all services health"""
    
    console.print("❤️ Checking services health...", style="blue")
    
    status = dev_env.get_service_status()
    
    # Create status table
    table = Table(title="🏥 Services Health Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Port", justify="center")
    table.add_column("Health", justify="center")
    table.add_column("URL", style="dim")
    
    for service, info in status.items():
        status_icon = "🟢" if info["status"] == "up" else "🔴"
        table.add_row(
            service,
            f"{status_icon} {info['status']}",
            str(info["port"]),
            info["health"],
            info["url"]
        )
    
    console.print(table)
    
    # Overall status
    running_services = sum(1 for s in status.values() if s["status"] == "up")
    total_services = len(status)
    
    if running_services == total_services:
        console.print(f"✅ All {total_services} services are running!", style="green")
    else:
        console.print(f"⚠️  {running_services}/{total_services} services running", style="yellow")


@app.command()
def shell():
    """🐍 Start interactive Python shell with context"""
    
    console.print("🐍 Starting interactive Python shell...", style="blue")
    
    # Set up Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/src:{env.get('PYTHONPATH', '')}"
    
    # Create startup script
    startup_script = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}/src')

# Pre-import common modules
try:
    from clarity.main import app
    from clarity.core.config import get_settings
    from clarity.services.health_data_service import HealthDataService
    print('🚀 Clarity Development Shell')
    print('Available imports:')
    print('  • app - FastAPI application')
    print('  • get_settings() - Configuration')
    print('  • HealthDataService - Health data service')
    print('')
    print('Happy coding! 🎉')
except ImportError as e:
    print(f'⚠️  Some imports failed: {{e}}')
    print('Make sure the development environment is set up correctly')
"""
    
    # Start IPython if available, otherwise regular Python
    try:
        subprocess.run(["ipython", "-c", startup_script, "-i"], env=env)
    except FileNotFoundError:
        subprocess.run(["python", "-c", startup_script, "-i"], env=env)


@app.command()
def dashboard():
    """🎛️ Show development dashboard"""
    
    console.print(Panel.fit(
        "🎛️  DEVELOPMENT DASHBOARD\n\n"
        "🚀 Main App:          http://localhost:8000\n"
        "📚 API Docs:          http://localhost:8000/docs\n"
        "🔍 Health Check:      http://localhost:8000/health\n"
        "📊 Swagger UI:        http://localhost:8001\n"
        "🗄️  Database Admin:    http://localhost:8002\n"
        "🔴 Redis Admin:       http://localhost:8003\n"
        "📁 File Browser:      http://localhost:8004\n"
        "📧 Mail Catcher:      http://localhost:8025\n"
        "📈 Grafana:           http://localhost:3000\n"
        "🎯 Prometheus:        http://localhost:9090\n"
        "📔 Jupyter Lab:       http://localhost:8888\n"
        "🌐 Traefik Dashboard: http://localhost:8080",
        style="bold blue",
        title="🎛️  Development Dashboard"
    ))


@app.command()
def clean():
    """🧹 Clean up development artifacts"""
    
    console.print("🧹 Cleaning up development artifacts...", style="blue")
    
    # Stop services
    stop()
    
    # Remove Docker artifacts
    cleanup_commands = [
        ["docker", "system", "prune", "-f"],
        ["docker", "volume", "prune", "-f"],
        ["docker", "image", "prune", "-f"],
    ]
    
    for cmd in cleanup_commands:
        try:
            subprocess.run(cmd, capture_output=True)
        except Exception:
            pass
    
    # Clean Python artifacts
    import shutil
    patterns = [
        PROJECT_ROOT.glob("**/__pycache__"),
        PROJECT_ROOT.glob("**/*.pyc"),
        PROJECT_ROOT.glob("**/.pytest_cache"),
        PROJECT_ROOT.glob("**/.coverage"),
        PROJECT_ROOT.glob("**/htmlcov"),
        PROJECT_ROOT.glob("**/.mypy_cache"),
        PROJECT_ROOT.glob("**/.ruff_cache"),
    ]
    
    for pattern in patterns:
        for path in pattern:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception:
                pass
    
    console.print("✅ Cleanup complete!", style="green")


def main():
    """Main CLI entry point"""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        console.print("\n🛑 Interrupted by user", style="yellow")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        app()
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()