# 🚀 Clarity Loop Backend - Ultimate Developer Makefile
# Production-grade automation for world-class AI health platform

.PHONY: help install dev test lint format clean docker docs deploy

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project settings
PYTHON_VERSION := 3.11
PROJECT_NAME := clarity-loop-backend
DOCKER_IMAGE := gcr.io/clarity-loop/$(PROJECT_NAME)

help: ## 🚀 Show this help message
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║           🏥 CLARITY LOOP BACKEND - DEVELOPER COMMANDS          ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ===== SETUP & INSTALLATION =====

install: ## 📦 Install all dependencies (Python + Node.js)
	@echo "$(BLUE)Installing Python dependencies...$(RESET)"
	pip install --upgrade pip setuptools wheel
	pip install -e .[dev]
	@echo "$(BLUE)Installing Node.js dependencies...$(RESET)"
	npm install
	@echo "$(GREEN)✅ All dependencies installed!$(RESET)"

install-prod: ## 🏭 Install production dependencies only
	pip install --upgrade pip setuptools wheel
	pip install -e .
	@echo "$(GREEN)✅ Production dependencies installed!$(RESET)"

venv: ## 🐍 Create virtual environment
	python$(PYTHON_VERSION) -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	@echo "$(GREEN)✅ Virtual environment created! Activate with: source venv/bin/activate$(RESET)"

# ===== DEVELOPMENT =====

dev: ## 🔥 Start development server with hot reload
	@echo "$(BLUE)Starting development server...$(RESET)"
	uvicorn clarity.main:app --host 0.0.0.0 --port 8080 --reload --log-level debug

dev-docker: ## 🐳 Start development environment with Docker Compose
	@echo "$(BLUE)Starting development environment...$(RESET)"
	docker-compose up --build

jupyter: ## 📊 Start Jupyter Lab for ML experimentation
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ===== TESTING =====

test: ## 🧪 Run full test suite
	@echo "$(BLUE)Running test suite...$(RESET)"
	pytest -v --cov=src/clarity --cov-report=term-missing --cov-report=html --maxfail=0

test-unit: ## ⚡ Run unit tests only
	pytest tests/unit/ -v

test-integration: ## 🔗 Run integration tests
	./scripts/create_test_user.sh test@example.com TestPassword123!
	pytest tests/integration/ -v

ci-integration: ## 🤖 Seed Cognito user + run integration tests for CI
	./scripts/create_test_user.sh ci_test_user@example.com Sup3r-Secret
	AUTH_BASE_URL=http://localhost:8000 pytest -m integration

test-ml: ## 🤖 Run ML model tests
	pytest tests/ml/ -v -m "pat or gemini"

test-watch: ## 👀 Run tests in watch mode
	pytest-watch -- -v --cov=clarity

test-cov: ## 📊 Generate test coverage report
	pytest --cov=clarity --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(RESET)"

# ===== CODE QUALITY =====

lint: ## 🔍 Run all linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	ruff check .
	black --check .
	mypy src/clarity/
	bandit -r src/clarity/ --configfile pyproject.toml
	safety scan --ignore=51457 --ignore=64459 --ignore=64396 --offline
	@echo "$(GREEN)✅ Markdown linting skipped for pre-production$(RESET)"

lint-fix: ## 🔧 Auto-fix linting issues
	@echo "$(BLUE)Auto-fixing linting issues...$(RESET)"
	ruff check . --fix
	black .
	npm run lint:md:fix

format: ## 🎨 Format code with Black and Ruff
	@echo "$(BLUE)Formatting code...$(RESET)"
	black .
	ruff check . --fix
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

typecheck: ## 🔍 Run type checking with MyPy
	mypy src/clarity/ --strict

security: ## 🛡️ Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r src/clarity/ --configfile pyproject.toml -f json -o reports/bandit-report.json
	safety scan --ignore=51457 --ignore=64459 --ignore=64396 --offline --save-as json reports/safety-report.json
	@echo "$(GREEN)✅ Security checks complete!$(RESET)"

# ===== DOCUMENTATION =====

docs: ## 📚 Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	mkdocs build
	@echo "$(GREEN)✅ Documentation built in site/$(RESET)"

docs-serve: ## 🌐 Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	mkdocs serve

docs-deploy: ## 🚀 Deploy documentation to GitHub Pages
	mkdocs gh-deploy --force

# ===== DOCKER OPERATIONS =====

docker-build: ## 🏗️ Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):latest .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE):latest$(RESET)"

docker-run: ## 🐳 Run Docker container
	docker run -p 8080:8080 --env-file .env $(DOCKER_IMAGE):latest

docker-push: ## 📤 Push Docker image to registry
	docker push $(DOCKER_IMAGE):latest

docker-clean: ## 🧹 Clean Docker images and containers
	docker system prune -f
	docker image prune -f

# ===== DATABASE & MIGRATIONS =====

db-migrate: ## 🗄️ Run database migrations
	alembic upgrade head

db-rollback: ## ⏪ Rollback last migration
	alembic downgrade -1

db-reset: ## 🔄 Reset database (DANGER: deletes all data)
	@echo "$(RED)⚠️  This will delete all data! Are you sure? (y/N)$(RESET)"
	@read confirm && [ "$$confirm" = "y" ]
	alembic downgrade base
	alembic upgrade head

# ===== ML MODEL OPERATIONS =====

train-pat: ## 🧠 Train PAT model
	@echo "$(BLUE)Starting PAT model training...$(RESET)"
	python -m clarity.ml.train --model pat --config configs/pat-training.yaml

evaluate-model: ## 📈 Evaluate model performance
	python -m clarity.ml.evaluate --model pat --dataset test

download-models: ## 📥 Download pre-trained models
	python -m clarity.ml.download --model pat-large-v1.2

# ===== DEPLOYMENT =====

deploy-dev: ## 🚀 Deploy to development environment
	@echo "$(BLUE)Deploying to development...$(RESET)"
	gcloud run deploy clarity-loop-backend-dev \
		--image $(DOCKER_IMAGE):latest \
		--platform managed \
		--region us-central1 \
		--set-env-vars="ENVIRONMENT=development"

deploy-prod: ## 🏭 Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	gcloud run deploy clarity-loop-backend \
		--image $(DOCKER_IMAGE):latest \
		--platform managed \
		--region us-central1 \
		--set-env-vars="ENVIRONMENT=production" \
		--memory 4Gi \
		--cpu 2 \
		--concurrency 1000 \
		--max-instances 100

# ===== UTILITIES =====

clean: ## 🧹 Clean up build artifacts and cache
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

logs: ## 📋 View application logs
	gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=clarity-loop-backend" --limit=50 --format="table(timestamp,severity,textPayload)"

monitor: ## 📊 Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboard...$(RESET)"
	open "https://console.cloud.google.com/run/detail/us-central1/clarity-loop-backend/metrics"

health: ## ❤️ Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	curl -f http://localhost:8080/health || echo "$(RED)❌ Health check failed$(RESET)"

setup-git-hooks: ## 🪝 Set up Git hooks for quality checks
	pre-commit install
	@echo "$(GREEN)✅ Git hooks installed!$(RESET)"

# ===== CI/CD HELPERS =====

ci-install: ## 🤖 Install dependencies for CI
	pip install --upgrade pip setuptools wheel
	pip install -e .[test]

ci-test: ## 🤖 Run tests for CI
	pytest --cov=clarity --cov-report=xml --cov-report=term

ci-lint: ## 🤖 Run linting for CI
	ruff check . --output-format=github
	black --check .
	mypy src/clarity/

# ===== PERFORMANCE =====

benchmark: ## ⚡ Run performance benchmarks
	python -m clarity.benchmarks.api_performance
	python -m clarity.benchmarks.ml_inference

load-test: ## 🔥 Run load tests
	locust -f tests/load/locustfile.py --host=http://localhost:8080

# ===== DATA OPERATIONS =====

generate-synthetic: ## 🎲 Generate synthetic health data
	python -m clarity.data.synthetic --users 1000 --days 30

export-data: ## 📤 Export data for analysis
	python -m clarity.data.export --format parquet --output data/exports/

# ===== QUALITY GATES =====

pre-commit: lint test ## ✅ Run pre-commit quality checks
	@echo "$(GREEN)✅ All quality checks passed!$(RESET)"

full-check: clean install lint test security docs ## 🔍 Run comprehensive quality check
	@echo "$(GREEN)🎉 Full quality check completed successfully!$(RESET)"

# ===== ENVIRONMENT INFO =====

info: ## ℹ️ Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Node: $(shell node --version 2>/dev/null || echo 'Not installed')"
	@echo "NPM: $(shell npm --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Image: $(DOCKER_IMAGE)"
