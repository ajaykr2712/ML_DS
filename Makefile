# ML Arsenal - Comprehensive Makefile
# =====================================
# This Makefile provides automation for common development, testing, and deployment tasks
# Usage: make <target> [ARGS="arguments"]

# Configuration
.PHONY: help install install-dev install-gpu setup clean test lint format type-check \
        benchmark docs serve migrate deploy build-docker run-docker docker-compose \
        notebooks train evaluate deploy-local deploy-cloud security audit

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Project configuration
PROJECT_NAME := ml_arsenal
PYTHON_VERSION := 3.9
VENV_NAME := venv
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_REGISTRY := your-registry.com
KUBERNETES_NAMESPACE := ml-arsenal

# Python and environment
PYTHON := python$(PYTHON_VERSION)
PIP := $(VENV_NAME)/bin/pip
PYTHON_VENV := $(VENV_NAME)/bin/python
PYTEST := $(VENV_NAME)/bin/pytest
BLACK := $(VENV_NAME)/bin/black
ISORT := $(VENV_NAME)/bin/isort
FLAKE8 := $(VENV_NAME)/bin/flake8
MYPY := $(VENV_NAME)/bin/mypy

# Source directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
SCRIPTS_DIR := scripts

##@ Help
help: ## Display this help
	@echo "$(CYAN)ML Arsenal - Development Automation$(NC)"
	@echo "$(CYAN)===================================$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(CYAN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Environment
install: ## Install project dependencies
	@echo "$(GREEN)Installing ML Arsenal dependencies...$(NC)"
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@$(PIP) install -r requirements-dev.txt
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

install-gpu: ## Install GPU-enabled dependencies
	@echo "$(GREEN)Installing GPU dependencies...$(NC)"
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .
	@echo "$(GREEN)✅ GPU environment ready!$(NC)"

setup: install-dev ## Complete development setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@$(VENV_NAME)/bin/pre-commit install
	@mkdir -p data/{raw,processed,interim,external,features,samples}
	@mkdir -p models/{trained,checkpoints,experiments,production,benchmarks}
	@mkdir -p experiments/{mlruns,wandb,tensorboard}
	@mkdir -p reports/{performance,figures,benchmarks}
	@touch data/.gitkeep models/.gitkeep experiments/.gitkeep reports/.gitkeep
	@echo "$(GREEN)✅ Development environment setup complete!$(NC)"

clean: ## Clean temporary files and caches
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name ".coverage" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf build/ dist/ htmlcov/
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

##@ Code Quality
lint: ## Run all linting tools
	@echo "$(BLUE)Running linting tools...$(NC)"
	@$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	@$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	@$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Code formatting complete!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	@$(MYPY) $(SRC_DIR)
	@echo "$(GREEN)✅ Type checking complete!$(NC)"

quality: lint type-check ## Run all quality checks
	@echo "$(GREEN)✅ All quality checks passed!$(NC)"

##@ Testing
test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(NC)"
	@$(PYTEST) $(TEST_DIR) -v --tb=short
	@echo "$(GREEN)✅ All tests passed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/unit -v
	@echo "$(GREEN)✅ Unit tests passed!$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/integration -v
	@echo "$(GREEN)✅ Integration tests passed!$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/performance -v --benchmark-only
	@echo "$(GREEN)✅ Performance tests complete!$(NC)"

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/e2e -v -s
	@echo "$(GREEN)✅ End-to-end tests passed!$(NC)"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

##@ Benchmarking
benchmark: ## Run comprehensive benchmarks
	@echo "$(BLUE)Running comprehensive benchmarks...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/benchmarks/run_all_benchmarks.py
	@echo "$(GREEN)✅ Benchmarks complete!$(NC)"

benchmark-classical: ## Benchmark classical ML algorithms
	@echo "$(BLUE)Benchmarking classical ML algorithms...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/benchmarks/benchmark_classical.py
	@echo "$(GREEN)✅ Classical ML benchmarks complete!$(NC)"

benchmark-deep: ## Benchmark deep learning models
	@echo "$(BLUE)Benchmarking deep learning models...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/benchmarks/benchmark_deep_learning.py
	@echo "$(GREEN)✅ Deep learning benchmarks complete!$(NC)"

benchmark-generative: ## Benchmark generative models
	@echo "$(BLUE)Benchmarking generative models...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/benchmarks/benchmark_generative.py
	@echo "$(GREEN)✅ Generative model benchmarks complete!$(NC)"

##@ Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@cd $(DOCS_DIR) && $(VENV_NAME)/../bin/sphinx-build -b html . _build/html
	@echo "$(GREEN)✅ Documentation built in docs/_build/html/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	@cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-api: ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@$(VENV_NAME)/bin/sphinx-apidoc -o $(DOCS_DIR)/api $(SRC_DIR)
	@echo "$(GREEN)✅ API documentation generated!$(NC)"

##@ Training and Evaluation
train: ## Train a model (specify MODEL_NAME and CONFIG)
	@echo "$(BLUE)Training model: $(or $(MODEL_NAME),default)$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/training/train_model.py \
		--model $(or $(MODEL_NAME),random_forest) \
		--config $(or $(CONFIG),configs/models/default.yaml) \
		$(ARGS)
	@echo "$(GREEN)✅ Training complete!$(NC)"

evaluate: ## Evaluate a model (specify MODEL_NAME)
	@echo "$(BLUE)Evaluating model: $(or $(MODEL_NAME),latest)$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/evaluation/evaluate_model.py \
		--model $(or $(MODEL_NAME),latest) \
		$(ARGS)
	@echo "$(GREEN)✅ Evaluation complete!$(NC)"

hyperparameter-search: ## Run hyperparameter optimization
	@echo "$(BLUE)Running hyperparameter search...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/training/hyperparameter_search.py \
		--model $(or $(MODEL_NAME),random_forest) \
		--trials $(or $(TRIALS),50) \
		$(ARGS)
	@echo "$(GREEN)✅ Hyperparameter search complete!$(NC)"

##@ Deployment
build-docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE)$(NC)"

build-docker-gpu: ## Build GPU-enabled Docker image
	@echo "$(BLUE)Building GPU Docker image...$(NC)"
	@docker build -f Dockerfile.gpu -t $(DOCKER_IMAGE)-gpu .
	@echo "$(GREEN)✅ GPU Docker image built: $(DOCKER_IMAGE)-gpu$(NC)"

run-docker: ## Run Docker container locally
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p 8080:8080 $(DOCKER_IMAGE)

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started!$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(YELLOW)Stopping services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

deploy-local: ## Deploy to local environment
	@echo "$(BLUE)Deploying to local environment...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/deploy_local.py \
		--model $(or $(MODEL_NAME),latest) \
		$(ARGS)
	@echo "$(GREEN)✅ Local deployment complete!$(NC)"

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/deploy_staging.py \
		--model $(or $(MODEL_NAME),latest) \
		$(ARGS)
	@echo "$(GREEN)✅ Staging deployment complete!$(NC)"

deploy-production: ## Deploy to production (requires confirmation)
	@echo "$(RED)⚠️  Production deployment requires confirmation!$(NC)"
	@read -p "Deploy $(or $(MODEL_NAME),latest) to production? [y/N] " confirm && \
		[ "$$confirm" = "y" ] || exit 1
	@echo "$(BLUE)Deploying to production...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/deploy_production.py \
		--model $(or $(MODEL_NAME),latest) \
		$(ARGS)
	@echo "$(GREEN)✅ Production deployment complete!$(NC)"

##@ Cloud Deployment
deploy-aws: ## Deploy to AWS
	@echo "$(BLUE)Deploying to AWS...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/cloud/deploy_aws.py \
		--model $(or $(MODEL_NAME),latest) \
		--region $(or $(AWS_REGION),us-east-1) \
		$(ARGS)
	@echo "$(GREEN)✅ AWS deployment complete!$(NC)"

deploy-gcp: ## Deploy to Google Cloud Platform
	@echo "$(BLUE)Deploying to GCP...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/cloud/deploy_gcp.py \
		--model $(or $(MODEL_NAME),latest) \
		--project $(or $(GCP_PROJECT),ml-arsenal) \
		$(ARGS)
	@echo "$(GREEN)✅ GCP deployment complete!$(NC)"

deploy-azure: ## Deploy to Microsoft Azure
	@echo "$(BLUE)Deploying to Azure...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/deployment/cloud/deploy_azure.py \
		--model $(or $(MODEL_NAME),latest) \
		--resource-group $(or $(AZURE_RG),ml-arsenal) \
		$(ARGS)
	@echo "$(GREEN)✅ Azure deployment complete!$(NC)"

##@ Kubernetes
k8s-deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	@kubectl apply -f deployment/kubernetes/ -n $(KUBERNETES_NAMESPACE)
	@echo "$(GREEN)✅ Kubernetes deployment complete!$(NC)"

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Checking Kubernetes status...$(NC)"
	@kubectl get pods,services,deployments -n $(KUBERNETES_NAMESPACE)

k8s-logs: ## View Kubernetes logs
	@echo "$(BLUE)Viewing Kubernetes logs...$(NC)"
	@kubectl logs -f deployment/ml-arsenal -n $(KUBERNETES_NAMESPACE)

k8s-delete: ## Delete Kubernetes deployment
	@echo "$(YELLOW)Deleting Kubernetes deployment...$(NC)"
	@kubectl delete -f deployment/kubernetes/ -n $(KUBERNETES_NAMESPACE)
	@echo "$(GREEN)✅ Kubernetes deployment deleted!$(NC)"

##@ Data Management
data-download: ## Download sample datasets
	@echo "$(BLUE)Downloading sample datasets...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/data/download_datasets.py $(ARGS)
	@echo "$(GREEN)✅ Datasets downloaded!$(NC)"

data-preprocess: ## Preprocess data
	@echo "$(BLUE)Preprocessing data...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/data/preprocess_data.py \
		--input $(or $(INPUT_DATA),data/raw/) \
		--output $(or $(OUTPUT_DATA),data/processed/) \
		$(ARGS)
	@echo "$(GREEN)✅ Data preprocessing complete!$(NC)"

data-validate: ## Validate data quality
	@echo "$(BLUE)Validating data quality...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/data/validate_data.py \
		--data $(or $(DATA_PATH),data/processed/) \
		$(ARGS)
	@echo "$(GREEN)✅ Data validation complete!$(NC)"

##@ Notebooks
notebooks: ## Start Jupyter Lab server
	@echo "$(BLUE)Starting Jupyter Lab server...$(NC)"
	@cd notebooks && $(VENV_NAME)/../bin/jupyter lab

notebooks-convert: ## Convert notebooks to scripts
	@echo "$(BLUE)Converting notebooks to scripts...$(NC)"
	@find notebooks -name "*.ipynb" -exec $(VENV_NAME)/bin/jupyter nbconvert --to script {} \;
	@echo "$(GREEN)✅ Notebooks converted!$(NC)"

notebooks-clean: ## Clean notebook outputs
	@echo "$(BLUE)Cleaning notebook outputs...$(NC)"
	@find notebooks -name "*.ipynb" -exec $(VENV_NAME)/bin/jupyter nbconvert --clear-output --inplace {} \;
	@echo "$(GREEN)✅ Notebook outputs cleaned!$(NC)"

##@ Monitoring
monitor-setup: ## Setup monitoring infrastructure
	@echo "$(BLUE)Setting up monitoring...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/monitoring/setup_monitoring.py $(ARGS)
	@echo "$(GREEN)✅ Monitoring setup complete!$(NC)"

monitor-health: ## Check model health
	@echo "$(BLUE)Checking model health...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/monitoring/check_model_health.py \
		--model $(or $(MODEL_NAME),production) \
		$(ARGS)
	@echo "$(GREEN)✅ Health check complete!$(NC)"

monitor-drift: ## Check for data drift
	@echo "$(BLUE)Checking for data drift...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/monitoring/drift_detection.py \
		--baseline $(or $(BASELINE_DATA),data/baseline/) \
		--current $(or $(CURRENT_DATA),data/current/) \
		$(ARGS)
	@echo "$(GREEN)✅ Drift detection complete!$(NC)"

##@ Security
security-scan: ## Run security scan
	@echo "$(BLUE)Running security scan...$(NC)"
	@$(VENV_NAME)/bin/bandit -r $(SRC_DIR) -f json -o security_report.json
	@$(VENV_NAME)/bin/safety check --json --output safety_report.json
	@echo "$(GREEN)✅ Security scan complete!$(NC)"

audit: ## Run dependency audit
	@echo "$(BLUE)Running dependency audit...$(NC)"
	@$(PIP) audit
	@echo "$(GREEN)✅ Dependency audit complete!$(NC)"

##@ Migration
migrate: ## Run migration from old structure
	@echo "$(BLUE)Running migration from old structure...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/migration/migrate_structure.py $(ARGS)
	@echo "$(GREEN)✅ Migration complete!$(NC)"

migrate-validate: ## Validate migration results
	@echo "$(BLUE)Validating migration...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/migration/validate_migration.py $(ARGS)
	@echo "$(GREEN)✅ Migration validation complete!$(NC)"

##@ Utilities
env-info: ## Display environment information
	@echo "$(CYAN)Environment Information$(NC)"
	@echo "$(CYAN)======================$(NC)"
	@echo "Python Version: $$($(PYTHON_VENV) --version)"
	@echo "Virtual Environment: $(VENV_NAME)"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Unknown')"
	@echo "Git Commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'Unknown')"
	@echo "Docker Version: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Kubernetes Version: $$(kubectl version --client --short 2>/dev/null || echo 'Not installed')"

version: ## Display project version
	@echo "$(CYAN)ML Arsenal Version: $$($(PYTHON_VENV) -c 'import src; print(src.__version__)' 2>/dev/null || echo 'Unknown')$(NC)"

requirements-update: ## Update requirements.txt
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	@$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✅ Requirements updated!$(NC)"

##@ Development Shortcuts
dev: install-dev setup ## Complete development setup (shortcut)
	@echo "$(GREEN)🚀 Development environment ready!$(NC)"

check: quality test ## Run all quality checks and tests
	@echo "$(GREEN)✅ All checks passed!$(NC)"

build: clean check build-docker ## Clean, check, and build
	@echo "$(GREEN)✅ Build complete!$(NC)"

deploy: build deploy-local ## Build and deploy locally
	@echo "$(GREEN)🚀 Local deployment complete!$(NC)"

ci: clean check coverage benchmark ## Run CI pipeline locally
	@echo "$(GREEN)✅ CI pipeline complete!$(NC)"

##@ Examples and Demos
demo-classification: ## Run classification demo
	@echo "$(BLUE)Running classification demo...$(NC)"
	@$(PYTHON_VENV) examples/classification_demo.py
	@echo "$(GREEN)✅ Classification demo complete!$(NC)"

demo-regression: ## Run regression demo
	@echo "$(BLUE)Running regression demo...$(NC)"
	@$(PYTHON_VENV) examples/regression_demo.py
	@echo "$(GREEN)✅ Regression demo complete!$(NC)"

demo-deep-learning: ## Run deep learning demo
	@echo "$(BLUE)Running deep learning demo...$(NC)"
	@$(PYTHON_VENV) examples/deep_learning_demo.py
	@echo "$(GREEN)✅ Deep learning demo complete!$(NC)"

demo-generative: ## Run generative AI demo
	@echo "$(BLUE)Running generative AI demo...$(NC)"
	@$(PYTHON_VENV) examples/generative_demo.py
	@echo "$(GREEN)✅ Generative AI demo complete!$(NC)"

##@ Advanced Operations
profile: ## Profile application performance
	@echo "$(BLUE)Profiling application performance...$(NC)"
	@$(PYTHON_VENV) -m cProfile -o profile.stats $(SCRIPTS_DIR)/profiling/profile_app.py
	@echo "$(GREEN)✅ Profiling complete! Results in profile.stats$(NC)"

memory-profile: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(NC)"
	@$(VENV_NAME)/bin/mprof run $(SCRIPTS_DIR)/profiling/memory_profile.py
	@$(VENV_NAME)/bin/mprof plot
	@echo "$(GREEN)✅ Memory profiling complete!$(NC)"

stress-test: ## Run stress tests
	@echo "$(BLUE)Running stress tests...$(NC)"
	@$(PYTHON_VENV) $(SCRIPTS_DIR)/testing/stress_test.py \
		--concurrent $(or $(CONCURRENT),10) \
		--duration $(or $(DURATION),60) \
		$(ARGS)
	@echo "$(GREEN)✅ Stress testing complete!$(NC)"

##@ Maintenance
update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install --upgrade -r requirements.txt
	@$(PIP) install --upgrade -r requirements-dev.txt
	@echo "$(GREEN)✅ Dependencies updated!$(NC)"

backup: ## Backup important files
	@echo "$(BLUE)Creating backup...$(NC)"
	@tar -czf backup_$$(date +%Y%m%d_%H%M%S).tar.gz \
		src/ tests/ docs/ configs/ notebooks/ scripts/ \
		requirements*.txt setup.py Makefile README.md
	@echo "$(GREEN)✅ Backup created!$(NC)"

# Help target to show available commands with colors
list-targets: ## List all available targets
	@echo "$(CYAN)Available Makefile targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  $(CYAN)%-25s$(NC) %s\n", $$1, $$2}'
