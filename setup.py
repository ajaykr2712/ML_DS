#!/usr/bin/env python3
"""
Setup script for ML/DS and Gen AI projects.
Installs dependencies, sets up environment, and validates installation.
"""

import sys
import subprocess
import platform
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(message: str, color: str = Colors.OKGREEN):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")


def run_command(command: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run shell command with error handling."""
    print_colored(f"Running: {command}", Colors.OKCYAN)
    
    try:
        result = subprocess.run(
            command.split(),
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running command: {e}", Colors.FAIL)
        if not check:
            return e
        raise


def check_python_version():
    """Check if Python version is compatible."""
    print_colored("Checking Python version...", Colors.HEADER)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"Python {version.major}.{version.minor} is not supported. Please use Python 3.8+", Colors.FAIL)
        sys.exit(1)
    
    print_colored(f"Python {version.major}.{version.minor}.{version.micro} - OK", Colors.OKGREEN)


def check_system_requirements():
    """Check system requirements."""
    print_colored("Checking system requirements...", Colors.HEADER)
    
    # Check OS
    os_name = platform.system()
    print_colored(f"Operating System: {os_name}", Colors.OKBLUE)
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print_colored(f"Available RAM: {memory_gb:.1f} GB", Colors.OKBLUE)
        
        if memory_gb < 4:
            print_colored("Warning: Less than 4GB RAM available. Some models may not run.", Colors.WARNING)
    except ImportError:
        print_colored("Could not check memory (psutil not installed)", Colors.WARNING)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_colored(f"GPU Available: {gpu_name} (x{gpu_count})", Colors.OKGREEN)
        else:
            print_colored("No GPU available - will use CPU", Colors.WARNING)
    except ImportError:
        print_colored("PyTorch not installed - cannot check GPU", Colors.WARNING)


def install_requirements(requirements_file: str):
    """Install Python requirements."""
    print_colored(f"Installing requirements from {requirements_file}...", Colors.HEADER)
    
    if not Path(requirements_file).exists():
        print_colored(f"Requirements file {requirements_file} not found", Colors.FAIL)
        return False
    
    try:
        run_command(f"pip install -r {requirements_file}")
        print_colored("Requirements installed successfully", Colors.OKGREEN)
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to install requirements", Colors.FAIL)
        return False


def setup_git_hooks():
    """Setup git hooks for code quality."""
    print_colored("Setting up git hooks...", Colors.HEADER)
    
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        print_colored("Not a git repository - skipping git hooks", Colors.WARNING)
        return
    
    # Pre-commit hook for code formatting
    pre_commit_hook = hooks_dir / "pre-commit"
    pre_commit_content = """#!/bin/bash
# Run black formatter
black --check .
if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Run 'black .' to fix."
    exit 1
fi

# Run flake8 linter
flake8 .
if [ $? -ne 0 ]; then
    echo "Linting issues found. Please fix before committing."
    exit 1
fi
"""
    
    try:
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        print_colored("Git hooks installed successfully", Colors.OKGREEN)
    except Exception as e:
        print_colored(f"Failed to install git hooks: {e}", Colors.WARNING)


def validate_installation():
    """Validate that key packages are installed correctly."""
    print_colored("Validating installation...", Colors.HEADER)
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib',
        'seaborn', 'jupyter', 'tqdm', 'pyyaml'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"✓ {package}", Colors.OKGREEN)
        except ImportError:
            print_colored(f"✗ {package}", Colors.FAIL)
            failed_packages.append(package)
    
    if failed_packages:
        print_colored(f"Failed to import: {', '.join(failed_packages)}", Colors.FAIL)
        return False
    
    print_colored("All packages imported successfully", Colors.OKGREEN)
    return True


def create_project_structure():
    """Create necessary project directories."""
    print_colored("Creating project structure...", Colors.HEADER)
    
    directories = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "outputs/figures",
        "outputs/reports",
        "logs",
        "configs",
        "scripts",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_colored(f"Created: {directory}", Colors.OKBLUE)
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# ML/DS specific
*.pkl
*.h5
*.hdf5
*.joblib
data/raw/*
!data/raw/.gitkeep
models/checkpoints/*
!models/checkpoints/.gitkeep
logs/*
!logs/.gitkeep
outputs/*
!outputs/.gitkeep

# VS Code
.vscode/

# MacOS
.DS_Store

# Large files
*.tar.gz
*.zip
*.7z
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print_colored("Created .gitignore", Colors.OKGREEN)


def setup_jupyter_extensions():
    """Setup useful Jupyter extensions."""
    print_colored("Setting up Jupyter extensions...", Colors.HEADER)
    
    try:
        # Install and enable useful extensions
        extensions = [
            "jupyter_contrib_nbextensions",
            "jupyterlab-git",
            "nbextensions"
        ]
        
        for ext in extensions:
            try:
                run_command(f"pip install {ext}", check=False)
            except Exception:
                print_colored(f"Could not install {ext}", Colors.WARNING)
        
        # Enable nbextensions
        try:
            run_command("jupyter contrib nbextension install --user", check=False)
            run_command("jupyter nbextension enable --py widgetsnbextension", check=False)
        except Exception:
            print_colored("Could not enable nbextensions", Colors.WARNING)
        
        print_colored("Jupyter extensions setup completed", Colors.OKGREEN)
        
    except Exception as e:
        print_colored(f"Error setting up Jupyter: {e}", Colors.WARNING)


def create_example_config():
    """Create example configuration files."""
    print_colored("Creating example configuration files...", Colors.HEADER)
    
    # Example training config
    training_config = """# Example Training Configuration
model:
  type: "transformer"
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  dropout: 0.1
  vocab_size: 50257

training:
  batch_size: 32
  learning_rate: 5e-5
  epochs: 10
  gradient_clip_norm: 1.0
  mixed_precision: true
  
data:
  max_length: 512
  train_path: "data/processed/train.txt"
  val_path: "data/processed/val.txt"

logging:
  log_dir: "logs"
  save_every: 1000
  eval_every: 500
"""
    
    config_path = Path("configs/example_training.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(training_config)
    
    print_colored(f"Created example config: {config_path}", Colors.OKGREEN)


def main():
    """Main setup function."""
    print_colored("🚀 ML/DS and Gen AI Project Setup", Colors.HEADER)
    print_colored("=" * 50, Colors.HEADER)
    
    # Check prerequisites
    check_python_version()
    check_system_requirements()
    
    # Setup project
    create_project_structure()
    
    # Install requirements
    requirements_files = [
        "requirements.txt",
        "gen_ai_project/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            if not install_requirements(req_file):
                print_colored(f"Failed to install {req_file}", Colors.WARNING)
    
    # Validate installation
    if not validate_installation():
        print_colored("Installation validation failed", Colors.FAIL)
        sys.exit(1)
    
    # Optional setup steps
    setup_git_hooks()
    setup_jupyter_extensions()
    create_example_config()
    
    # Final message
    print_colored("\n" + "=" * 50, Colors.HEADER)
    print_colored("🎉 Setup completed successfully!", Colors.OKGREEN)
    print_colored("\nNext steps:", Colors.HEADER)
    print_colored("1. Activate your virtual environment if not already active", Colors.OKBLUE)
    print_colored("2. Run 'jupyter lab' to start Jupyter", Colors.OKBLUE)
    print_colored("3. Check the examples in the gen_ai_project/examples/ directory", Colors.OKBLUE)
    print_colored("4. Review the documentation in README.md", Colors.OKBLUE)
    print_colored("\nHappy coding! 🚀", Colors.OKGREEN)


if __name__ == "__main__":
    main()
