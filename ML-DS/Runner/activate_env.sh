#!/bin/bash

# Source the virtual environment
source venv/bin/activate

# Set environment variables from .env
set -a
source .env
set +a

# Print environment info
echo "ML Environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo "Working directory: $PWD"