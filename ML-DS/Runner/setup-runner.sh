#!/bin/bash

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

# Create and activate virtual environment
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install monitoring tools
sudo apt-get install -y \
    htop \
    nvidia-smi \
    iotop \
    nmon

# Create data directories
mkdir -p data/{raw,processed,interim}
mkdir -p models/{checkpoints,final}
mkdir -p logs 