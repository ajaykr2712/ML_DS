#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python test_setup.py

# Create .env file
cat > .env << 'EOF'
# ML Environment Variables
PYTHONPATH=${PYTHONPATH}:${PWD}
CUDA_VISIBLE_DEVICES=0
WANDB_API_KEY=your_wandb_key_here
MLFLOW_TRACKING_URI=http://localhost:5000

# Training Parameters
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=100
MODEL_SAVE_PATH=./models/checkpoints

# Data Paths
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
MODEL_ARTIFACTS_PATH=./models/final

# Logging
LOG_LEVEL=INFO
LOG_PATH=./logs
EOF

# Make scripts executable
chmod +x setup-runner.sh monitor.sh

echo "Virtual environment setup complete!" 