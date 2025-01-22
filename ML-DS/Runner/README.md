# ML/DS Runner Setup

This directory contains the setup for a self-hosted GitHub Actions runner optimized for Machine Learning and Data Science workloads.

## Requirements

- Ubuntu 20.04 or later
- NVIDIA GPU with CUDA support
- Minimum 32GB RAM
- 500GB Storage

## Setup Instructions

1. Install system dependencies:
```bash
./setup-runner.sh
```

2. Configure GitHub Runner:
- Go to GitHub repository → Settings → Actions → Runners
- Click "New self-hosted runner"
- Follow the installation instructions

3. Test the setup:
```bash
source venv/bin/activate
python test_setup.py
```

4. Start monitoring:
```bash
./monitor.sh &
```

## Directory Structure

```
Runner/
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── models/
│   ├── checkpoints/
│   └── final/
├── logs/
├── requirements.txt
├── setup-runner.sh
├── monitor.sh
├── runner-config.yml
├── test_setup.py
└── README.md
```

## Monitoring

System metrics are logged to `logs/system.log` every minute, including:
- CPU usage
- Memory usage
- GPU usage
- Disk usage 

## Environment Setup

1. Create and activate virtual environment:
```bash
# Create and setup virtual environment
./setup_venv.sh

# Activate environment
source activate_env.sh
```

2. Verify installation:
```bash
python test_setup.py
```

3. Environment Variables:
The following environment variables are set in `.env`:
- `PYTHONPATH`: Adds project root to Python path
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `WANDB_API_KEY`: Weights & Biases API key
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL
- `BATCH_SIZE`: Default batch size for training
- `LEARNING_RATE`: Default learning rate
- `NUM_EPOCHS`: Default number of training epochs
- `MODEL_SAVE_PATH`: Path for model checkpoints 