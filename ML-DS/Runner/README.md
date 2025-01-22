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