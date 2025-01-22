import os
import sys
import torch
import tensorflow as tf

def check_environment():
    """Check and print environment information."""
    print("\n=== Environment Information ===\n")
    
    # Python Info
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # CUDA Info
    print(f"\nCUDA Available (PyTorch): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # TensorFlow GPU Info
    print(f"\nTensorFlow GPU Devices: {tf.config.list_physical_devices('GPU')}")
    
    # Environment Variables
    print("\nEnvironment Variables:")
    env_vars = [
        'PYTHONPATH',
        'CUDA_VISIBLE_DEVICES',
        'WANDB_API_KEY',
        'MLFLOW_TRACKING_URI',
        'BATCH_SIZE',
        'LEARNING_RATE',
        'NUM_EPOCHS',
        'MODEL_SAVE_PATH'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                value = '*' * 8  # Mask API keys
            print(f"{var}: {value}")
    
    print("\n=== Environment Check Complete ===")

if __name__ == "__main__":
    check_environment() 