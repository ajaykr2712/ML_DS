import sys
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

def test_environment():
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test TensorFlow
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Test basic ML workflow
    print("\nTesting scikit-learn workflow:")
    iris = datasets.load_iris()
    print(f"Dataset shape: {iris.data.shape}")
    
    # Test plotting
    plt.figure(figsize=(6, 4))
    plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    plt.savefig('logs/test_plot.png')
    plt.close()
    
    print("\nSetup test completed successfully!")

if __name__ == "__main__":
    test_environment() 