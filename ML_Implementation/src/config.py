from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/Users/aponduga/Desktop/Personal/ML_DS/Project_Implementation/Emotion_Recognition_System/data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Data files
TWEET_EMOTIONS_FILE = RAW_DATA_DIR / "tweet_emotions.csv"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "model_name": "bert-base-uncased",
    "num_labels": 6,
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3
}

# Training Configuration
TRAIN_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
} 