import logging
import torch
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL_CONFIG, MODEL_DIR, TRAIN_CONFIG, TWEET_EMOTIONS_FILE
from preprocessing.text_processor import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and preprocess the tweet emotions dataset."""
    logger.info(f"Loading data from: {TWEET_EMOTIONS_FILE}")
    if not TWEET_EMOTIONS_FILE.exists():
        raise FileNotFoundError(f"Data file not found at: {TWEET_EMOTIONS_FILE}")
        
    df = pd.read_csv(TWEET_EMOTIONS_FILE)
    logger.info(f"Loaded {len(df)} records from dataset")
    return df

def prepare_data(texts, labels, tokenizer):
    """Prepare data for BERT model."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MODEL_CONFIG["max_length"],
        return_tensors="pt"
    )
    
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels)
    )

def train():
    """Main training function."""
    logger.info("Starting training process...")
    
    # Load data
    df = load_data()
    
    # Initialize preprocessor and tokenizer
    preprocessor = TextPreprocessor()
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    
    # Preprocess texts
    texts = preprocessor.prepare_batch(df["text"].tolist())
    labels = df["emotion"].astype("category").cat.codes.tolist()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=1-TRAIN_CONFIG["train_split"],
        random_state=TRAIN_CONFIG["random_seed"]
    )
    
    # Prepare datasets
    train_dataset = prepare_data(train_texts, train_labels, tokenizer)
    val_dataset = prepare_data(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(  # noqa: F841
        val_dataset,
        batch_size=MODEL_CONFIG["batch_size"]
    )
    
    # Initialize model and optimizer
    model = BertForSequenceClassification.from_pretrained(
        MODEL_CONFIG["model_name"],
        num_labels=MODEL_CONFIG["num_labels"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"])
    
    # Training loop
    for epoch in range(MODEL_CONFIG["num_epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        logger.info(f"Epoch {epoch+1}/{MODEL_CONFIG['num_epochs']} completed")
    
    # Save the model
    model.save_pretrained(MODEL_DIR / "final_model")
    logger.info("Training completed successfully")

if __name__ == "__main__":
    train() 