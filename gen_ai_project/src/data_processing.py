"""
Data processing utilities for generative AI models.
Handles text preprocessing, tokenization, and dataset preparation.
"""

import os
import json
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
try:
    import datasets
except ImportError:
    datasets = None
from tqdm.auto import tqdm


@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_name: str = "custom"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    max_length: int = 1024
    tokenizer_name: str = "gpt2"
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42


class TextPreprocessor:
    """Advanced text preprocessing utilities."""
    
    def __init__(self, 
                 lowercase: bool = False,
                 remove_special_chars: bool = False,
                 min_length: int = 10,
                 max_length: int = 1024):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
        self.max_length = max_length
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special_chars:
            # Keep only alphanumeric characters, spaces, and basic punctuation
            import re
            text = re.sub(r'[^\w\s.,!?;:()\-"\']', ' ', text)
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text
    
    def filter_by_length(self, texts: List[str]) -> List[str]:
        """Filter texts by length criteria."""
        filtered = []
        for text in texts:
            if self.min_length <= len(text) <= self.max_length:
                filtered.append(text)
        
        self.logger.info(f"Filtered {len(texts)} -> {len(filtered)} texts by length")
        return filtered
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts."""
        unique_texts = list(dict.fromkeys(texts))  # Preserves order
        self.logger.info(f"Deduplicated {len(texts)} -> {len(unique_texts)} texts")
        return unique_texts
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        processed = []
        for text in texts:
            cleaned = self.clean_text(text)
            if cleaned:  # Only keep non-empty texts
                processed.append(cleaned)
        return processed
    
    def process_parallel(self, texts: List[str], n_workers: int = None) -> List[str]:
        """Process texts in parallel."""
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)
        
        # Split texts into chunks
        chunk_size = max(1, len(texts) // n_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        processed_texts = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self.process_batch, chunk): chunk for chunk in chunks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing texts"):
                processed_texts.extend(future.result())
        
        return processed_texts


class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024,
                 return_tensors: str = "pt"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=self.return_tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': text
        }


class TextDataLoader:
    """Advanced data loading utilities for text data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and cache tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                cache_dir=self.config.cache_dir
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.tokenizer
    
    def load_text_files(self, data_path: str) -> List[str]:
        """Load texts from files in a directory."""
        texts = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Single file
            if data_path.suffix == '.txt':
                with open(data_path, 'r', encoding='utf-8') as f:
                    texts.extend(f.read().splitlines())
            elif data_path.suffix == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts.extend([str(item) for item in data])
                    elif isinstance(data, dict) and 'texts' in data:
                        texts.extend(data['texts'])
            elif data_path.suffix == '.jsonl':
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if isinstance(data, dict) and 'text' in data:
                            texts.append(data['text'])
                        else:
                            texts.append(str(data))
        
        elif data_path.is_dir():
            # Directory of files
            for file_path in data_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.txt', '.json', '.jsonl']:
                    texts.extend(self.load_text_files(file_path))
        
        self.logger.info(f"Loaded {len(texts)} texts from {data_path}")
        return texts
    
    def load_huggingface_dataset(self, dataset_name: str, split: str = "train") -> List[str]:
        """Load dataset from Hugging Face."""
        if datasets is None:
            self.logger.error("datasets library not available. Install with: pip install datasets")
            return []
        
        try:
            dataset = datasets.load_dataset(
                dataset_name, 
                split=split,
                cache_dir=self.config.cache_dir
            )
            
            # Extract text field (common field names)
            text_fields = ['text', 'content', 'body', 'article', 'document']
            text_field = None
            
            for field in text_fields:
                if field in dataset.column_names:
                    text_field = field
                    break
            
            if text_field is None:
                raise ValueError(f"No text field found in dataset. Available fields: {dataset.column_names}")
            
            texts = [item[text_field] for item in dataset]
            self.logger.info(f"Loaded {len(texts)} texts from HuggingFace dataset: {dataset_name}")
            return texts
        
        except Exception as e:
            self.logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            return []
    
    def split_data(self, texts: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split data into train/val/test sets."""
        np.random.seed(self.config.seed)
        
        # Shuffle texts
        indices = np.random.permutation(len(texts))
        shuffled_texts = [texts[i] for i in indices]
        
        # Calculate split points
        n_train = int(len(texts) * self.config.train_split)
        n_val = int(len(texts) * self.config.val_split)
        
        # Split data
        train_texts = shuffled_texts[:n_train]
        val_texts = shuffled_texts[n_train:n_train + n_val]
        test_texts = shuffled_texts[n_train + n_val:]
        
        self.logger.info(f"Split data: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
        return train_texts, val_texts, test_texts
    
    def create_dataloader(self, 
                         texts: List[str], 
                         shuffle: bool = True,
                         drop_last: bool = False) -> DataLoader:
        """Create PyTorch DataLoader from texts."""
        tokenizer = self.load_tokenizer()
        
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last
        )
        
        return dataloader
    
    def prepare_datasets(self, data_source: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train/val/test dataloaders."""
        # Load texts
        if data_source.startswith("hf:"):
            # HuggingFace dataset
            dataset_name = data_source[3:]
            texts = self.load_huggingface_dataset(dataset_name)
        else:
            # Local files
            texts = self.load_text_files(data_source)
        
        if not texts:
            raise ValueError(f"No texts loaded from {data_source}")
        
        # Preprocess texts
        preprocessor = TextPreprocessor(
            min_length=10,
            max_length=self.config.max_length
        )
        
        texts = preprocessor.process_parallel(texts)
        texts = preprocessor.filter_by_length(texts)
        texts = preprocessor.deduplicate(texts)
        
        # Split data
        train_texts, val_texts, test_texts = self.split_data(texts)
        
        # Create dataloaders
        train_loader = self.create_dataloader(train_texts, shuffle=True, drop_last=True)
        val_loader = self.create_dataloader(val_texts, shuffle=False, drop_last=False)
        test_loader = self.create_dataloader(test_texts, shuffle=False, drop_last=False)
        
        return train_loader, val_loader, test_loader
    
    def save_processed_data(self, 
                           texts: List[str], 
                           filename: str,
                           format: str = "jsonl") -> None:
        """Save processed texts to file."""
        output_path = Path(self.config.data_dir) / filename
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    json.dump({"text": text}, f, ensure_ascii=False)
                    f.write('\n')
        
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
        
        self.logger.info(f"Saved {len(texts)} texts to {output_path}")


class DatasetStatistics:
    """Compute and analyze dataset statistics."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def compute_text_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Compute comprehensive text statistics."""
        if not texts:
            return {}
        
        # Length statistics
        lengths = [len(text) for text in texts]
        token_lengths = []
        
        for text in tqdm(texts[:1000], desc="Computing token lengths"):  # Sample for efficiency
            tokens = self.tokenizer.tokenize(text)
            token_lengths.append(len(tokens))
        
        stats = {
            'num_texts': len(texts),
            'char_length': {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'percentiles': {
                    '25': np.percentile(lengths, 25),
                    '75': np.percentile(lengths, 75),
                    '90': np.percentile(lengths, 90),
                    '95': np.percentile(lengths, 95),
                    '99': np.percentile(lengths, 99)
                }
            },
            'token_length': {
                'mean': np.mean(token_lengths),
                'median': np.median(token_lengths),
                'std': np.std(token_lengths),
                'min': np.min(token_lengths),
                'max': np.max(token_lengths),
                'percentiles': {
                    '25': np.percentile(token_lengths, 25),
                    '75': np.percentile(token_lengths, 75),
                    '90': np.percentile(token_lengths, 90),
                    '95': np.percentile(token_lengths, 95),
                    '99': np.percentile(token_lengths, 99)
                }
            }
        }
        
        return stats
    
    def analyze_vocabulary(self, texts: List[str], top_k: int = 100) -> Dict[str, Any]:
        """Analyze vocabulary statistics."""
        # Tokenize all texts and count tokens
        token_counts = {}
        
        for text in tqdm(texts, desc="Analyzing vocabulary"):
            tokens = self.tokenizer.tokenize(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        vocab_stats = {
            'vocab_size': len(token_counts),
            'total_tokens': sum(token_counts.values()),
            'unique_tokens': len(token_counts),
            'top_tokens': sorted_tokens[:top_k],
            'singleton_count': sum(1 for count in token_counts.values() if count == 1)
        }
        
        return vocab_stats
    
    def print_stats(self, stats: Dict[str, Any]) -> None:
        """Print formatted statistics."""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"\nNumber of texts: {stats['num_texts']:,}")
        
        print("\nCharacter Length Statistics:")
        char_stats = stats['char_length']
        print(f"  Mean: {char_stats['mean']:.1f}")
        print(f"  Median: {char_stats['median']:.1f}")
        print(f"  Std: {char_stats['std']:.1f}")
        print(f"  Range: {char_stats['min']} - {char_stats['max']}")
        print(f"  95th percentile: {char_stats['percentiles']['95']:.1f}")
        
        print("\nToken Length Statistics:")
        token_stats = stats['token_length']
        print(f"  Mean: {token_stats['mean']:.1f}")
        print(f"  Median: {token_stats['median']:.1f}")
        print(f"  Std: {token_stats['std']:.1f}")
        print(f"  Range: {token_stats['min']} - {token_stats['max']}")
        print(f"  95th percentile: {token_stats['percentiles']['95']:.1f}")


def main():
    """Example usage of data processing utilities."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = DataConfig(
        data_dir="./data",
        cache_dir="./cache",
        tokenizer_name="gpt2",
        max_length=512,
        batch_size=16
    )
    
    # Initialize data loader
    data_loader = TextDataLoader(config)
    
    # Example: Load and process data
    try:
        # Try to load from local files first
        texts = data_loader.load_text_files(config.data_dir)
        
        if not texts:
            print("No local data found. You can:")
            print("1. Add .txt, .json, or .jsonl files to the data directory")
            print("2. Use HuggingFace datasets (modify data_source below)")
            
            # Example: Load from HuggingFace
            # texts = data_loader.load_huggingface_dataset("wikitext", "wikitext-2-raw-v1")
            
        if texts:
            print(f"Loaded {len(texts)} texts")
            
            # Prepare datasets
            train_loader, val_loader, test_loader = data_loader.prepare_datasets(config.data_dir)
            
            print("Created dataloaders:")
            print(f"  Train: {len(train_loader)} batches")
            print(f"  Val: {len(val_loader)} batches")
            print(f"  Test: {len(test_loader)} batches")
            
            # Compute statistics
            tokenizer = data_loader.load_tokenizer()
            stats_analyzer = DatasetStatistics(tokenizer)
            
            # Use a sample for statistics
            sample_texts = texts[:1000] if len(texts) > 1000 else texts
            stats = stats_analyzer.compute_text_stats(sample_texts)
            stats_analyzer.print_stats(stats)
    
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()
