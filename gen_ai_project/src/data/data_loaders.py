"""
Advanced data loading utilities for generative AI models.
Provides efficient, scalable data loading with preprocessing and augmentation.
"""

import warnings
from typing import Dict, List, Optional, Callable, Any

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("datasets not available. Install with: pip install datasets")

try:
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")


class TextDataset(Dataset):
    """Dataset for text generation tasks."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: Optional[object] = None,
        max_length: int = 512,
        stride: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        self.return_tensors = return_tensors
        
        # Preprocess texts
        self.examples = self._preprocess_texts()
    
    def _preprocess_texts(self) -> List[Dict[str, Any]]:
        """Preprocess texts into model inputs."""
        examples = []
        
        for text in tqdm(self.texts, desc="Preprocessing texts"):
            if self.tokenizer:
                # Tokenize text
                if TRANSFORMERS_AVAILABLE and hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                else:
                    # Fallback for custom tokenizers
                    tokens = text.split()  # Simple word-level tokenization
                
                # Split into chunks if text is too long
                if len(tokens) > self.max_length:
                    for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                        chunk = tokens[i:i + self.max_length]
                        examples.append({
                            'input_ids': chunk,
                            'labels': chunk.copy()
                        })
                else:
                    examples.append({
                        'input_ids': tokens,
                        'labels': tokens.copy()
                    })
            else:
                # No tokenizer - just split by characters or words
                if len(text) > self.max_length:
                    for i in range(0, len(text) - self.max_length + 1, self.stride):
                        chunk = text[i:i + self.max_length]
                        examples.append({
                            'input_ids': list(chunk),
                            'labels': list(chunk)
                        })
                else:
                    examples.append({
                        'input_ids': list(text),
                        'labels': list(text)
                    })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        if self.return_tensors == "pt":
            return {
                'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
                'labels': torch.tensor(example['labels'], dtype=torch.long)
            }
        else:
            return example


class ImageTextDataset(Dataset):
    """Dataset for image-text tasks (image captioning, VQA, etc.)."""
    
    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        tokenizer: Optional[object] = None,
        image_transform: Optional[Callable] = None,
        max_length: int = 512
    ):
        assert len(image_paths) == len(texts), "Number of images and texts must match"
        
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        try:
            from PIL import Image
            image = Image.open(self.image_paths[idx]).convert('RGB')
            
            if self.image_transform:
                image = self.image_transform(image)
            else:
                # Default transform: resize and normalize
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
        except Exception as e:
            warnings.warn(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return dummy image
            image = torch.zeros(3, 224, 224)
        
        # Process text
        text = self.texts[idx]
        if self.tokenizer and TRANSFORMERS_AVAILABLE:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            text_tokens = encoding['input_ids'].squeeze(0)
        else:
            # Simple tokenization
            text_tokens = torch.tensor([ord(c) for c in text[:self.max_length]])
        
        return {
            'image': image,
            'input_ids': text_tokens,
            'labels': text_tokens.clone()
        }


class SequenceDataset(Dataset):
    """Generic sequence dataset for various tasks."""
    
    def __init__(
        self,
        sequences: List[List[int]],
        labels: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        pad_token_id: int = 0
    ):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Pad or truncate sequence
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [self.pad_token_id] * (self.max_length - len(sequence))
        
        result = {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if token != self.pad_token_id else 0 for token in sequence],
                dtype=torch.long
            )
        }
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return result


class DataCollator:
    """Custom data collator for batch processing."""
    
    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        padding: str = "longest",  # "longest", "max_length", "do_not_pad"
        return_attention_mask: bool = True
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding = padding
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        # Get all keys from the first example
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ['input_ids', 'labels', 'token_type_ids']:
                # Pad sequence data
                sequences = [item[key] for item in batch]
                
                if self.padding == "longest":
                    # Pad to longest sequence in batch
                    padded = pad_sequence(
                        sequences,
                        batch_first=True,
                        padding_value=self.pad_token_id
                    )
                elif self.padding == "max_length" and self.max_length:
                    # Pad to fixed max length
                    max_len = self.max_length
                    padded = torch.full(
                        (len(sequences), max_len),
                        self.pad_token_id,
                        dtype=sequences[0].dtype
                    )
                    for i, seq in enumerate(sequences):
                        seq_len = min(len(seq), max_len)
                        padded[i, :seq_len] = seq[:seq_len]
                else:
                    # Stack without padding
                    padded = torch.stack(sequences, dim=0)
                
                collated[key] = padded
                
                # Create attention mask if requested
                if key == 'input_ids' and self.return_attention_mask:
                    attention_mask = (padded != self.pad_token_id).long()
                    collated['attention_mask'] = attention_mask
            
            elif key == 'attention_mask':
                # Handle pre-computed attention masks
                masks = [item[key] for item in batch]
                if self.padding == "longest":
                    padded_masks = pad_sequence(
                        masks,
                        batch_first=True,
                        padding_value=0
                    )
                elif self.padding == "max_length" and self.max_length:
                    max_len = self.max_length
                    padded_masks = torch.zeros(
                        (len(masks), max_len),
                        dtype=masks[0].dtype
                    )
                    for i, mask in enumerate(masks):
                        mask_len = min(len(mask), max_len)
                        padded_masks[i, :mask_len] = mask[:mask_len]
                else:
                    padded_masks = torch.stack(masks, dim=0)
                
                collated[key] = padded_masks
            
            else:
                # Stack other tensor data (images, etc.)
                tensors = [item[key] for item in batch]
                collated[key] = torch.stack(tensors, dim=0)
        
        return collated


class DataLoaderFactory:
    """Factory for creating optimized data loaders."""
    
    @staticmethod
    def create_text_dataloader(
        texts: List[str],
        tokenizer: Optional[object] = None,
        batch_size: int = 32,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create data loader for text generation."""
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        collator = DataCollator(
            pad_token_id=tokenizer.pad_token_id if tokenizer and hasattr(tokenizer, 'pad_token_id') else 0,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    @staticmethod
    def create_image_text_dataloader(
        image_paths: List[str],
        texts: List[str],
        tokenizer: Optional[object] = None,
        image_transform: Optional[Callable] = None,
        batch_size: int = 32,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create data loader for image-text tasks."""
        dataset = ImageTextDataset(
            image_paths=image_paths,
            texts=texts,
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_length=max_length
        )
        
        collator = DataCollator(
            pad_token_id=tokenizer.pad_token_id if tokenizer and hasattr(tokenizer, 'pad_token_id') else 0,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    @staticmethod
    def create_sequence_dataloader(
        sequences: List[List[int]],
        labels: Optional[List[int]] = None,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        pad_token_id: int = 0,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create data loader for sequence tasks."""
        dataset = SequenceDataset(
            sequences=sequences,
            labels=labels,
            max_length=max_length,
            pad_token_id=pad_token_id
        )
        
        collator = DataCollator(
            pad_token_id=pad_token_id,
            max_length=max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )


def load_text_from_files(
    file_paths: List[str],
    encoding: str = 'utf-8',
    min_length: int = 10,
    max_files: Optional[int] = None
) -> List[str]:
    """Load and preprocess text from multiple files."""
    texts = []
    
    for i, file_path in enumerate(tqdm(file_paths, desc="Loading text files")):
        if max_files and i >= max_files:
            break
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                
                # Filter by minimum length
                if len(content) >= min_length:
                    texts.append(content)
        
        except Exception as e:
            warnings.warn(f"Error reading {file_path}: {e}")
    
    return texts


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = False
) -> List[str]:
    """Load dataset from Hugging Face datasets."""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets")
    
    # Load dataset
    dataset = datasets.load_dataset(dataset_name, split=split, streaming=streaming)
    
    texts = []
    count = 0
    
    for example in tqdm(dataset, desc=f"Loading {dataset_name}"):
        if max_samples and count >= max_samples:
            break
        
        text = example.get(text_column, "")
        if text:
            texts.append(text)
            count += 1
    
    return texts


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create data loader for distributed training."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Don't shuffle when using DistributedSampler
        **kwargs
    )


# Utility functions for data preprocessing
def clean_text(text: str) -> str:
    """Basic text cleaning."""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    return text.strip()


def split_text_by_sentences(text: str, max_length: int = 512) -> List[str]:
    """Split text into chunks by sentences."""
    import re
    
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


if __name__ == "__main__":
    # Example usage
    
    # Create sample text data
    sample_texts = [
        "This is a sample text for training a language model.",
        "Another example of text that could be used for training.",
        "Machine learning is fascinating and powerful."
    ]
    
    # Create text data loader
    text_loader = DataLoaderFactory.create_text_dataloader(
        texts=sample_texts,
        batch_size=2,
        max_length=50,
        shuffle=True
    )
    
    # Test the data loader
    for batch in text_loader:
        print("Batch keys:", list(batch.keys()))
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        break
    
    print("Data loading test completed successfully!")
