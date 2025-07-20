"""
Text Generation Example using Simple Transformer
Demonstrates basic text generation capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional


class SimpleGPT(nn.Module):
    """Simplified GPT model for demonstration"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256, num_layers: int = 4, 
                 num_heads: int = 8, max_length: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                logits = self.forward(input_ids)
                
                # Get the last token's logits and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_logits[:, -1:]] = float('-inf')
                
                # Sample the next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we reach max length
                if input_ids.size(1) >= self.max_length:
                    break
        
        return input_ids


class CharTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        chars = set()
        for text in texts:
            chars.update(text.lower())
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        vocab = special_tokens + sorted(list(chars))
        
        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        return [self.char_to_id.get(char.lower(), self.char_to_id['<unk>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text"""
        return ''.join([self.id_to_char.get(id, '<unk>') for id in token_ids])


def create_sample_dataset():
    """Create a simple dataset for training"""
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with multiple layers",
        "natural language processing helps computers understand text",
        "transformers have revolutionized the field of nlp",
        "attention mechanisms allow models to focus on relevant information",
        "language models can generate coherent and contextual text",
        "artificial intelligence is transforming many industries",
        "data science combines statistics programming and domain expertise",
        "python is a popular programming language for machine learning",
    ]
    
    # Expand dataset by repeating texts
    expanded_texts = []
    for text in texts:
        expanded_texts.extend([text] * 20)
    
    return expanded_texts


def train_model():
    """Train a simple language model"""
    print("Creating dataset...")
    texts = create_sample_dataset()
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Prepare training data
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    # Create sequences
    seq_length = 64
    sequences = []
    for i in range(len(all_tokens) - seq_length):
        sequences.append(all_tokens[i:i + seq_length + 1])
    
    print(f"Created {len(sequences)} training sequences")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGPT(vocab_size=tokenizer.vocab_size, d_model=128, num_layers=3, num_heads=4)
    model.to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {device}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    batch_size = 16
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        for i in range(0, len(sequences) - batch_size, batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Prepare batch
            input_ids = torch.tensor([seq[:-1] for seq in batch_sequences], device=device)
            targets = torch.tensor([seq[1:] for seq in batch_sequences], device=device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, tokenizer


def generate_text_sample(model, tokenizer, prompt: str = "the", max_tokens: int = 100):
    """Generate text using the trained model"""
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate
    generated_ids = model.generate(
        input_ids, 
        max_new_tokens=max_tokens, 
        temperature=0.8, 
        top_k=10
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    return generated_text


def main():
    """Main demonstration function"""
    print("Simple Text Generation Example")
    print("=" * 40)
    
    try:
        # Train model
        print("Training model...")
        model, tokenizer = train_model()
        
        # Generate examples
        prompts = ["the", "machine", "deep", "artificial", "python"]
        
        print("\nGenerating text samples:")
        print("-" * 40)
        
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            generated = generate_text_sample(model, tokenizer, prompt, max_tokens=50)
            print(f"Generated: {generated}")
        
        print("\nText generation completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
