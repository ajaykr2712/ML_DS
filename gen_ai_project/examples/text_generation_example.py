"""
Text generation example using GPT-style transformer.
"""

import torch
from transformers import GPT2Tokenizer
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import GPTModel


def setup_tokenizer():
    """Setup GPT-2 tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main example function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = setup_tokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example 1: Generate from scratch (untrained model)
    print("\n" + "="*50)
    print("EXAMPLE 1: Untrained Model Generation")
    print("="*50)
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, in a distant galaxy",
        "The key to success in machine learning is",
        "Climate change is one of the most pressing issues"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print(f"Generated: {generated}")
    
    # Example 2: Load pre-trained checkpoint (if available)
    print("\n" + "="*50)
    print("EXAMPLE 2: Loading Checkpoint")
    print("="*50)
    
    checkpoint_path = "../checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully!")
            
            # Generate with trained model
            for prompt in prompts[:2]:
                print(f"\nPrompt: {prompt}")
                generated = generate_text(model, tokenizer, prompt, max_new_tokens=100)
                print(f"Generated: {generated}")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found. To train a model, run the training script first.")
    
    # Example 3: Different generation parameters
    print("\n" + "="*50)
    print("EXAMPLE 3: Different Generation Parameters")
    print("="*50)
    
    prompt = "The development of renewable energy technologies"
    
    # Conservative generation (low temperature)
    print("\nConservative generation (temperature=0.3):")
    conservative = generate_text(model, tokenizer, prompt, temperature=0.3)
    print(conservative)
    
    # Creative generation (high temperature)
    print("\nCreative generation (temperature=1.2):")
    creative = generate_text(model, tokenizer, prompt, temperature=1.2)
    print(creative)
    
    # Greedy generation (no sampling)
    print("\nGreedy generation (no sampling):")
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        greedy_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False
        )
    greedy = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)
    print(greedy)


if __name__ == "__main__":
    main()
