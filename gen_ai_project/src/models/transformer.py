"""
Advanced Transformer model implementations for generative AI.
Includes GPT-style models with modern improvements and optimizations.
"""

from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Embedding, Dropout
from torch.utils.checkpoint import checkpoint


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    Provides better length extrapolation than learned positional embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for position encodings
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            
            self._cached_cos = freqs.cos()
            self._cached_sin = freqs.sin()
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (..., seq_len, dim)
            seq_len: Sequence length (if None, inferred from x)
        
        Returns:
            cos, sin: Cosine and sine components for RoPE
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        self._update_cache(seq_len, x.device, x.dtype)
        
        cos = self._cached_cos[:seq_len]
        sin = self._cached_sin[:seq_len]
        
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to queries and keys."""
    
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional improvements:
    - Flash Attention compatibility
    - Rotary positional embeddings
    - Memory-efficient implementation
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 use_rope: bool = True,
                 use_flash: bool = True,
                 max_seq_len: int = 8192):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_flash = use_flash
        
        # Linear projections
        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.o_proj = Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # Rotary positional embedding
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Check for Flash Attention availability
        self.flash_available = hasattr(F, 'scaled_dot_product_attention') and use_flash
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_value_states: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            key_value_states: Key-value states for cross-attention
            use_cache: Whether to return key-value cache
            past_key_value: Previous key-value cache
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            present_key_value: Current key-value cache (if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Self-attention or cross-attention
        is_cross_attn = key_value_states is not None
        kv_states = key_value_states if is_cross_attn else hidden_states
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(kv_states)
        value = self.v_proj(kv_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        # Apply rotary positional embedding
        if self.use_rope and not is_cross_attn:
            cos, sin = self.rope(query, seq_len=key.shape[-2])
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Prepare cache for next iteration
        present_key_value = (key, value) if use_cache else None
        
        # Compute attention
        if self.flash_available and attention_mask is None:
            # Use Flash Attention if available and no custom mask
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=not is_cross_attn
            )
        else:
            # Standard attention computation
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            
            # Apply attention mask
            if attention_mask is not None:
                # Expand mask for multi-head
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            # Apply causal mask for self-attention
            if not is_cross_attn:
                causal_mask = torch.triu(
                    torch.ones(seq_len, key.shape[-2], device=query.device),
                    diagonal=1
                ).bool()
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output, present_key_value


class FeedForward(nn.Module):
    """
    Feed-forward network with optional improvements:
    - SwiGLU activation (used in LLaMA)
    - Configurable activation functions
    """
    
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 bias: bool = True,
                 use_swiglu: bool = False):
        super().__init__()
        
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU: Split the hidden dimension and use SiLU activation
            self.gate_proj = Linear(d_model, d_ff, bias=bias)
            self.up_proj = Linear(d_model, d_ff, bias=bias)
            self.down_proj = Linear(d_ff, d_model, bias=bias)
            self.act_fn = nn.SiLU()
        else:
            # Standard FFN
            self.fc1 = Linear(d_model, d_ff, bias=bias)
            self.fc2 = Linear(d_ff, d_model, bias=bias)
            
            # Activation function
            if activation == "gelu":
                self.act_fn = nn.GELU()
            elif activation == "relu":
                self.act_fn = nn.ReLU()
            elif activation == "swish" or activation == "silu":
                self.act_fn = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: gate * activation(up_projection)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = self.act_fn(gate) * up
            output = self.down_proj(hidden)
        else:
            # Standard FFN
            hidden = self.act_fn(self.fc1(x))
            hidden = self.dropout(hidden)
            output = self.fc2(hidden)
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm and used in modern models like LLaMA.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with modern improvements.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 use_rope: bool = True,
                 use_flash: bool = True,
                 use_rms_norm: bool = False,
                 use_swiglu: bool = False,
                 max_seq_len: int = 8192):
        super().__init__()
        
        # Normalization layers
        norm_class = RMSNorm if use_rms_norm else LayerNorm
        self.ln_1 = norm_class(d_model)
        self.ln_2 = norm_class(d_model)
        
        # Attention layer
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            use_flash=use_flash,
            max_seq_len=max_seq_len
        )
        
        # Feed-forward layer
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_swiglu=use_swiglu
        )
        
        self.dropout = Dropout(dropout)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Attention block
        ln_1_output = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attention(
            ln_1_output, attention_mask, use_cache=use_cache, past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # FFN block
        ln_2_output = self.ln_2(hidden_states)
        ffn_output = self.ffn(ln_2_output)
        output = hidden_states + self.dropout(ffn_output)
        
        return output, present_key_value


class GPTModel(nn.Module):
    """
    Modern GPT-style transformer model with various improvements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Model configuration
        self.config = config
        self.vocab_size = config.get('vocab_size', 50257)
        self.d_model = config.get('d_model', 768)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 12)
        self.d_ff = config.get('d_ff', 3072)
        self.max_seq_length = config.get('max_seq_length', 1024)
        self.dropout = config.get('dropout', 0.1)
        
        # Advanced features
        self.use_rope = config.get('use_rope', True)
        self.use_flash = config.get('use_flash', True)
        self.use_rms_norm = config.get('use_rms_norm', False)
        self.use_swiglu = config.get('use_swiglu', False)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        self.tie_word_embeddings = config.get('tie_word_embeddings', True)
        
        # Embedding layers
        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        
        if not self.use_rope:
            # Use learned positional embeddings if not using RoPE
            self.position_embedding = Embedding(self.max_seq_length, self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=config.get('activation', 'gelu'),
                use_rope=self.use_rope,
                use_flash=self.use_flash,
                use_rms_norm=self.use_rms_norm,
                use_swiglu=self.use_swiglu,
                max_seq_len=self.max_seq_length
            )
            for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        norm_class = RMSNorm if self.use_rms_norm else LayerNorm
        self.ln_f = norm_class(self.d_model)
        
        # Language modeling head
        if self.tie_word_embeddings:
            self.lm_head = None  # Will use token_embedding.weight
        else:
            self.lm_head = Linear(self.d_model, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {total_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self):
        return self.token_embedding
    
    def set_input_embeddings(self, new_embeddings):
        self.token_embedding = new_embeddings
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GPT model.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Handle past key values
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
        else:
            past_length = 0
            past_key_values = [None] * self.num_layers
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_len + past_length, 
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional embeddings (if not using RoPE)
        if not self.use_rope:
            position_embeds = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeds
        
        # Apply dropout
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Forward through transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, present_kv = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    use_cache,
                    past_key_values[i]
                )
            else:
                hidden_states, present_kv = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=past_key_values[i]
                )
            
            if use_cache:
                present_key_values.append(present_kv)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = F.linear(hidden_states, self.token_embedding.weight)
        
        if not return_dict:
            return (logits, present_key_values) if use_cache else logits
        
        return {
            'logits': logits,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': hidden_states
        }
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 repetition_penalty: float = 1.0,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                # First step - process entire sequence
                model_input = generated
            else:
                # Subsequent steps - only process last token
                model_input = generated[:, -1:]
            
            outputs = self.forward(
                model_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs['logits'][:, -1, :]  # Get last token logits
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(generated.shape[0]):
                    for token in set(generated[i].tolist()):
                        if logits[i, token] < 0:
                            logits[i, token] *= repetition_penalty
                        else:
                            logits[i, token] /= repetition_penalty
            
            # Sample next token
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(logits.shape[0]):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i][indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end of sequence
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated


def create_model_from_config(config_path: str) -> GPTModel:
    """Create a GPT model from a configuration file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model configuration
    model_config = config.get('model', {})
    
    return GPTModel(model_config)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'd_ff': 3072,
        'max_seq_length': 1024,
        'dropout': 0.1,
        'use_rope': True,
        'use_flash': True,
        'use_rms_norm': False,
        'tie_word_embeddings': True
    }
    
    # Create model
    model = GPTModel(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {outputs['logits'].shape}")
        
        # Test generation
        generated = model.generate(
            input_ids[:1, :10],  # Use first sample, first 10 tokens
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True
        )
        print(f"Generated shape: {generated.shape}")

