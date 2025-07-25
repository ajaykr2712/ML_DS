# Advanced Deep Learning Model Architectures
# =========================================

"""
Custom neural network architectures for various ML tasks including:
- Attention mechanisms
- Residual networks
- Transformer architectures
- Generative models

Author: Deep Learning Team
Date: July 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class AttentionMechanism:
    """Self-attention mechanism implementation."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through attention mechanism."""
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V matrices
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # Apply softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attended = attended.reshape(batch_size, seq_len, self.embedding_dim)
        output = attended @ self.W_o
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class ResidualBlock:
    """Residual block for deep networks."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(input_dim)
        
        # Layer normalization parameters
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through residual block."""
        # Store input for residual connection
        residual = x
        
        # Layer normalization
        x = self._layer_norm(x)
        
        # First linear transformation + activation
        x = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        
        # Dropout during training
        if training and self.dropout_rate > 0:
            x = self._dropout(x, self.dropout_rate)
        
        # Second linear transformation
        x = x @ self.W2 + self.b2
        
        # Residual connection
        return x + residual
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return self.gamma * normalized + self.beta
    
    def _dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Dropout regularization."""
        mask = np.random.random(x.shape) > rate
        return x * mask / (1.0 - rate)

class TransformerEncoder:
    """Transformer encoder implementation."""
    
    def __init__(self, num_layers: int, embedding_dim: int, num_heads: int, ff_dim: int):
        self.num_layers = num_layers
        self.layers = []
        
        for _ in range(num_layers):
            layer = {
                'attention': AttentionMechanism(embedding_dim, num_heads),
                'feedforward': ResidualBlock(embedding_dim, ff_dim)
            }
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through transformer encoder."""
        for layer in self.layers:
            # Self-attention
            attn_output, _ = layer['attention'].forward(x, mask)
            x = x + attn_output  # Residual connection
            
            # Feedforward
            ff_output = layer['feedforward'].forward(x)
            x = ff_output  # Already includes residual connection
        
        return x

class VariationalAutoencoder:
    """Variational Autoencoder for generative modeling."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        self.encoder_layers = self._build_encoder()
        
        # Decoder
        self.decoder_layers = self._build_decoder()
    
    def _build_encoder(self) -> List[Dict]:
        """Build encoder network."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layer = {
                'W': np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim),
                'b': np.zeros(hidden_dim)
            }
            layers.append(layer)
            prev_dim = hidden_dim
        
        # Mean and log variance layers
        layers.append({
            'W_mean': np.random.randn(prev_dim, self.latent_dim) * 0.1,
            'b_mean': np.zeros(self.latent_dim),
            'W_logvar': np.random.randn(prev_dim, self.latent_dim) * 0.1,
            'b_logvar': np.zeros(self.latent_dim)
        })
        
        return layers
    
    def _build_decoder(self) -> List[Dict]:
        """Build decoder network."""
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            layer = {
                'W': np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim),
                'b': np.zeros(hidden_dim)
            }
            layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append({
            'W': np.random.randn(prev_dim, self.input_dim) * np.sqrt(2.0 / prev_dim),
            'b': np.zeros(self.input_dim)
        })
        
        return layers
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent space."""
        h = x
        
        # Forward through encoder layers
        for i, layer in enumerate(self.encoder_layers[:-1]):
            h = np.maximum(0, h @ layer['W'] + layer['b'])  # ReLU
        
        # Get mean and log variance
        final_layer = self.encoder_layers[-1]
        mean = h @ final_layer['W_mean'] + final_layer['b_mean']
        logvar = h @ final_layer['W_logvar'] + final_layer['b_logvar']
        
        return mean, logvar
    
    def reparameterize(self, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mean.shape)
        return mean + eps * std
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to output."""
        h = z
        
        # Forward through decoder layers
        for i, layer in enumerate(self.decoder_layers[:-1]):
            h = np.maximum(0, h @ layer['W'] + layer['b'])  # ReLU
        
        # Output layer with sigmoid activation
        final_layer = self.decoder_layers[-1]
        output = h @ final_layer['W'] + final_layer['b']
        return 1.0 / (1.0 + np.exp(-output))  # Sigmoid
    
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Full forward pass through VAE."""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mean': mean,
            'logvar': logvar,
            'latent': z
        }
    
    def loss(self, x: np.ndarray, output: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute VAE loss."""
        reconstruction = output['reconstruction']
        mean = output['mean']
        logvar = output['logvar']
        
        # Reconstruction loss (binary cross-entropy)
        reconstruction_loss = -np.mean(
            x * np.log(reconstruction + 1e-8) + 
            (1 - x) * np.log(1 - reconstruction + 1e-8)
        )
        
        # KL divergence loss
        kl_loss = -0.5 * np.mean(1 + logvar - mean**2 - np.exp(logvar))
        
        total_loss = reconstruction_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }

# Example usage and testing
if __name__ == "__main__":
    print("Advanced Deep Learning Architectures Demo")
    print("=" * 50)
    
    # Test 1: Attention Mechanism
    print("\n1. Testing Attention Mechanism:")
    attention = AttentionMechanism(embedding_dim=64, num_heads=8)
    
    # Sample input (batch_size=2, seq_len=10, embedding_dim=64)
    x = np.random.randn(2, 10, 64)
    output, weights = attention.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test 2: Residual Block
    print("\n2. Testing Residual Block:")
    residual_block = ResidualBlock(input_dim=128, hidden_dim=256)
    
    x_res = np.random.randn(32, 128)  # Batch of 32 samples
    output_res = residual_block.forward(x_res, training=True)
    
    print(f"Residual input shape: {x_res.shape}")
    print(f"Residual output shape: {output_res.shape}")
    
    # Test 3: Transformer Encoder
    print("\n3. Testing Transformer Encoder:")
    transformer = TransformerEncoder(
        num_layers=3, 
        embedding_dim=64, 
        num_heads=8, 
        ff_dim=256
    )
    
    x_transformer = np.random.randn(4, 20, 64)  # Batch=4, seq_len=20, dim=64
    transformer_output = transformer.forward(x_transformer)
    
    print(f"Transformer input shape: {x_transformer.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Test 4: Variational Autoencoder
    print("\n4. Testing Variational Autoencoder:")
    vae = VariationalAutoencoder(
        input_dim=784,  # MNIST-like input
        latent_dim=20,
        hidden_dims=[512, 256]
    )
    
    x_vae = np.random.rand(16, 784)  # Batch of 16 images
    vae_output = vae.forward(x_vae)
    loss_dict = vae.loss(x_vae, vae_output)
    
    print(f"VAE input shape: {x_vae.shape}")
    print(f"VAE reconstruction shape: {vae_output['reconstruction'].shape}")
    print(f"VAE latent shape: {vae_output['latent'].shape}")
    print(f"VAE total loss: {loss_dict['total_loss']:.4f}")
    print(f"VAE reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    print(f"VAE KL loss: {loss_dict['kl_loss']:.4f}")
    
    print("\nDeep learning architectures demo completed!")
