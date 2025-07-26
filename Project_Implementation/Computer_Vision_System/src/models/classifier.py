"""
Advanced computer vision classifier with multiple architecture support.
Includes attention mechanisms, self-supervised learning, and interpretability features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class AttentionModule(nn.Module):
    """Self-attention module for enhanced feature extraction."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class ResidualBlock(nn.Module):
    """Enhanced residual block with attention and dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = AttentionModule(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        
        out += residual
        return F.relu(out)

class VisionTransformerBlock(nn.Module):
    """Vision Transformer block for image classification."""
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class HybridVisionClassifier(nn.Module):
    """Hybrid classifier combining CNN and Vision Transformer components."""
    
    def __init__(self, num_classes: int, input_channels: int = 3,
                 base_channels: int = 64, num_blocks: int = 4,
                 embed_dim: int = 768, num_heads: int = 12,
                 num_transformer_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # CNN backbone for feature extraction
        self.conv_stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Residual blocks
        self.layers = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1
            self.layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
            in_channels = out_channels
        
        # Adaptive pooling to fixed size for transformer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        
        # Projection to transformer dimension
        final_channels = base_channels * (2 ** (num_blocks - 1))
        self.patch_projection = nn.Linear(final_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 196 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Feature extraction hooks
        self.feature_maps = {}
        self.gradients = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for interpretability."""
        def save_feature_map(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # Register hooks on key layers
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(save_feature_map(f'layer_{i}'))
            layer.register_backward_hook(save_gradient(f'layer_{i}'))
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv_stem(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # Prepare for transformer
        x = self.adaptive_pool(x)  # [B, C, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, C]
        x = self.patch_projection(x)  # [B, 196, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Classification using [CLS] token
        cls_token_final = x[:, 0]  # [B, embed_dim]
        logits = self.classifier(cls_token_final)
        
        if return_features:
            return logits, {
                'cls_token': cls_token_final,
                'feature_maps': self.feature_maps,
                'patch_features': x[:, 1:]  # Exclude [CLS] token
            }
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for interpretability."""
        attention_maps = {}
        
        def attention_hook(module, input, output):
            # Extract attention weights from MultiheadAttention
            if hasattr(module, 'in_proj_weight'):
                attention_maps[id(module)] = output[1]  # Attention weights
        
        # Register temporary hooks
        hooks = []
        for i, layer in enumerate(self.transformer_layers):
            hook = layer.attn.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def grad_cam(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Generate Grad-CAM visualization."""
        # Forward pass
        x.requires_grad_()
        logits = self.forward(x)
        
        # Backward pass for target class
        logits[0, class_idx].backward()
        
        # Get gradients and feature maps
        target_layer_name = f'layer_{len(self.layers)-1}'
        gradients = self.gradients[target_layer_name]
        feature_maps = self.feature_maps[target_layer_name]
        
        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def freeze_backbone(self):
        """Freeze CNN backbone for fine-tuning."""
        for param in self.conv_stem.parameters():
            param.requires_grad = False
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze CNN backbone."""
        for param in self.conv_stem.parameters():
            param.requires_grad = True
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True

class EfficientClassifier(nn.Module):
    """Memory and computationally efficient classifier for mobile deployment."""
    
    def __init__(self, num_classes: int, width_multiplier: float = 1.0):
        super().__init__()
        
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Define channel sizes based on width multiplier
        self.channels = [
            make_divisible(32 * width_multiplier),
            make_divisible(64 * width_multiplier),
            make_divisible(128 * width_multiplier),
            make_divisible(256 * width_multiplier),
            make_divisible(512 * width_multiplier)
        ]
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU6()
        )
        
        # Depthwise separable convolutions
        self.layers = nn.ModuleList()
        in_channels = self.channels[0]
        
        for out_channels in self.channels[1:]:
            self.layers.append(self._make_layer(in_channels, out_channels))
            in_channels = out_channels
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.channels[-1], num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create depthwise separable convolution layer."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            
            # Stride for downsampling
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def create_classifier(architecture: str, num_classes: int, **kwargs) -> nn.Module:
    """Factory function to create different classifier architectures."""
    if architecture.lower() == 'hybrid':
        return HybridVisionClassifier(num_classes, **kwargs)
    elif architecture.lower() == 'efficient':
        return EfficientClassifier(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
