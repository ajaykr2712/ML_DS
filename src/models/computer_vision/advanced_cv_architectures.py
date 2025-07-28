"""
Advanced Computer Vision Module
Implements state-of-the-art computer vision architectures and techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    embed_dim: int
    num_heads: int
    dropout: float = 0.1
    bias: bool = True

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        
        assert self.head_dim * config.num_heads == config.embed_dim
        
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class PatchEmbedding(nn.Module):
    """Image to patch embedding for Vision Transformers."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        
        x = self.proj(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            AttentionConfig(embed_dim, num_heads, dropout)
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output

class VisionTransformer(nn.Module):
    """Vision Transformer implementation."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, 
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use class token
        x = self.head(x)
        
        return x

class ConvMixer(nn.Module):
    """ConvMixer architecture for efficient image classification."""
    
    def __init__(self, dim: int = 256, depth: int = 8, kernel_size: int = 9,
                 patch_size: int = 7, num_classes: int = 1000):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                # Pointwise convolution
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)
        ])
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
            
        x = self.head(x)
        return x

class EfficientAttention(nn.Module):
    """Efficient attention mechanism for large images."""
    
    def __init__(self, dim: int, num_heads: int = 8, reduction_ratio: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.reduction_ratio = reduction_ratio
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # Reduction for keys and values
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
            
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key and Value with spatial reduction
        if hasattr(self, 'sr'):
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv.unbind(0)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class PyramidVisionTransformer(nn.Module):
    """Pyramid Vision Transformer for dense prediction tasks."""
    
    def __init__(self, img_size: int = 224, patch_sizes: List[int] = [4, 2, 2, 2],
                 embed_dims: List[int] = [64, 128, 256, 512],
                 num_heads: List[int] = [1, 2, 4, 8],
                 mlp_ratios: List[int] = [8, 8, 4, 4],
                 depths: List[int] = [3, 4, 6, 3],
                 num_classes: int = 1000):
        super().__init__()
        
        self.num_stages = len(embed_dims)
        
        # Patch embeddings for each stage
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = nn.Conv2d(3, embed_dims[i], kernel_size=patch_sizes[i], 
                                      stride=patch_sizes[i])
            else:
                patch_embed = nn.Conv2d(embed_dims[i-1], embed_dims[i], 
                                      kernel_size=patch_sizes[i], stride=patch_sizes[i])
            
            self.patch_embeds.append(patch_embed)
            
            # Position embedding
            if i == 0:
                num_patches = (img_size // patch_sizes[i]) ** 2
            else:
                num_patches = num_patches // (patch_sizes[i] ** 2)
            
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            self.pos_embeds.append(pos_embed)
            self.pos_drops.append(nn.Dropout(0.1))
        
        # Transformer blocks for each stage
        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embed_dims[i]),
                    EfficientAttention(embed_dims[i], num_heads[i]),
                    nn.LayerNorm(embed_dims[i]),
                    nn.Sequential(
                        nn.Linear(embed_dims[i], embed_dims[i] * mlp_ratios[i]),
                        nn.GELU(),
                        nn.Linear(embed_dims[i] * mlp_ratios[i], embed_dims[i])
                    )
                ) for _ in range(depths[i])
            ])
            self.blocks.append(stage_blocks)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        for i in range(self.num_stages):
            # Patch embedding
            x = self.patch_embeds[i](x)
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            
            # Position embedding
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            
            # Transformer blocks
            for block in self.blocks[i]:
                x = x + block(x)
            
            # Reshape for next stage (except last)
            if i < self.num_stages - 1:
                x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x

class ObjectDetectionHead(nn.Module):
    """Object detection head for YOLO-style detection."""
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Predictions: [x, y, w, h, confidence] + class_probabilities
        self.pred_size = 5 + num_classes
        
        self.conv = nn.Conv2d(in_channels, num_anchors * self.pred_size, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # Apply convolution
        prediction = self.conv(x)
        
        # Reshape to [batch, anchors, grid_h, grid_w, predictions]
        prediction = prediction.view(
            batch_size, self.num_anchors, self.pred_size, height, width
        ).permute(0, 1, 3, 4, 2).contiguous()
        
        return prediction

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: List of feature maps from different stages, ordered from low to high resolution
        """
        # Start from the highest level (smallest feature map)
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        # Process from high to low level
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results

class ComputerVisionModel(nn.Module):
    """Unified computer vision model supporting multiple architectures."""
    
    def __init__(self, architecture: str = 'vit', **kwargs):
        super().__init__()
        self.architecture = architecture
        
        if architecture == 'vit':
            self.model = VisionTransformer(**kwargs)
        elif architecture == 'convmixer':
            self.model = ConvMixer(**kwargs)
        elif architecture == 'pvt':
            self.model = PyramidVisionTransformer(**kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Factory functions for common configurations
def create_vit_small(num_classes: int = 1000) -> VisionTransformer:
    """Create a small Vision Transformer."""
    return VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, num_classes=num_classes
    )

def create_vit_base(num_classes: int = 1000) -> VisionTransformer:
    """Create a base Vision Transformer."""
    return VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, num_classes=num_classes
    )

def create_convmixer_small() -> ConvMixer:
    """Create a small ConvMixer."""
    return ConvMixer(dim=256, depth=8, kernel_size=9, patch_size=7)

def demo_computer_vision():
    """Demonstrate computer vision capabilities."""
    # Create models
    vit = create_vit_small(num_classes=10)
    convmixer = create_convmixer_small()
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print("Computer Vision Models Demo:")
    
    # Test ViT
    with torch.no_grad():
        vit_output = vit(x)
        print(f"ViT output shape: {vit_output.shape}")
    
    # Test ConvMixer
    with torch.no_grad():
        conv_output = convmixer(x)
        print(f"ConvMixer output shape: {conv_output.shape}")
    
    # Test Feature Pyramid Network
    feature_maps = [
        torch.randn(batch_size, 64, 56, 56),
        torch.randn(batch_size, 128, 28, 28),
        torch.randn(batch_size, 256, 14, 14),
    ]
    
    fpn = FeaturePyramidNetwork([64, 128, 256], 256)
    fpn_outputs = fpn(feature_maps)
    print(f"FPN outputs: {[out.shape for out in fpn_outputs]}")

if __name__ == "__main__":
    demo_computer_vision()
