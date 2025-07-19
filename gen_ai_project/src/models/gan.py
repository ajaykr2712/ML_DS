"""
StyleGAN2 implementation for high-quality image generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, lr_mul: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None
        
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.scale
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, weight, bias)


class EqualizedConv2d(nn.Module):
    """Conv2d layer with equalized learning rate."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0,
        bias: bool = True, 
        lr_mul: float = 1.0
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        self.stride = stride
        self.padding = padding
        self.scale = (1 / math.sqrt(in_channels * kernel_size * kernel_size)) * lr_mul
        self.lr_mul = lr_mul
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.scale
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, weight, bias, self.stride, self.padding)


class ModulatedConv2d(nn.Module):
    """Modulated convolution for style injection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        downsample: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        
        # Weight and modulation
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = EqualizedLinear(style_dim, in_channels, bias=True)
        
        # Blur kernel for upsampling/downsampling
        self.register_buffer('blur_kernel', self._make_blur_kernel())
        
    def _make_blur_kernel(self) -> torch.Tensor:
        """Create blur kernel for antialiasing."""
        kernel = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        return kernel[None, None, :, :]
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Modulate weights
        style = self.modulation(style).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.weight * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)
        
        # Reshape for grouped convolution
        weight = weight.view(
            batch_size * self.out_channels, self.in_channels, 
            self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])
        
        # Apply convolution
        x = F.conv2d(x, weight, padding=1, groups=batch_size)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])
        
        return x


class StyleBlock(nn.Module):
    """Style-based generator block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        upsample: bool = False
    ):
        super().__init__()
        
        self.conv1 = ModulatedConv2d(
            in_channels, out_channels, 3, style_dim, upsample=upsample
        )
        self.conv2 = ModulatedConv2d(
            out_channels, out_channels, 3, style_dim
        )
        
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(
        self, 
        x: torch.Tensor, 
        style: torch.Tensor, 
        noise1: Optional[torch.Tensor] = None,
        noise2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.conv1(x, style)
        x = self.noise1(x, noise1)
        x = self.activation(x)
        
        x = self.conv2(x, style)
        x = self.noise2(x, noise2)
        x = self.activation(x)
        
        return x


class NoiseInjection(nn.Module):
    """Inject noise into feature maps."""
    
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class MappingNetwork(nn.Module):
    """Maps latent codes to intermediate latent space."""
    
    def __init__(self, latent_dim: int, style_dim: int, num_layers: int = 8):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                EqualizedLinear(
                    latent_dim if i == 0 else style_dim, 
                    style_dim
                ),
                nn.LeakyReLU(0.2)
            ])
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)


class Generator(nn.Module):
    """StyleGAN2 Generator."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        num_layers: int = 8,
        image_size: int = 256,
        base_channels: int = 512
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        # Mapping network
        self.mapping = MappingNetwork(latent_dim, style_dim, num_layers)
        
        # Constant input
        self.const_input = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        in_channels = base_channels
        
        # Calculate number of blocks needed
        num_blocks = int(math.log2(image_size)) - 1
        
        for i in range(num_blocks):
            out_channels = max(base_channels // (2 ** (i + 1)), 32)
            upsample = i > 0
            
            self.blocks.append(
                StyleBlock(in_channels, out_channels, style_dim, upsample=upsample)
            )
            in_channels = out_channels
        
        # Output layer
        self.to_rgb = ModulatedConv2d(in_channels, 3, 1, style_dim)
        
    def forward(
        self, 
        z: torch.Tensor, 
        truncation_psi: float = 1.0,
        inject_noise: bool = True
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Map to style space
        w = self.mapping(z)
        
        # Truncation trick
        if truncation_psi < 1.0:
            w_avg = w.mean(dim=0, keepdim=True)
            w = w_avg + truncation_psi * (w - w_avg)
        
        # Start with constant input
        x = self.const_input.repeat(batch_size, 1, 1, 1)
        
        # Generate through blocks
        for block in self.blocks:
            noise1 = torch.randn_like(x[:, :1, :, :]) if inject_noise else None
            noise2 = torch.randn_like(x[:, :1, :, :]) if inject_noise else None
            x = block(x, w, noise1, noise2)
        
        # Convert to RGB
        rgb = self.to_rgb(x, w)
        return torch.tanh(rgb)


class Discriminator(nn.Module):
    """StyleGAN2 Discriminator."""
    
    def __init__(
        self,
        image_size: int = 256,
        base_channels: int = 64,
        max_channels: int = 512
    ):
        super().__init__()
        
        channels = [3]  # RGB input
        current_size = image_size
        current_channels = base_channels
        
        # Calculate channel progression
        while current_size > 4:
            channels.append(min(current_channels, max_channels))
            current_channels *= 2
            current_size //= 2
        
        # Convolutional blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                DiscriminatorBlock(
                    channels[i], 
                    channels[i + 1],
                    downsample=i > 0
                )
            )
        
        # Final block
        self.final_conv = EqualizedConv2d(channels[-1], channels[-1], 3, padding=1)
        self.final_linear = EqualizedLinear(channels[-1] * 4 * 4, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        
        x = F.leaky_relu(self.final_conv(x), 0.2)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.AvgPool2d(2)
        
        # Skip connection
        self.skip = EqualizedConv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        
        if self.downsample:
            x = self.downsample_layer(x)
            skip = self.downsample_layer(self.skip(skip))
        else:
            skip = self.skip(skip)
        
        return (x + skip) / math.sqrt(2)


class StyleGAN(nn.Module):
    """Complete StyleGAN2 model."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        image_size: int = 256,
        base_channels: int = 512
    ):
        super().__init__()
        
        self.generator = Generator(
            latent_dim=latent_dim,
            style_dim=style_dim,
            image_size=image_size,
            base_channels=base_channels
        )
        
        self.discriminator = Discriminator(
            image_size=image_size
        )
    
    def generate(
        self, 
        batch_size: int = 1, 
        device: torch.device = torch.device('cpu'),
        truncation_psi: float = 0.7
    ) -> torch.Tensor:
        """Generate random images."""
        z = torch.randn(batch_size, self.generator.latent_dim, device=device)
        return self.generator(z, truncation_psi=truncation_psi)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleGAN(latent_dim=512, image_size=64).to(device)
    
    # Test generation
    with torch.no_grad():
        fake_images = model.generate(batch_size=4, device=device)
        print(f"Generated images shape: {fake_images.shape}")
    
    # Test discriminator
    real_images = torch.randn(4, 3, 64, 64, device=device)
    with torch.no_grad():
        real_scores = model.discriminator(real_images)
        fake_scores = model.discriminator(fake_images)
        print(f"Real scores: {real_scores.mean().item():.3f}")
        print(f"Fake scores: {fake_scores.mean().item():.3f}")
