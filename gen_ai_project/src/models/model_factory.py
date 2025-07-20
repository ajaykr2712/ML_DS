"""
Model registry and factory for generative AI models.
Provides a centralized way to create, configure, and manage different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    model_type: str
    model_name: str
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    vocab_size: Optional[int] = None
    max_length: Optional[int] = None
    
    # Model-specific parameters
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseModelFactory(ABC):
    """Abstract base class for model factories."""
    
    @abstractmethod
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create a model instance from config."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> ModelConfig:
        """Get default configuration for this model type."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ModelConfig) -> bool:
        """Validate model configuration."""
        pass


class TransformerFactory(BaseModelFactory):
    """Factory for creating Transformer models."""
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create a Transformer model."""
        from .transformer import AdvancedTransformer
        
        return AdvancedTransformer(
            vocab_size=config.vocab_size or 50257,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_length=config.max_length or 2048,
            dropout=config.dropout,
            **config.extra_params
        )
    
    def get_default_config(self) -> ModelConfig:
        """Get default Transformer configuration."""
        return ModelConfig(
            model_type="transformer",
            model_name="gpt-small",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            dropout=0.1,
            vocab_size=50257,
            max_length=2048,
            extra_params={
                "use_flash_attention": True,
                "use_rope": True,
                "rope_theta": 10000.0
            }
        )
    
    def validate_config(self, config: ModelConfig) -> bool:
        """Validate Transformer configuration."""
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        
        if config.vocab_size is None:
            raise ValueError("vocab_size is required for Transformer models")
        
        return True


class VAEFactory(BaseModelFactory):
    """Factory for creating VAE models."""
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create a VAE model."""
        # Import VAE model (placeholder - would need actual implementation)
        class SimpleVAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim * 2)  # mu and logvar
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )
                self.latent_dim = latent_dim
            
            def encode(self, x):
                h = self.encoder(x)
                mu, logvar = torch.chunk(h, 2, dim=-1)
                return mu, logvar
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
        
        input_dim = config.extra_params.get("input_dim", 784)
        latent_dim = config.extra_params.get("latent_dim", 20)
        
        return SimpleVAE(
            input_dim=input_dim,
            hidden_dim=config.hidden_size,
            latent_dim=latent_dim
        )
    
    def get_default_config(self) -> ModelConfig:
        """Get default VAE configuration."""
        return ModelConfig(
            model_type="vae",
            model_name="simple-vae",
            hidden_size=512,
            num_layers=2,
            num_heads=1,  # Not used for VAE
            dropout=0.2,
            extra_params={
                "input_dim": 784,
                "latent_dim": 20,
                "beta": 1.0  # For beta-VAE
            }
        )
    
    def validate_config(self, config: ModelConfig) -> bool:
        """Validate VAE configuration."""
        if "input_dim" not in config.extra_params:
            raise ValueError("input_dim is required for VAE models")
        
        if "latent_dim" not in config.extra_params:
            raise ValueError("latent_dim is required for VAE models")
        
        return True


class GANFactory(BaseModelFactory):
    """Factory for creating GAN models."""
    
    def create_model(self, config: ModelConfig) -> Tuple[nn.Module, nn.Module]:
        """Create GAN models (generator and discriminator)."""
        
        class Generator(nn.Module):
            def __init__(self, noise_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(noise_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                )
            
            def forward(self, z):
                return self.net(z)
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        noise_dim = config.extra_params.get("noise_dim", 100)
        output_dim = config.extra_params.get("output_dim", 784)
        
        generator = Generator(noise_dim, config.hidden_size, output_dim)
        discriminator = Discriminator(output_dim, config.hidden_size)
        
        return generator, discriminator
    
    def get_default_config(self) -> ModelConfig:
        """Get default GAN configuration."""
        return ModelConfig(
            model_type="gan",
            model_name="simple-gan",
            hidden_size=256,
            num_layers=3,
            num_heads=1,  # Not used for GAN
            dropout=0.3,
            extra_params={
                "noise_dim": 100,
                "output_dim": 784,
                "lr_generator": 0.0002,
                "lr_discriminator": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999
            }
        )
    
    def validate_config(self, config: ModelConfig) -> bool:
        """Validate GAN configuration."""
        if "noise_dim" not in config.extra_params:
            raise ValueError("noise_dim is required for GAN models")
        
        if "output_dim" not in config.extra_params:
            raise ValueError("output_dim is required for GAN models")
        
        return True


class DiffusionFactory(BaseModelFactory):
    """Factory for creating Diffusion models."""
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create a Diffusion model."""
        
        class SimpleDiffusionModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, time_embed_dim=128):
                super().__init__()
                self.time_embed = nn.Sequential(
                    nn.Linear(1, time_embed_dim),
                    nn.ReLU(),
                    nn.Linear(time_embed_dim, time_embed_dim)
                )
                
                self.net = nn.Sequential(
                    nn.Linear(input_dim + time_embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x, t):
                # t should be normalized timesteps
                t_embed = self.time_embed(t.float().unsqueeze(-1))
                
                # Expand t_embed to match batch size
                if t_embed.dim() == 2 and x.dim() == 3:
                    t_embed = t_embed.unsqueeze(1).expand(-1, x.size(1), -1)
                
                # Concatenate input and time embedding
                combined = torch.cat([x, t_embed], dim=-1)
                return self.net(combined)
        
        input_dim = config.extra_params.get("input_dim", 784)
        time_embed_dim = config.extra_params.get("time_embed_dim", 128)
        
        return SimpleDiffusionModel(
            input_dim=input_dim,
            hidden_dim=config.hidden_size,
            time_embed_dim=time_embed_dim
        )
    
    def get_default_config(self) -> ModelConfig:
        """Get default Diffusion configuration."""
        return ModelConfig(
            model_type="diffusion",
            model_name="simple-diffusion",
            hidden_size=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            extra_params={
                "input_dim": 784,
                "time_embed_dim": 128,
                "num_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule_type": "linear"
            }
        )
    
    def validate_config(self, config: ModelConfig) -> bool:
        """Validate Diffusion configuration."""
        if "input_dim" not in config.extra_params:
            raise ValueError("input_dim is required for Diffusion models")
        
        return True


class ModelRegistry:
    """Central registry for all model factories."""
    
    def __init__(self):
        self._factories: Dict[str, BaseModelFactory] = {}
        self._register_default_factories()
    
    def _register_default_factories(self):
        """Register default model factories."""
        self.register("transformer", TransformerFactory())
        self.register("gpt", TransformerFactory())  # Alias
        self.register("vae", VAEFactory())
        self.register("gan", GANFactory())
        self.register("diffusion", DiffusionFactory())
    
    def register(self, model_type: str, factory: BaseModelFactory):
        """Register a new model factory."""
        self._factories[model_type.lower()] = factory
    
    def create_model(
        self, 
        model_type: str, 
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> nn.Module:
        """Create a model of the specified type."""
        model_type = model_type.lower()
        
        if model_type not in self._factories:
            raise ValueError(f"Unknown model type: {model_type}")
        
        factory = self._factories[model_type]
        
        # Use provided config or create default
        if config is None:
            config = factory.get_default_config()
            # Override with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.extra_params[key] = value
        
        # Validate configuration
        factory.validate_config(config)
        
        return factory.create_model(config)
    
    def get_default_config(self, model_type: str) -> ModelConfig:
        """Get default configuration for a model type."""
        model_type = model_type.lower()
        
        if model_type not in self._factories:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self._factories[model_type].get_default_config()
    
    def list_available_models(self) -> List[str]:
        """List all available model types."""
        return list(self._factories.keys())
    
    def load_config_from_yaml(self, config_path: str) -> ModelConfig:
        """Load model configuration from YAML file."""
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract model config
        model_config = yaml_config.get('model', {})
        
        return ModelConfig(
            model_type=model_config.get('type', 'transformer'),
            model_name=model_config.get('name', 'default'),
            hidden_size=model_config.get('hidden_size', 768),
            num_layers=model_config.get('num_layers', 12),
            num_heads=model_config.get('num_heads', 12),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            vocab_size=model_config.get('vocab_size'),
            max_length=model_config.get('max_length'),
            extra_params=model_config.get('extra_params', {})
        )


# Global model registry instance
model_registry = ModelRegistry()


# Convenience functions
def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model using the global registry."""
    return model_registry.create_model(model_type, **kwargs)


def create_model_from_config(config_path: str) -> nn.Module:
    """Create a model from a YAML configuration file."""
    config = model_registry.load_config_from_yaml(config_path)
    return model_registry.create_model(config.model_type, config)


def register_custom_factory(model_type: str, factory: BaseModelFactory):
    """Register a custom model factory."""
    model_registry.register(model_type, factory)


def list_models() -> List[str]:
    """List all available model types."""
    return model_registry.list_available_models()


# Example usage and model presets
class ModelPresets:
    """Predefined model configurations for common use cases."""
    
    @staticmethod
    def gpt_small() -> ModelConfig:
        """GPT-2 Small configuration."""
        return ModelConfig(
            model_type="transformer",
            model_name="gpt-small",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            dropout=0.1,
            vocab_size=50257,
            max_length=1024,
            extra_params={
                "use_flash_attention": True,
                "use_rope": True
            }
        )
    
    @staticmethod
    def gpt_medium() -> ModelConfig:
        """GPT-2 Medium configuration."""
        return ModelConfig(
            model_type="transformer",
            model_name="gpt-medium",
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            dropout=0.1,
            vocab_size=50257,
            max_length=1024,
            extra_params={
                "use_flash_attention": True,
                "use_rope": True
            }
        )
    
    @staticmethod
    def simple_vae_mnist() -> ModelConfig:
        """VAE configuration for MNIST."""
        return ModelConfig(
            model_type="vae",
            model_name="vae-mnist",
            hidden_size=512,
            dropout=0.2,
            extra_params={
                "input_dim": 784,
                "latent_dim": 20,
                "beta": 1.0
            }
        )
    
    @staticmethod
    def dcgan_mnist() -> ModelConfig:
        """DCGAN configuration for MNIST."""
        return ModelConfig(
            model_type="gan",
            model_name="dcgan-mnist",
            hidden_size=256,
            dropout=0.3,
            extra_params={
                "noise_dim": 100,
                "output_dim": 784,
                "lr_generator": 0.0002,
                "lr_discriminator": 0.0002
            }
        )


if __name__ == "__main__":
    # Example usage
    print("Available models:", list_models())
    
    # Create different types of models
    transformer = create_model("transformer", vocab_size=50000, hidden_size=512)
    vae = create_model("vae", input_dim=784, hidden_size=256, latent_dim=32)
    
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Using presets
    gpt_config = ModelPresets.gpt_small()
    gpt_model = model_registry.create_model(gpt_config.model_type, gpt_config)
    print(f"GPT Small parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")
