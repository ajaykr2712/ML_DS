"""
Unit tests for the Model Factory module.
Tests model creation, configuration, and registry functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import yaml

# Import the modules to test
import sys
sys.path.append('/Users/aponduga/Desktop/Personal/ML_DS/gen_ai_project/src')

try:
    from models.model_factory import (
        ModelConfig, BaseModelFactory, TransformerFactory,
        VAEFactory, GANFactory, DiffusionFactory,
        ModelRegistry, create_model, ModelPresets
    )
except ImportError as e:
    warnings.warn(f"Could not import model factory modules: {e}")
    ModelConfig = None


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_initialization(self):
        """Test ModelConfig initialization with default values."""
        config = ModelConfig(
            model_type="transformer",
            model_name="test-model"
        )
        
        self.assertEqual(config.model_type, "transformer")
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.activation, "gelu")
        self.assertIsNone(config.vocab_size)
        self.assertIsNone(config.max_length)
        self.assertEqual(config.extra_params, {})
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_custom_values(self):
        """Test ModelConfig with custom values."""
        extra_params = {"use_flash_attention": True, "rope_theta": 10000.0}
        
        config = ModelConfig(
            model_type="transformer",
            model_name="custom-model",
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            dropout=0.2,
            vocab_size=50257,
            max_length=2048,
            extra_params=extra_params
        )
        
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.vocab_size, 50257)
        self.assertEqual(config.max_length, 2048)
        self.assertEqual(config.extra_params, extra_params)


class TestTransformerFactory(unittest.TestCase):
    """Test cases for TransformerFactory."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_default_config(self):
        """Test default transformer configuration."""
        factory = TransformerFactory()
        config = factory.get_default_config()
        
        self.assertEqual(config.model_type, "transformer")
        self.assertEqual(config.model_name, "gpt-small")
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.vocab_size, 50257)
        self.assertEqual(config.max_length, 2048)
        self.assertIn("use_flash_attention", config.extra_params)
        self.assertIn("use_rope", config.extra_params)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_validation(self):
        """Test transformer configuration validation."""
        factory = TransformerFactory()
        
        # Valid config
        valid_config = ModelConfig(
            model_type="transformer",
            model_name="test",
            hidden_size=768,
            num_heads=12,
            vocab_size=50257
        )
        self.assertTrue(factory.validate_config(valid_config))
        
        # Invalid config - hidden_size not divisible by num_heads
        invalid_config = ModelConfig(
            model_type="transformer",
            model_name="test",
            hidden_size=770,  # Not divisible by 12
            num_heads=12,
            vocab_size=50257
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config)
        
        # Invalid config - missing vocab_size
        missing_vocab_config = ModelConfig(
            model_type="transformer",
            model_name="test",
            hidden_size=768,
            num_heads=12
        )
        with self.assertRaises(ValueError):
            factory.validate_config(missing_vocab_config)


class TestVAEFactory(unittest.TestCase):
    """Test cases for VAEFactory."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_default_config(self):
        """Test default VAE configuration."""
        factory = VAEFactory()
        config = factory.get_default_config()
        
        self.assertEqual(config.model_type, "vae")
        self.assertEqual(config.model_name, "simple-vae")
        self.assertEqual(config.hidden_size, 512)
        self.assertIn("input_dim", config.extra_params)
        self.assertIn("latent_dim", config.extra_params)
        self.assertIn("beta", config.extra_params)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_validation(self):
        """Test VAE configuration validation."""
        factory = VAEFactory()
        
        # Valid config
        valid_config = ModelConfig(
            model_type="vae",
            model_name="test",
            extra_params={"input_dim": 784, "latent_dim": 20}
        )
        self.assertTrue(factory.validate_config(valid_config))
        
        # Invalid config - missing input_dim
        invalid_config1 = ModelConfig(
            model_type="vae",
            model_name="test",
            extra_params={"latent_dim": 20}
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config1)
        
        # Invalid config - missing latent_dim
        invalid_config2 = ModelConfig(
            model_type="vae",
            model_name="test",
            extra_params={"input_dim": 784}
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config2)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_model_creation(self):
        """Test VAE model creation."""
        factory = VAEFactory()
        config = ModelConfig(
            model_type="vae",
            model_name="test",
            hidden_size=256,
            extra_params={"input_dim": 784, "latent_dim": 20}
        )
        
        model = factory.create_model(config)
        
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, 'encode'))
        self.assertTrue(hasattr(model, 'decode'))
        self.assertTrue(hasattr(model, 'reparameterize'))
        self.assertTrue(hasattr(model, 'forward'))
        self.assertEqual(model.latent_dim, 20)


class TestGANFactory(unittest.TestCase):
    """Test cases for GANFactory."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_default_config(self):
        """Test default GAN configuration."""
        factory = GANFactory()
        config = factory.get_default_config()
        
        self.assertEqual(config.model_type, "gan")
        self.assertEqual(config.model_name, "simple-gan")
        self.assertIn("noise_dim", config.extra_params)
        self.assertIn("output_dim", config.extra_params)
        self.assertIn("lr_generator", config.extra_params)
        self.assertIn("lr_discriminator", config.extra_params)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_validation(self):
        """Test GAN configuration validation."""
        factory = GANFactory()
        
        # Valid config
        valid_config = ModelConfig(
            model_type="gan",
            model_name="test",
            extra_params={"noise_dim": 100, "output_dim": 784}
        )
        self.assertTrue(factory.validate_config(valid_config))
        
        # Invalid config - missing noise_dim
        invalid_config1 = ModelConfig(
            model_type="gan",
            model_name="test",
            extra_params={"output_dim": 784}
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config1)
        
        # Invalid config - missing output_dim
        invalid_config2 = ModelConfig(
            model_type="gan",
            model_name="test",
            extra_params={"noise_dim": 100}
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config2)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_model_creation(self):
        """Test GAN model creation."""
        factory = GANFactory()
        config = ModelConfig(
            model_type="gan",
            model_name="test",
            hidden_size=256,
            extra_params={"noise_dim": 100, "output_dim": 784}
        )
        
        generator, discriminator = factory.create_model(config)
        
        self.assertIsInstance(generator, nn.Module)
        self.assertIsInstance(discriminator, nn.Module)
        
        # Test generator
        noise = torch.randn(1, 100)
        generated = generator(noise)
        self.assertEqual(generated.shape, (1, 784))
        
        # Test discriminator
        fake_data = torch.randn(1, 784)
        discrimination = discriminator(fake_data)
        self.assertEqual(discrimination.shape, (1, 1))


class TestDiffusionFactory(unittest.TestCase):
    """Test cases for DiffusionFactory."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_default_config(self):
        """Test default Diffusion configuration."""
        factory = DiffusionFactory()
        config = factory.get_default_config()
        
        self.assertEqual(config.model_type, "diffusion")
        self.assertEqual(config.model_name, "simple-diffusion")
        self.assertIn("input_dim", config.extra_params)
        self.assertIn("time_embed_dim", config.extra_params)
        self.assertIn("num_timesteps", config.extra_params)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_config_validation(self):
        """Test Diffusion configuration validation."""
        factory = DiffusionFactory()
        
        # Valid config
        valid_config = ModelConfig(
            model_type="diffusion",
            model_name="test",
            extra_params={"input_dim": 784}
        )
        self.assertTrue(factory.validate_config(valid_config))
        
        # Invalid config - missing input_dim
        invalid_config = ModelConfig(
            model_type="diffusion",
            model_name="test",
            extra_params={}
        )
        with self.assertRaises(ValueError):
            factory.validate_config(invalid_config)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_model_creation(self):
        """Test Diffusion model creation."""
        factory = DiffusionFactory()
        config = ModelConfig(
            model_type="diffusion",
            model_name="test",
            hidden_size=256,
            extra_params={"input_dim": 784, "time_embed_dim": 128}
        )
        
        model = factory.create_model(config)
        
        self.assertIsInstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 784)
        t = torch.randint(0, 1000, (2,))
        output = model(x, t)
        self.assertEqual(output.shape, (2, 784))


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_registry_initialization(self):
        """Test ModelRegistry initialization."""
        registry = ModelRegistry()
        
        available_models = registry.list_available_models()
        
        self.assertIn("transformer", available_models)
        self.assertIn("gpt", available_models)
        self.assertIn("vae", available_models)
        self.assertIn("gan", available_models)
        self.assertIn("diffusion", available_models)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_model_creation_from_registry(self):
        """Test model creation through registry."""
        registry = ModelRegistry()
        
        # Create VAE model
        vae_model = registry.create_model(
            "vae",
            hidden_size=256,
            input_dim=784,
            latent_dim=20
        )
        self.assertIsInstance(vae_model, nn.Module)
        
        # Create transformer model with custom config
        transformer_config = ModelConfig(
            model_type="transformer",
            model_name="custom-gpt",
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            vocab_size=30000,
            max_length=1024
        )
        transformer_model = registry.create_model("transformer", transformer_config)
        self.assertIsInstance(transformer_model, nn.Module)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_unknown_model_type(self):
        """Test error handling for unknown model type."""
        registry = ModelRegistry()
        
        with self.assertRaises(ValueError):
            registry.create_model("unknown_model_type")
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        registry = ModelRegistry()
        
        # Create temporary YAML config
        config_data = {
            'model': {
                'type': 'transformer',
                'name': 'yaml-test-model',
                'hidden_size': 512,
                'num_layers': 8,
                'num_heads': 8,
                'vocab_size': 25000,
                'max_length': 1024,
                'extra_params': {
                    'use_flash_attention': True,
                    'dropout': 0.15
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = registry.load_config_from_yaml(config_path)
            
            self.assertEqual(config.model_type, 'transformer')
            self.assertEqual(config.model_name, 'yaml-test-model')
            self.assertEqual(config.hidden_size, 512)
            self.assertEqual(config.num_layers, 8)
            self.assertEqual(config.num_heads, 8)
            self.assertEqual(config.vocab_size, 25000)
            self.assertEqual(config.max_length, 1024)
            self.assertIn('use_flash_attention', config.extra_params)
            self.assertEqual(config.extra_params['dropout'], 0.15)
        finally:
            Path(config_path).unlink()


class TestModelPresets(unittest.TestCase):
    """Test cases for ModelPresets."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_gpt_small_preset(self):
        """Test GPT Small preset configuration."""
        config = ModelPresets.gpt_small()
        
        self.assertEqual(config.model_type, "transformer")
        self.assertEqual(config.model_name, "gpt-small")
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.vocab_size, 50257)
        self.assertEqual(config.max_length, 1024)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_gpt_medium_preset(self):
        """Test GPT Medium preset configuration."""
        config = ModelPresets.gpt_medium()
        
        self.assertEqual(config.model_type, "transformer")
        self.assertEqual(config.model_name, "gpt-medium")
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.vocab_size, 50257)
        self.assertEqual(config.max_length, 1024)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_simple_vae_mnist_preset(self):
        """Test Simple VAE MNIST preset configuration."""
        config = ModelPresets.simple_vae_mnist()
        
        self.assertEqual(config.model_type, "vae")
        self.assertEqual(config.model_name, "vae-mnist")
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.extra_params["input_dim"], 784)
        self.assertEqual(config.extra_params["latent_dim"], 20)
        self.assertEqual(config.extra_params["beta"], 1.0)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_dcgan_mnist_preset(self):
        """Test DCGAN MNIST preset configuration."""
        config = ModelPresets.dcgan_mnist()
        
        self.assertEqual(config.model_type, "gan")
        self.assertEqual(config.model_name, "dcgan-mnist")
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.extra_params["noise_dim"], 100)
        self.assertEqual(config.extra_params["output_dim"], 784)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_create_model_function(self):
        """Test global create_model function."""
        model = create_model("vae", input_dim=784, latent_dim=20, hidden_size=256)
        self.assertIsInstance(model, nn.Module)
    
    @unittest.skipIf(ModelConfig is None, "Model factory modules not available")
    def test_create_model_from_config(self):
        """Test create_model_from_config function."""
        # Create temporary YAML config
        config_data = {
            'model': {
                'type': 'vae',
                'name': 'test-vae',
                'hidden_size': 256,
                'extra_params': {
                    'input_dim': 784,
                    'latent_dim': 20
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            from models.model_factory import create_model_from_config
            model = create_model_from_config(config_path)
            self.assertIsInstance(model, nn.Module)
        finally:
            Path(config_path).unlink()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
