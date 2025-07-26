"""
Configuration management for ML experiments.
Provides structured configuration handling with validation and environment-specific settings.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    name: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    max_sequence_length: int = 512
    vocab_size: int = 50257
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    early_stopping_patience: int = 5
    
    # Advanced training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_name: str = "custom"
    data_path: str = "./data"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_samples: Optional[int] = None
    
    # Preprocessing
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Augmentation
    enable_augmentation: bool = False
    augmentation_probability: float = 0.1
    
    def validate(self) -> None:
        """Validate data configuration."""
        if not os.path.exists(self.data_path):
            logging.warning(f"Data path {self.data_path} does not exist")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Environment
    device: str = "auto"
    seed: int = 42
    
    # Monitoring
    wandb_project: Optional[str] = None
    tensorboard_log: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        self.setup_directories()
    
    def validate(self) -> None:
        """Validate all configuration components."""
        self.model.validate()
        self.training.validate()
        self.data.validate()
        
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Extract remaining fields for main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'training', 'data']}
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'tags': self.tags,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'seed': self.seed,
            'wandb_project': self.wandb_project,
            'tensorboard_log': self.tensorboard_log
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        if config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def merge_from_dict(self, override_dict: Dict[str, Any]) -> None:
        """Merge configuration with override dictionary."""
        for key, value in override_dict.items():
            if key == 'model' and isinstance(value, dict):
                for model_key, model_value in value.items():
                    if hasattr(self.model, model_key):
                        setattr(self.model, model_key, model_value)
            elif key == 'training' and isinstance(value, dict):
                for train_key, train_value in value.items():
                    if hasattr(self.training, train_key):
                        setattr(self.training, train_key, train_value)
            elif key == 'data' and isinstance(value, dict):
                for data_key, data_value in value.items():
                    if hasattr(self.data, data_key):
                        setattr(self.data, data_key, data_value)
            elif hasattr(self, key):
                setattr(self, key, value)

class ConfigManager:
    """Manager for handling multiple configurations and environments."""
    
    def __init__(self, base_config_dir: str = "./configs"):
        self.base_config_dir = Path(base_config_dir)
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs = {}
        self.active_config = None
    
    def register_config(self, name: str, config: ExperimentConfig) -> None:
        """Register a configuration."""
        self.configs[name] = config
    
    def load_config(self, name: str, config_path: Optional[str] = None) -> ExperimentConfig:
        """Load configuration by name or path."""
        if config_path:
            config = ExperimentConfig.from_file(config_path)
        elif name in self.configs:
            config = self.configs[name]
        else:
            # Try loading from standard location
            standard_path = self.base_config_dir / f"{name}.yaml"
            if standard_path.exists():
                config = ExperimentConfig.from_file(str(standard_path))
            else:
                raise ValueError(f"Configuration '{name}' not found")
        
        self.active_config = config
        return config
    
    def save_config(self, name: str, config: ExperimentConfig) -> None:
        """Save configuration to standard location."""
        config_path = self.base_config_dir / f"{name}.yaml"
        config.save(str(config_path))
        self.configs[name] = config
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        config_files = list(self.base_config_dir.glob("*.yaml")) + \
                      list(self.base_config_dir.glob("*.json"))
        config_names = [f.stem for f in config_files]
        config_names.extend(self.configs.keys())
        return sorted(set(config_names))
    
    def get_active_config(self) -> Optional[ExperimentConfig]:
        """Get currently active configuration."""
        return self.active_config

# Environment-specific configurations
def get_development_config() -> ExperimentConfig:
    """Get configuration for development environment."""
    config = ExperimentConfig()
    config.experiment_name = "development"
    config.training.batch_size = 8
    config.training.num_epochs = 2
    config.training.save_every = 100
    config.training.eval_every = 50
    config.model.num_layers = 6
    config.data.max_samples = 1000
    return config

def get_production_config() -> ExperimentConfig:
    """Get configuration for production environment."""
    config = ExperimentConfig()
    config.experiment_name = "production"
    config.training.batch_size = 64
    config.training.num_epochs = 50
    config.training.mixed_precision = True
    config.model.num_layers = 24
    config.model.hidden_size = 1024
    return config

def get_config_for_environment(env: str = "development") -> ExperimentConfig:
    """Get configuration based on environment."""
    env = env.lower()
    if env == "development" or env == "dev":
        return get_development_config()
    elif env == "production" or env == "prod":
        return get_production_config()
    else:
        # Default configuration
        return ExperimentConfig()
