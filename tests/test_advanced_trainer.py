"""
Unit tests for the Advanced Trainer module.
Tests training configuration, model training, and evaluation functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the modules to test
import sys
sys.path.append('/Users/aponduga/Desktop/Personal/ML_DS/gen_ai_project/src')

try:
    from training.advanced_trainer import (
        TrainingConfig, EMAModel, AdvancedTrainer,
        create_text_generation_trainer, create_diffusion_trainer
    )
except ImportError as e:
    warnings.warn(f"Could not import training modules: {e}")
    TrainingConfig = None


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_config_initialization(self):
        """Test TrainingConfig initialization with default values."""
        config = TrainingConfig(output_dir=self.temp_dir)
        
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.optimizer, "adamw")
        self.assertTrue(config.mixed_precision)
        self.assertTrue(config.use_ema)
        
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            epochs=50,
            batch_size=16,
            learning_rate=5e-5,
            optimizer="adam",
            output_dir=self.temp_dir
        )
        
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.optimizer, "adam")
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_config_directory_creation(self):
        """Test that configuration creates necessary directories."""
        config = TrainingConfig(output_dir=self.temp_dir)
        
        self.assertTrue(Path(config.output_dir).exists())
        self.assertTrue(Path(config.log_dir).exists())
        self.assertTrue(Path(config.checkpoint_dir).exists())


class TestEMAModel(unittest.TestCase):
    """Test cases for EMAModel."""
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_ema_initialization(self):
        """Test EMA model initialization."""
        model = nn.Linear(10, 5)
        ema_model = EMAModel(model, decay=0.999)
        
        self.assertEqual(ema_model.decay, 0.999)
        self.assertEqual(len(ema_model.shadow), len(list(model.named_parameters())))
        
        # Check that shadow parameters are initialized correctly
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertTrue(name in ema_model.shadow)
                self.assertTrue(torch.equal(ema_model.shadow[name], param.data))
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_ema_update(self):
        """Test EMA parameter update."""
        model = nn.Linear(10, 5)
        ema_model = EMAModel(model, decay=0.9)
        
        # Store original shadow values
        original_shadow = {name: param.clone() for name, param in ema_model.shadow.items()}
        
        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.data += 1.0
        
        # Update EMA
        ema_model.update()
        
        # Check that shadow parameters have been updated
        for name in ema_model.shadow:
            self.assertFalse(torch.equal(ema_model.shadow[name], original_shadow[name]))
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_ema_apply_restore(self):
        """Test EMA apply and restore functionality."""
        model = nn.Linear(10, 5)
        ema_model = EMAModel(model, decay=0.999)
        
        # Store original parameters
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # Apply shadow parameters
        ema_model.apply_shadow()
        
        # Parameters should now be equal to shadow
        for name, param in model.named_parameters():
            if name in ema_model.shadow:
                self.assertTrue(torch.equal(param.data, ema_model.shadow[name]))
        
        # Restore original parameters
        ema_model.restore()
        
        # Parameters should be back to original values
        for name, param in model.named_parameters():
            if name in original_params:
                self.assertTrue(torch.equal(param.data, original_params[name]))


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


class TestAdvancedTrainer(unittest.TestCase):
    """Test cases for AdvancedTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data
        self.batch_size = 4
        self.input_size = 10
        self.output_size = 2
        self.num_samples = 20
        
        # Create synthetic dataset
        X = torch.randn(self.num_samples, self.input_size)
        y = torch.randint(0, self.output_size, (self.num_samples,))
        
        dataset = TensorDataset(X, y)
        self.train_dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.eval_dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Create mock model
        self.model = MockModel(self.input_size, self.output_size)
        
        # Create config
        self.config = TrainingConfig(
            epochs=2,  # Small number for testing
            batch_size=self.batch_size,
            learning_rate=1e-3,
            output_dir=self.temp_dir,
            log_dir=str(Path(self.temp_dir) / "logs"),
            checkpoint_dir=str(Path(self.temp_dir) / "checkpoints"),
            eval_every=5,
            save_every=10,
            log_every=2,
            wandb_project=None  # Disable wandb for testing
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_trainer_initialization(self):
        """Test AdvancedTrainer initialization."""
        trainer = AdvancedTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.best_loss, float('inf'))
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_optimizer_creation(self):
        """Test different optimizer creation."""
        # Test AdamW
        config_adamw = TrainingConfig(optimizer="adamw", output_dir=self.temp_dir)
        trainer_adamw = AdvancedTrainer(
            model=self.model,
            config=config_adamw,
            train_dataloader=self.train_dataloader
        )
        self.assertIsInstance(trainer_adamw.optimizer, torch.optim.AdamW)
        
        # Test Adam
        config_adam = TrainingConfig(optimizer="adam", output_dir=self.temp_dir)
        trainer_adam = AdvancedTrainer(
            model=self.model,
            config=config_adam,
            train_dataloader=self.train_dataloader
        )
        self.assertIsInstance(trainer_adam.optimizer, torch.optim.Adam)
        
        # Test SGD
        config_sgd = TrainingConfig(optimizer="sgd", output_dir=self.temp_dir)
        trainer_sgd = AdvancedTrainer(
            model=self.model,
            config=config_sgd,
            train_dataloader=self.train_dataloader
        )
        self.assertIsInstance(trainer_sgd.optimizer, torch.optim.SGD)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_scheduler_creation(self):
        """Test different scheduler creation."""
        # Test Cosine scheduler
        config_cosine = TrainingConfig(scheduler="cosine", output_dir=self.temp_dir)
        trainer_cosine = AdvancedTrainer(
            model=self.model,
            config=config_cosine,
            train_dataloader=self.train_dataloader
        )
        self.assertIsInstance(trainer_cosine.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test OneCycle scheduler
        config_onecycle = TrainingConfig(scheduler="onecycle", output_dir=self.temp_dir)
        trainer_onecycle = AdvancedTrainer(
            model=self.model,
            config=config_onecycle,
            train_dataloader=self.train_dataloader
        )
        self.assertIsInstance(trainer_onecycle.scheduler, torch.optim.lr_scheduler.OneCycleLR)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    @patch('torch.cuda.is_available', return_value=False)  # Force CPU for testing
    def test_train_step(self, mock_cuda):
        """Test single training step."""
        trainer = AdvancedTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        # Get a batch from dataloader
        batch_data = next(iter(self.train_dataloader))
        batch = {'input_ids': batch_data[0], 'labels': batch_data[1]}
        
        # Perform training step
        metrics = trainer.train_step(batch)
        
        self.assertIn('loss', metrics)
        self.assertIn('learning_rate', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['learning_rate'], float)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    @patch('torch.cuda.is_available', return_value=False)  # Force CPU for testing
    def test_evaluation(self, mock_cuda):
        """Test model evaluation."""
        trainer = AdvancedTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        eval_metrics = trainer.evaluate()
        
        self.assertIn('eval_loss', eval_metrics)
        self.assertIsInstance(eval_metrics['eval_loss'], float)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        trainer = AdvancedTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        # Save checkpoint
        trainer.global_step = 100
        trainer.epoch = 5
        trainer.best_loss = 0.5
        trainer.save_checkpoint()
        
        # Check that checkpoint file exists
        checkpoint_files = list(Path(self.config.checkpoint_dir).glob("checkpoint_step_*.pt"))
        self.assertTrue(len(checkpoint_files) > 0)
        
        # Create new trainer and load checkpoint
        new_trainer = AdvancedTrainer(
            model=MockModel(self.input_size, self.output_size),  # New model instance
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        # Load checkpoint
        checkpoint_path = checkpoint_files[0]
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check that state was restored
        self.assertEqual(new_trainer.global_step, 100)
        self.assertEqual(new_trainer.epoch, 5)
        self.assertEqual(new_trainer.best_loss, 0.5)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_training_stats(self):
        """Test training statistics collection."""
        trainer = AdvancedTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader
        )
        
        # Simulate some training steps
        trainer.global_step = 50
        trainer.epoch = 2
        trainer.best_loss = 0.3
        trainer.step_times = [0.1, 0.12, 0.11, 0.13, 0.1]
        trainer.start_time = 1000.0
        
        with patch('time.time', return_value=1100.0):  # Mock current time
            stats = trainer.get_training_stats()
        
        self.assertIn('total_training_time', stats)
        self.assertIn('average_step_time', stats)
        self.assertIn('steps_per_second', stats)
        self.assertIn('total_steps', stats)
        self.assertIn('epochs_completed', stats)
        self.assertIn('best_loss', stats)
        
        self.assertEqual(stats['total_steps'], 50)
        self.assertEqual(stats['epochs_completed'], 2)
        self.assertEqual(stats['best_loss'], 0.3)


class TestTrainerFactories(unittest.TestCase):
    """Test cases for trainer factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=4)
        
        self.model = MockModel(10, 2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_text_generation_trainer_factory(self):
        """Test text generation trainer factory."""
        trainer = create_text_generation_trainer(
            model=self.model,
            train_dataloader=self.dataloader,
            output_dir=self.temp_dir,
            epochs=1
        )
        
        self.assertIsInstance(trainer, AdvancedTrainer)
        self.assertEqual(trainer.config.optimizer, "adamw")
        self.assertEqual(trainer.config.scheduler, "cosine")
        self.assertTrue(trainer.config.mixed_precision)
        self.assertTrue(trainer.config.use_ema)
    
    @unittest.skipIf(TrainingConfig is None, "Training modules not available")
    def test_diffusion_trainer_factory(self):
        """Test diffusion trainer factory."""
        trainer = create_diffusion_trainer(
            model=self.model,
            train_dataloader=self.dataloader,
            output_dir=self.temp_dir,
            epochs=1
        )
        
        self.assertIsInstance(trainer, AdvancedTrainer)
        self.assertEqual(trainer.config.optimizer, "adamw")
        self.assertEqual(trainer.config.scheduler, "cosine")
        self.assertTrue(trainer.config.mixed_precision)
        self.assertTrue(trainer.config.use_ema)
        self.assertEqual(trainer.config.ema_decay, 0.995)  # Diffusion-specific


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
