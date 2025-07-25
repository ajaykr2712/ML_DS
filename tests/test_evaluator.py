"""
Unit tests for the Evaluation module.
Tests model evaluation, metrics calculation, and report generation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import warnings
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import the modules to test
import sys
sys.path.append('/Users/aponduga/Desktop/Personal/ML_DS/scripts')

try:
    from evaluate_models import ModelEvaluator
except ImportError as e:
    warnings.warn(f"Could not import evaluation modules: {e}")
    ModelEvaluator = None


class MockGenerativeModel(nn.Module):
    """Mock generative model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        # Simple sum pooling
        x = x.mean(dim=1)
        return self.linear(x)
    
    def compute_loss(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.current:self.current + self.batch_size]
        self.current += self.batch_size
        
        # Convert to tensors
        input_ids = torch.randint(0, 1000, (len(batch), 10))
        labels = torch.randint(0, 1000, (len(batch),))
        
        return {'input_ids': input_ids, 'labels': labels}


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.classification_data = make_classification(
            n_samples=100, n_features=20, n_classes=3, random_state=42
        )
        self.regression_data = make_regression(
            n_samples=100, n_features=20, random_state=42
        )
        
        # Split data
        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = \
            train_test_split(*self.classification_data, test_size=0.3, random_state=42)
        
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = \
            train_test_split(*self.regression_data, test_size=0.3, random_state=42)
        
        # Train models
        self.clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf_model.fit(self.X_train_clf, self.y_train_clf)
        
        self.reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.reg_model.fit(self.X_train_reg, self.y_train_reg)
        
        # Create generative model and mock dataloader
        self.gen_model = MockGenerativeModel()
        self.mock_dataloader = MockDataLoader(list(range(20)))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        self.assertEqual(evaluator.output_dir, Path(self.temp_dir))
        self.assertTrue(evaluator.output_dir.exists())
        self.assertEqual(evaluator.results, {})
        self.assertIsNotNone(evaluator.timestamp)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_classification_evaluation(self):
        """Test classification model evaluation."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        results = evaluator.evaluate_classification_model(
            model=self.clf_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="TestClassifier",
            class_names=['Class_A', 'Class_B', 'Class_C']
        )
        
        # Check results structure
        self.assertIn('model_name', results)
        self.assertIn('model_type', results)
        self.assertIn('metrics', results)
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('predictions', results)
        self.assertIn('true_labels', results)
        
        # Check model info
        self.assertEqual(results['model_name'], 'TestClassifier')
        self.assertEqual(results['model_type'], 'classification')
        
        # Check metrics
        metrics = results['metrics']
        required_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted'
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
        
        # Check that results are stored
        self.assertIn('TestClassifier', evaluator.results)
        
        # Check that confusion matrix plot is created
        plot_files = list(Path(self.temp_dir).glob('*confusion_matrix*.png'))
        self.assertTrue(len(plot_files) > 0)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_regression_evaluation(self):
        """Test regression model evaluation."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        results = evaluator.evaluate_regression_model(
            model=self.reg_model,
            X_test=self.X_test_reg,
            y_test=self.y_test_reg,
            model_name="TestRegressor"
        )
        
        # Check results structure
        self.assertIn('model_name', results)
        self.assertIn('model_type', results)
        self.assertIn('metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('true_values', results)
        
        # Check model info
        self.assertEqual(results['model_name'], 'TestRegressor')
        self.assertEqual(results['model_type'], 'regression')
        
        # Check metrics
        metrics = results['metrics']
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
        
        # R² should be between -inf and 1 (typically)
        self.assertLessEqual(metrics['r2'], 1.0)
        
        # MSE and RMSE should be non-negative
        self.assertGreaterEqual(metrics['mse'], 0.0)
        self.assertGreaterEqual(metrics['rmse'], 0.0)
        self.assertGreaterEqual(metrics['mae'], 0.0)
        
        # Check that results are stored
        self.assertIn('TestRegressor', evaluator.results)
        
        # Check that plots are created
        plot_files = list(Path(self.temp_dir).glob('*predictions*.png'))
        self.assertTrue(len(plot_files) > 0)
        
        residual_files = list(Path(self.temp_dir).glob('*residuals*.png'))
        self.assertTrue(len(residual_files) > 0)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_generative_evaluation(self):
        """Test generative model evaluation."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        results = evaluator.evaluate_generative_model(
            model=self.gen_model,
            dataloader=self.mock_dataloader,
            model_name="TestGenerative",
            num_samples=5
        )
        
        # Check results structure
        self.assertIn('model_name', results)
        self.assertIn('model_type', results)
        self.assertIn('metrics', results)
        self.assertIn('generated_samples', results)
        
        # Check model info
        self.assertEqual(results['model_name'], 'TestGenerative')
        self.assertEqual(results['model_type'], 'generative')
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('average_loss', metrics)
        self.assertIn('num_batches_evaluated', metrics)
        self.assertIsInstance(metrics['average_loss'], float)
        self.assertIsInstance(metrics['num_batches_evaluated'], int)
        
        # Check that results are stored
        self.assertIn('TestGenerative', evaluator.results)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_model_comparison(self):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        # Evaluate multiple classification models
        evaluator.evaluate_classification_model(
            model=self.clf_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="Model1"
        )
        
        # Create a second model for comparison
        clf_model2 = RandomForestClassifier(n_estimators=20, random_state=123)
        clf_model2.fit(self.X_train_clf, self.y_train_clf)
        
        evaluator.evaluate_classification_model(
            model=clf_model2,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="Model2"
        )
        
        # Compare models
        comparison_df = evaluator.compare_models(metric='accuracy')
        
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('Type', comparison_df.columns)
        self.assertIn('Accuracy', comparison_df.columns)
        
        # Check that comparison plot is created
        comparison_files = list(Path(self.temp_dir).glob('*model_comparison*.png'))
        self.assertTrue(len(comparison_files) > 0)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_report_generation(self):
        """Test report generation."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        # Evaluate models
        evaluator.evaluate_classification_model(
            model=self.clf_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="TestClassifier"
        )
        
        evaluator.evaluate_regression_model(
            model=self.reg_model,
            X_test=self.X_test_reg,
            y_test=self.y_test_reg,
            model_name="TestRegressor"
        )
        
        # Generate report
        report_content = evaluator.generate_report()
        
        # Check report content
        self.assertIn("# Model Evaluation Report", report_content)
        self.assertIn("## Summary", report_content)
        self.assertIn("Total models evaluated: 2", report_content)
        self.assertIn("### TestClassifier", report_content)
        self.assertIn("### TestRegressor", report_content)
        self.assertIn("## Best Performing Models", report_content)
        
        # Check that report file is created
        report_files = list(Path(self.temp_dir).glob('*evaluation_report*.md'))
        self.assertTrue(len(report_files) > 0)
        
        # Check report file content
        with open(report_files[0], 'r') as f:
            file_content = f.read()
            self.assertIn("# Model Evaluation Report", file_content)
            self.assertIn("TestClassifier", file_content)
            self.assertIn("TestRegressor", file_content)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_results_saving(self):
        """Test results saving to JSON."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        # Evaluate a model
        evaluator.evaluate_classification_model(
            model=self.clf_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="TestModel"
        )
        
        # Save results
        evaluator.save_results()
        
        # Check that JSON file is created
        json_files = list(Path(self.temp_dir).glob('*evaluation_results*.json'))
        self.assertTrue(len(json_files) > 0)
        
        # Load and verify JSON content
        with open(json_files[0], 'r') as f:
            saved_results = json.load(f)
        
        self.assertIn('TestModel', saved_results)
        self.assertEqual(saved_results['TestModel']['model_name'], 'TestModel')
        self.assertEqual(saved_results['TestModel']['model_type'], 'classification')
        self.assertIn('metrics', saved_results['TestModel'])
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_custom_filename_saving(self):
        """Test saving results with custom filename."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        # Evaluate a model
        evaluator.evaluate_classification_model(
            model=self.clf_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="TestModel"
        )
        
        # Save with custom filename
        custom_filename = "custom_results.json"
        evaluator.save_results(filename=custom_filename)
        
        # Check that file exists with custom name
        custom_file = Path(self.temp_dir) / custom_filename
        self.assertTrue(custom_file.exists())
        
        # Verify content
        with open(custom_file, 'r') as f:
            saved_results = json.load(f)
        
        self.assertIn('TestModel', saved_results)
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_pytorch_model_evaluation(self):
        """Test evaluation of PyTorch models."""
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        
        # Create simple PyTorch classification model
        class SimpleClassifier(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.linear = nn.Linear(input_size, num_classes)
            
            def forward(self, x):
                return self.linear(x)
        
        pytorch_model = SimpleClassifier(self.X_test_clf.shape[1], 3)
        
        # Evaluate PyTorch model
        results = evaluator.evaluate_classification_model(
            model=pytorch_model,
            X_test=self.X_test_clf,
            y_test=self.y_test_clf,
            model_name="PyTorchClassifier"
        )
        
        # Check that evaluation works
        self.assertIn('model_name', results)
        self.assertEqual(results['model_name'], 'PyTorchClassifier')
        self.assertIn('metrics', results)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    @unittest.skipIf(ModelEvaluator is None, "Evaluation modules not available")
    def test_error_handling(self):
        """Test error handling in evaluation."""
        evaluator = ModelEvaluator()
        
        # Test with invalid model (no predict method)
        class InvalidModel:
            pass
        
        invalid_model = InvalidModel()
        
        with self.assertRaises(ValueError):
            evaluator.evaluate_classification_model(
                model=invalid_model,
                X_test=np.random.randn(10, 5),
                y_test=np.random.randint(0, 2, 10),
                model_name="InvalidModel"
            )


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
