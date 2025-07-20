"""
Comprehensive evaluation script for ML/DS and Gen AI projects.
Evaluates models, generates reports, and provides insights.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import pandas as pd

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def evaluate_classification_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a classification model."""
        print(f"Evaluating classification model: {model_name}")
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        elif hasattr(model, 'forward'):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_test = torch.tensor(X_test, dtype=torch.float32)
                outputs = model(X_test)
                y_pred = torch.argmax(outputs, dim=1).numpy()
        else:
            raise ValueError("Model must have 'predict' or 'forward' method")
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'forward'):
            # PyTorch model
            with torch.no_grad():
                outputs = model(X_test)
                y_pred_proba = torch.softmax(outputs, dim=1).numpy()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Add AUC if probabilities available and binary/multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    # Multiclass
                    metrics['auc_roc_ovr'] = roc_auc_score(
                        y_test, y_pred_proba, multi_class='ovr'
                    )
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'model_type': 'classification',
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist(),
            'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        # Store results
        self.results[model_name] = results
        
        # Generate plots
        self._plot_confusion_matrix(cm, model_name, class_names)
        self._plot_classification_metrics(metrics, model_name)
        
        return results
    
    def evaluate_regression_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Evaluate a regression model."""
        print(f"Evaluating regression model: {model_name}")
        
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error
        )
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        elif hasattr(model, 'forward'):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_test = torch.tensor(X_test, dtype=torch.float32)
                y_pred = model(X_test).numpy()
        else:
            raise ValueError("Model must have 'predict' or 'forward' method")
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        results = {
            'model_name': model_name,
            'model_type': 'regression',
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist()
        }
        
        # Store results
        self.results[model_name] = results
        
        # Generate plots
        self._plot_regression_results(y_test, y_pred, model_name)
        self._plot_residuals(y_test, y_pred, model_name)
        
        return results
    
    def evaluate_generative_model(
        self,
        model: Any,
        dataloader: Any,
        model_name: str = "generative_model",
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Evaluate a generative model."""
        print(f"Evaluating generative model: {model_name}")
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Calculate perplexity for language models
        perplexities = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    # Handle different batch formats
                    if 'input_ids' in batch:
                        inputs = batch['input_ids']
                        targets = batch.get('labels', inputs)
                    else:
                        inputs = batch.get('inputs', list(batch.values())[0])
                        targets = batch.get('targets', inputs)
                else:
                    inputs = targets = batch
                
                try:
                    outputs = model(inputs)
                    
                    # Calculate loss
                    if hasattr(model, 'compute_loss'):
                        loss = model.compute_loss(outputs, targets)
                    else:
                        # Generic loss calculation
                        if outputs.dim() > 2:
                            # Language model case
                            loss = nn.CrossEntropyLoss()(
                                outputs.view(-1, outputs.size(-1)),
                                targets.view(-1)
                            )
                        else:
                            # Simple case
                            loss = nn.MSELoss()(outputs, targets.float())
                    
                    total_loss += loss.item()
                    
                    # Calculate perplexity for language models
                    if outputs.dim() > 2:  # Language model
                        perplexity = torch.exp(loss).item()
                        perplexities.append(perplexity)
                    
                    num_batches += 1
                    
                    if num_batches >= num_samples:
                        break
                        
                except Exception as e:
                    print(f"Error in batch evaluation: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_perplexity = np.mean(perplexities) if perplexities else None
        
        # Generate samples if possible
        generated_samples = []
        try:
            generated_samples = self._generate_samples(model, num_samples=10)
        except Exception as e:
            print(f"Could not generate samples: {e}")
        
        metrics = {
            'average_loss': avg_loss,
            'perplexity': avg_perplexity,
            'num_batches_evaluated': num_batches
        }
        
        results = {
            'model_name': model_name,
            'model_type': 'generative',
            'metrics': metrics,
            'generated_samples': generated_samples
        }
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def _generate_samples(self, model: Any, num_samples: int = 10) -> List[Any]:
        """Generate samples from a generative model."""
        samples = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                try:
                    if hasattr(model, 'sample'):
                        sample = model.sample()
                    elif hasattr(model, 'generate'):
                        sample = model.generate()
                    else:
                        # Try to generate with random input
                        noise = torch.randn(1, 100)  # Assuming noise dimension
                        sample = model(noise)
                    
                    samples.append(sample.cpu().numpy().tolist())
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    break
        
        return samples
    
    def _plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        model_name: str, 
        class_names: Optional[List[str]] = None
    ):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        if class_names:
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names
            )
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_confusion_matrix_{self.timestamp}.png')
        plt.close()
    
    def _plot_classification_metrics(self, metrics: Dict[str, float], model_name: str):
        """Plot classification metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Macro metrics
        macro_metrics = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision_macro'],
            'Recall': metrics['recall_macro'],
            'F1-Score': metrics['f1_macro']
        }
        
        ax1.bar(macro_metrics.keys(), macro_metrics.values())
        ax1.set_title(f'Classification Metrics (Macro) - {model_name}')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Weighted metrics
        weighted_metrics = {
            'Precision': metrics['precision_weighted'],
            'Recall': metrics['recall_weighted'],
            'F1-Score': metrics['f1_weighted']
        }
        
        ax2.bar(weighted_metrics.keys(), weighted_metrics.values())
        ax2.set_title(f'Classification Metrics (Weighted) - {model_name}')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_metrics_{self.timestamp}.png')
        plt.close()
    
    def _plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot regression results."""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted Values - {model_name}')
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_predictions_{self.timestamp}.png')
        plt.close()
    
    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot residuals."""
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Residuals vs Predicted - {model_name}')
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residuals Distribution - {model_name}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_residuals_{self.timestamp}.png')
        plt.close()
    
    def compare_models(self, metric: str = 'accuracy') -> pd.DataFrame:
        """Compare multiple models based on a specific metric."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            if metric in results.get('metrics', {}):
                comparison_data.append({
                    'Model': model_name,
                    'Type': results['model_type'],
                    metric.title(): results['metrics'][metric]
                })
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            # Plot comparison
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='Model', y=metric.title(), hue='Type')
            plt.title(f'Model Comparison - {metric.title()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'model_comparison_{metric}_{self.timestamp}.png')
            plt.close()
        
        return df
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = [
            "# Model Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Total models evaluated: {len(self.results)}",
            ""
        ]
        
        # Model summaries
        for model_name, results in self.results.items():
            report_lines.extend([
                f"### {model_name}",
                f"**Type:** {results['model_type']}",
                "",
                "**Metrics:**"
            ])
            
            for metric, value in results['metrics'].items():
                if isinstance(value, float):
                    report_lines.append(f"- {metric}: {value:.4f}")
                else:
                    report_lines.append(f"- {metric}: {value}")
            
            report_lines.append("")
        
        # Best performing models
        report_lines.extend([
            "## Best Performing Models",
            ""
        ])
        
        # Find best models by type and metric
        model_types = set(r['model_type'] for r in self.results.values())
        
        for model_type in model_types:
            type_models = {k: v for k, v in self.results.items() 
                          if v['model_type'] == model_type}
            
            if model_type == 'classification':
                best_metric = 'f1_macro'
            elif model_type == 'regression':
                best_metric = 'r2'
            else:
                best_metric = 'average_loss'
            
            if type_models:
                best_model = max(
                    type_models.items(),
                    key=lambda x: x[1]['metrics'].get(best_metric, -float('inf'))
                    if model_type != 'generative' 
                    else -x[1]['metrics'].get(best_metric, float('inf'))
                )
                
                report_lines.extend([
                    f"**Best {model_type} model:** {best_model[0]}",
                    f"- {best_metric}: {best_model[1]['metrics'].get(best_metric, 'N/A')}",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f'evaluation_report_{self.timestamp}.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to: {report_path}")
        return report_content
    
    def save_results(self, filename: Optional[str] = None):
        """Save evaluation results to JSON."""
        if filename is None:
            filename = f'evaluation_results_{self.timestamp}.json'
        
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = results.copy()
            
            # Handle numpy arrays
            for key, value in serializable_results[model_name].items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate ML/DS models')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--model-type', type=str, choices=['classification', 'regression', 'generative'],
                       help='Type of model to evaluate')
    parser.add_argument('--data-path', type=str, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    # Example evaluation (would need to be adapted based on actual models)
    print("Starting model evaluation...")
    
    # This is where you would load your actual models and data
    # For demonstration, we'll create dummy data
    
    if args.model_type == 'classification':
        # Dummy classification example
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        evaluator.evaluate_classification_model(
            model, X_test, y_test, 
            model_name="RandomForest",
            class_names=['Class_A', 'Class_B', 'Class_C']
        )
    
    elif args.model_type == 'regression':
        # Dummy regression example
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        evaluator.evaluate_regression_model(
            model, X_test, y_test,
            model_name="RandomForestRegressor"
        )
    
    # Generate report and save results
    evaluator.generate_report()
    evaluator.save_results()
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
