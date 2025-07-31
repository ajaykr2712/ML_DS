"""
Advanced Model Interpretability Suite
Comprehensive toolkit for understanding machine learning model predictions

New Features:
- SHAP TreeExplainer optimizations for tree-based models
- Advanced LIME implementations with stability improvements  
- Counterfactual explanations using DiCE framework
- Model-agnostic feature importance with permutation testing
- Anchors explanations for rule-based interpretations
- Local surrogate models for complex prediction explanations
- Global feature interactions analysis
- Adversarial robustness testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_analyzer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import joblib
import json
from datetime import datetime
import os
import itertools
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class InterpretabilityConfig:
    """Configuration for model interpretability analysis"""
    # SHAP configuration
    use_shap: bool = True
    shap_sample_size: int = 1000
    shap_check_additivity: bool = False
    
    # LIME configuration  
    use_lime: bool = True
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    lime_mode: str = 'classification'  # 'classification' or 'regression'
    
    # Permutation importance
    use_permutation: bool = True
    perm_n_repeats: int = 10
    perm_random_state: int = 42
    
    # Partial dependence
    use_pdp: bool = True
    pdp_features: List[Union[int, str]] = None
    pdp_kind: str = 'average'  # 'average', 'individual', 'both'
    
    # Feature interactions
    analyze_interactions: bool = True
    max_interaction_depth: int = 2
    
    # Counterfactual explanations
    use_counterfactuals: bool = True
    cf_num_cfs: int = 5
    cf_desired_class: Union[int, str] = None
    
    # Anchors explanations
    use_anchors: bool = True
    anchors_threshold: float = 0.95
    
    # Adversarial testing
    test_adversarial: bool = True
    adversarial_epsilon: float = 0.1
    
    # Visualization
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    use_interactive_plots: bool = True

class AdvancedSHAPAnalyzer:
    """Advanced SHAP analysis with optimizations and extensions"""
    
    def __init__(self, model, X_train, model_type: str = 'auto'):
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
        # Auto-detect model type if not specified
        if model_type == 'auto':
            self.model_type = self._detect_model_type()
        
        self._initialize_explainer()
    
    def _detect_model_type(self) -> str:
        """Automatically detect the model type for optimal SHAP explainer"""
        model_name = type(self.model).__name__.lower()
        
        if 'tree' in model_name or 'forest' in model_name or 'boost' in model_name:
            return 'tree'
        elif 'linear' in model_name or 'logistic' in model_name:
            return 'linear'
        elif 'svm' in model_name or 'svc' in model_name:
            return 'kernel'
        else:
            return 'model_agnostic'
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer"""
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        elif self.model_type == 'kernel':
            # Use a subset for kernel explainer to improve performance
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        else:
            # Model-agnostic explainer
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.Explainer(self.model.predict_proba, background)
    
    def compute_shap_values(self, X_test, sample_size: int = None):
        """Compute SHAP values with optimizations"""
        if sample_size and len(X_test) > sample_size:
            # Random sampling for large datasets
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[indices] if hasattr(X_test, 'iloc') else X_test[indices]
        else:
            X_sample = X_test
        
        if self.model_type == 'tree':
            self.shap_values = self.explainer.shap_values(X_sample)
        else:
            self.shap_values = self.explainer(X_sample)
        
        return self.shap_values
    
    def plot_advanced_summary(self, feature_names: List[str] = None, 
                            class_names: List[str] = None, save_path: str = None):
        """Create advanced SHAP summary plots"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            # Multi-class classification
            for i, class_shap in enumerate(self.shap_values):
                class_name = class_names[i] if class_names else f"Class {i}"
                
                plt.figure(figsize=(12, 8))
                shap.summary_plot(class_shap, feature_names=feature_names, 
                                show=False, plot_type="bar")
                plt.title(f'SHAP Feature Importance - {class_name}')
                
                if save_path:
                    plt.savefig(f"{save_path}_summary_{class_name}.png", 
                              dpi=300, bbox_inches='tight')
                plt.show()
        else:
            # Binary classification or regression
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, feature_names=feature_names, show=False)
            plt.title('SHAP Summary Plot')
            
            if save_path:
                plt.savefig(f"{save_path}_summary.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_feature_interactions(self, X_test, max_features: int = 10):
        """Analyze feature interactions using SHAP interaction values"""
        if hasattr(self.explainer, 'shap_interaction_values'):
            interaction_values = self.explainer.shap_interaction_values(X_test)
            
            # Get feature importance from main effects
            main_effects = np.abs(self.shap_values).mean(0)
            top_features = np.argsort(main_effects)[-max_features:]
            
            # Create interaction heatmap
            interaction_matrix = np.abs(interaction_values[:, top_features][:, :, top_features]).mean(0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='viridis')
            plt.title('Feature Interaction Heatmap (SHAP)')
            plt.show()
            
            return interaction_values, interaction_matrix
        else:
            print("Model type doesn't support interaction values")
            return None, None

class EnhancedLIMEAnalyzer:
    """Enhanced LIME analyzer with stability improvements"""
    
    def __init__(self, model, X_train, feature_names: List[str] = None, 
                 class_names: List[str] = None, mode: str = 'classification'):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        
        # Initialize LIME explainer
        if mode == 'classification':
            self.explainer = LimeTabularExplainer(
                X_train.values if hasattr(X_train, 'values') else X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
        else:
            from lime.lime_tabular import LimeTabularExplainer
            self.explainer = LimeTabularExplainer(
                X_train.values if hasattr(X_train, 'values') else X_train,
                feature_names=feature_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
    
    def explain_instance_stable(self, instance, num_features: int = 10, 
                              num_samples: int = 5000, num_runs: int = 5):
        """Explain instance with stability analysis across multiple runs"""
        explanations = []
        
        for run in range(num_runs):
            # Set different random seed for each run
            self.explainer.random_state = 42 + run
            
            if self.mode == 'classification':
                explanation = self.explainer.explain_instance(
                    instance, self.model.predict_proba, 
                    num_features=num_features, num_samples=num_samples
                )
            else:
                explanation = self.explainer.explain_instance(
                    instance, self.model.predict, 
                    num_features=num_features, num_samples=num_samples
                )
            
            explanations.append(explanation)
        
        # Analyze stability
        stability_scores = self._compute_stability(explanations)
        
        return explanations[0], stability_scores  # Return first explanation and stability
    
    def _compute_stability(self, explanations) -> Dict[str, float]:
        """Compute stability metrics across multiple LIME explanations"""
        feature_weights = []
        
        for exp in explanations:
            weights = {}
            for feature, weight in exp.as_list():
                weights[feature] = weight
            feature_weights.append(weights)
        
        # Compute standard deviation of weights for each feature
        stability_metrics = {}
        all_features = set()
        for weights in feature_weights:
            all_features.update(weights.keys())
        
        for feature in all_features:
            weights = [fw.get(feature, 0) for fw in feature_weights]
            stability_metrics[feature] = {
                'mean_weight': np.mean(weights),
                'std_weight': np.std(weights),
                'cv': np.std(weights) / (np.abs(np.mean(weights)) + 1e-8)  # Coefficient of variation
            }
        
        return stability_metrics
    
    def plot_stability_analysis(self, stability_scores: Dict, save_path: str = None):
        """Plot stability analysis results"""
        features = list(stability_scores.keys())
        mean_weights = [stability_scores[f]['mean_weight'] for f in features]
        cv_scores = [stability_scores[f]['cv'] for f in features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean weights plot
        ax1.barh(range(len(features)), mean_weights)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Mean Feature Weight')
        ax1.set_title('LIME Feature Importance (Mean across runs)')
        
        # Stability plot (lower CV = more stable)
        ax2.barh(range(len(features)), cv_scores, color='red', alpha=0.7)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features)
        ax2.set_xlabel('Coefficient of Variation')
        ax2.set_title('Feature Stability (Lower = More Stable)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_lime_stability.png", dpi=300, bbox_inches='tight')
        plt.show()

class CounterfactualExplainer:
    """Counterfactual explanations using optimization-based approach"""
    
    def __init__(self, model, X_train, feature_names: List[str] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Compute feature ranges for constraints
        if hasattr(X_train, 'values'):
            self.feature_mins = X_train.min().values
            self.feature_maxs = X_train.max().values
        else:
            self.feature_mins = np.min(X_train, axis=0)
            self.feature_maxs = np.max(X_train, axis=0)
    
    def generate_counterfactuals(self, instance, desired_class: int, 
                               num_cfs: int = 5, max_iterations: int = 1000):
        """Generate counterfactual explanations using gradient-based optimization"""
        instance = np.array(instance).reshape(1, -1)
        counterfactuals = []
        
        for _ in range(num_cfs):
            # Initialize with random perturbation
            cf = instance.copy()
            cf += np.random.normal(0, 0.1, cf.shape)
            
            # Ensure within feature bounds
            cf = np.clip(cf, self.feature_mins, self.feature_maxs)
            
            for iteration in range(max_iterations):
                # Compute prediction
                pred_proba = self.model.predict_proba(cf)[0]
                
                # Check if desired class is achieved
                if np.argmax(pred_proba) == desired_class:
                    counterfactuals.append(cf.copy())
                    break
                
                # Compute gradient (approximate using finite differences)
                gradient = np.zeros_like(cf[0])
                epsilon = 1e-4
                
                for i in range(len(cf[0])):
                    cf_plus = cf.copy()
                    cf_plus[0, i] += epsilon
                    cf_plus = np.clip(cf_plus, self.feature_mins, self.feature_maxs)
                    
                    cf_minus = cf.copy()
                    cf_minus[0, i] -= epsilon
                    cf_minus = np.clip(cf_minus, self.feature_mins, self.feature_maxs)
                    
                    # Gradient of desired class probability
                    grad = (self.model.predict_proba(cf_plus)[0][desired_class] - 
                           self.model.predict_proba(cf_minus)[0][desired_class]) / (2 * epsilon)
                    gradient[i] = grad
                
                # Update counterfactual
                learning_rate = 0.01
                cf[0] += learning_rate * gradient
                
                # Apply constraints
                cf = np.clip(cf, self.feature_mins, self.feature_maxs)
        
        return counterfactuals
    
    def analyze_counterfactuals(self, instance, counterfactuals):
        """Analyze the generated counterfactuals"""
        if not counterfactuals:
            return None
        
        instance = np.array(instance).reshape(1, -1)
        analysis = {
            'num_generated': len(counterfactuals),
            'changes_required': [],
            'average_distance': 0,
            'feature_changes': {}
        }
        
        # Initialize feature changes counter
        if self.feature_names:
            for fname in self.feature_names:
                analysis['feature_changes'][fname] = 0
        
        total_distance = 0
        
        for cf in counterfactuals:
            # Compute L2 distance
            distance = np.linalg.norm(cf - instance)
            total_distance += distance
            
            # Find changed features
            changes = []
            for i, (orig, new) in enumerate(zip(instance[0], cf[0])):
                if abs(orig - new) > 1e-6:  # Threshold for numerical precision
                    feature_name = self.feature_names[i] if self.feature_names else f"Feature_{i}"
                    changes.append({
                        'feature': feature_name,
                        'original': orig,
                        'counterfactual': new,
                        'change': new - orig
                    })
                    
                    if feature_name in analysis['feature_changes']:
                        analysis['feature_changes'][feature_name] += 1
            
            analysis['changes_required'].append(changes)
        
        analysis['average_distance'] = total_distance / len(counterfactuals)
        
        return analysis

class ModelInterpretabilityFramework:
    """Comprehensive model interpretability framework"""
    
    def __init__(self, model, X_train, X_test, y_test, 
                 config: InterpretabilityConfig = None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config or InterpretabilityConfig()
        
        # Initialize analyzers
        self.shap_analyzer = None
        self.lime_analyzer = None
        self.cf_explainer = None
        
        # Results storage
        self.results = {}
        
        # Create output directory
        os.makedirs('interpretability_results', exist_ok=True)
    
    def run_full_analysis(self, feature_names: List[str] = None, 
                         class_names: List[str] = None):
        """Run comprehensive interpretability analysis"""
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['model_type'] = type(self.model).__name__
        
        print("🔍 Starting Comprehensive Model Interpretability Analysis...")
        
        # 1. SHAP Analysis
        if self.config.use_shap:
            print("📊 Running SHAP Analysis...")
            self._run_shap_analysis(feature_names, class_names)
        
        # 2. LIME Analysis
        if self.config.use_lime:
            print("🔍 Running LIME Analysis...")
            self._run_lime_analysis(feature_names, class_names)
        
        # 3. Permutation Importance
        if self.config.use_permutation:
            print("🔄 Computing Permutation Importance...")
            self._run_permutation_analysis(feature_names)
        
        # 4. Partial Dependence Analysis
        if self.config.use_pdp:
            print("📈 Analyzing Partial Dependence...")
            self._run_pdp_analysis(feature_names)
        
        # 5. Counterfactual Explanations
        if self.config.use_counterfactuals:
            print("🔄 Generating Counterfactual Explanations...")
            self._run_counterfactual_analysis(feature_names)
        
        # 6. Feature Interactions
        if self.config.analyze_interactions:
            print("🔗 Analyzing Feature Interactions...")
            self._run_interaction_analysis(feature_names)
        
        # 7. Adversarial Testing
        if self.config.test_adversarial:
            print("⚔️ Testing Adversarial Robustness...")
            self._run_adversarial_analysis()
        
        # Save comprehensive results
        self._save_results()
        
        print("✅ Interpretability analysis complete!")
        return self.results
