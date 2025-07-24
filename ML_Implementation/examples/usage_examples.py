"""
Comprehensive usage examples for the enhanced ML Implementation modules.

This file demonstrates how to use all the new advanced ML modules:
- Ensemble Methods
- Model Interpretability
- MLOps Toolkit
- Deep Learning Framework
"""

import sys
import os
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def ensemble_methods_example():
    """Demonstrate ensemble methods usage."""
    print("=" * 60)
    print("ENSEMBLE METHODS EXAMPLE")
    print("=" * 60)
    
    from ensemble_methods import RandomForestFromScratch, GradientBoostingFromScratch
    
    # Create dataset
    print("Creating classification dataset...")
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                             n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Example
    print("\n1. Random Forest from Scratch")
    print("-" * 40)
    
    rf = RandomForestFromScratch(n_trees=50, max_depth=10, min_samples_split=5)
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    # Make predictions
    rf_predictions = rf.predict(X_test)
    rf_accuracy = np.mean(rf_predictions == y_test)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Feature importance
    feature_importance = rf.feature_importance()
    print("Top 3 most important features:")
    top_features = np.argsort(feature_importance)[-3:][::-1]
    for i, feat_idx in enumerate(top_features):
        print(f"  {i+1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    # OOB Score
    oob_score = rf.oob_score()
    print(f"Out-of-Bag Score: {oob_score:.4f}")
    
    # Gradient Boosting Example
    print("\n2. Gradient Boosting from Scratch")
    print("-" * 40)
    
    gb = GradientBoostingFromScratch(n_estimators=50, learning_rate=0.1, max_depth=3)
    print("Training Gradient Boosting...")
    gb.fit(X_train, y_train)
    
    gb_predictions = gb.predict(X_test)
    gb_accuracy = np.mean((gb_predictions > 0.5).astype(int) == y_test)
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # Feature importance for GB
    gb_importance = gb.feature_importance()
    print("Top 3 most important features (GB):")
    top_gb_features = np.argsort(gb_importance)[-3:][::-1]
    for i, feat_idx in enumerate(top_gb_features):
        print(f"  {i+1}. Feature {feat_idx}: {gb_importance[feat_idx]:.4f}")


def interpretability_example():
    """Demonstrate model interpretability usage."""
    print("\n" + "=" * 60)
    print("MODEL INTERPRETABILITY EXAMPLE")
    print("=" * 60)
    
    try:
        from model_interpretability import (
            SHAPExplainer, LIMEExplainer, PermutationImportance, 
            PartialDependencePlotter, ModelInterpreter
        )
        from sklearn.ensemble import RandomForestClassifier
        
        # Create dataset
        print("Creating dataset for interpretation...")
        X, y = make_classification(n_samples=500, n_features=8, n_classes=2, 
                                 n_informative=6, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train a model
        print("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # 1. SHAP Explanations
        print("\n1. SHAP Explanations")
        print("-" * 30)
        try:
            shap_explainer = SHAPExplainer()
            shap_values = shap_explainer.explain_prediction(model, X_test[:5])
            print("SHAP values computed successfully!")
            print(f"Shape of SHAP values: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
        
        # 2. LIME Explanations
        print("\n2. LIME Explanations")
        print("-" * 30)
        try:
            lime_explainer = LIMEExplainer()
            lime_explainer.explain_instance(
                model, X_train, X_test[0], mode='classification'
            )
            print("LIME explanation generated successfully!")
        except Exception as e:
            print(f"LIME explanation failed: {e}")
        
        # 3. Permutation Importance
        print("\n3. Permutation Importance")
        print("-" * 30)
        perm_imp = PermutationImportance()
        importance = perm_imp.compute_importance(model, X_test, y_test, n_repeats=5)
        
        print("Permutation Importance scores:")
        for i, imp in enumerate(importance):
            print(f"  {feature_names[i]}: {imp:.4f}")
        
        # 4. Partial Dependence
        print("\n4. Partial Dependence")
        print("-" * 30)
        pdp = PartialDependencePlotter()
        
        # Single feature PDP
        pd_values, pd_grid = pdp.compute_partial_dependence(model, X_test, feature_idx=0)
        print(f"PDP computed for feature 0: {len(pd_values)} points")
        
        # Two feature PDP
        pd_values_2d, pd_grid_2d = pdp.compute_partial_dependence(model, X_test, feature_idx=[0, 1])
        print(f"2D PDP computed for features 0,1: {pd_values_2d.shape}")
        
        # 5. Comprehensive Interpretation
        print("\n5. Comprehensive Interpretation")
        print("-" * 30)
        interpreter = ModelInterpreter()
        results = interpreter.interpret_model(
            model, X_train, X_test, y_test,
            feature_names=feature_names,
            methods=['permutation', 'pdp']
        )
        
        print("Available interpretation results:")
        for key in results.keys():
            print(f"  - {key}")
    
    except ImportError as e:
        print(f"Interpretability modules not available: {e}")


def mlops_example():
    """Demonstrate MLOps toolkit usage."""
    print("\n" + "=" * 60)
    print("MLOPS TOOLKIT EXAMPLE")
    print("=" * 60)
    
    try:
        from mlops_toolkit import ModelRegistry, DataDriftDetector, ModelMonitor, ABTestFramework
        from sklearn.ensemble import RandomForestClassifier
        import tempfile
        import shutil
        
        # Create temporary directory for model registry
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create dataset and models
            print("Creating dataset and training models...")
            X, y = make_classification(n_samples=800, n_features=6, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train two different models
            model_a = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            model_a.fit(X_train, y_train)
            
            model_b = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=123)
            model_b.fit(X_train, y_train)
            
            # 1. Model Registry
            print("\n1. Model Registry")
            print("-" * 25)
            registry = ModelRegistry(base_path=temp_dir)
            
            # Register models
            model_a_id = registry.register_model(
                model_a, "random_forest_a", "1.0", 
                {"accuracy": model_a.score(X_test, y_test), "n_estimators": 50}
            )
            
            model_b_id = registry.register_model(
                model_b, "random_forest_b", "1.0",
                {"accuracy": model_b.score(X_test, y_test), "n_estimators": 100}
            )
            
            print(f"Registered Model A with ID: {model_a_id[:8]}...")
            print(f"Registered Model B with ID: {model_b_id[:8]}...")
            
            # List models
            models = registry.list_models()
            print(f"Total registered models: {len(models)}")
            
            # Load model
            loaded_model, metadata = registry.load_model(model_a_id)
            print(f"Loaded model accuracy: {metadata['accuracy']:.4f}")
            
            # 2. Data Drift Detection
            print("\n2. Data Drift Detection")
            print("-" * 30)
            drift_detector = DataDriftDetector()
            
            # Create slightly drifted data
            X_drifted = X_test + np.random.normal(0, 0.5, X_test.shape)
            
            # Detect drift using KS test
            drift_detected, p_values = drift_detector.detect_drift(X_train, X_test, method='ks')
            print(f"Drift detected (original data): {drift_detected}")
            
            drift_detected_2, p_values_2 = drift_detector.detect_drift(X_train, X_drifted, method='ks')
            print(f"Drift detected (drifted data): {drift_detected_2}")
            
            # Detect drift using PSI
            drift_detected_psi, psi_values = drift_detector.detect_drift(X_train, X_drifted, method='psi')
            print(f"Average PSI value: {np.mean(psi_values):.4f}")
            
            # 3. Model Monitoring
            print("\n3. Model Monitoring")
            print("-" * 25)
            monitor = ModelMonitor()
            
            # Simulate predictions and log them
            y_pred = model_a.predict(X_test)
            y_proba = model_a.predict_proba(X_test)[:, 1]
            
            for i in range(min(50, len(X_test))):
                monitor.log_prediction(
                    prediction_id=f"pred_{i}",
                    features=X_test[i].tolist(),
                    prediction=int(y_pred[i]),
                    confidence=float(y_proba[i]),
                    actual=int(y_test[i])
                )
            
            # Get performance metrics
            metrics = monitor.get_performance_metrics()
            if metrics:
                print("Current model performance:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Check for performance degradation
            degraded = monitor.detect_performance_degradation(
                metric='accuracy', threshold=0.7, window_size=20
            )
            print(f"Performance degradation detected: {degraded}")
            
            # 4. A/B Testing
            print("\n4. A/B Testing Framework")
            print("-" * 35)
            ab_test = ABTestFramework()
            
            # Create experiment
            exp_id = ab_test.create_experiment(
                name="model_comparison",
                model_a=model_a,
                model_b=model_b,
                traffic_split=0.5
            )
            print(f"Created A/B test experiment: {exp_id[:8]}...")
            
            # Simulate traffic routing and results
            correct_a, correct_b = 0, 0
            total_a, total_b = 0, 0
            
            for i in range(100):
                user_id = f"user_{i}"
                assigned_model = ab_test.route_traffic(exp_id, user_id)
                
                # Make prediction
                sample_idx = i % len(X_test)
                prediction = assigned_model.predict([X_test[sample_idx]])[0]
                actual = y_test[sample_idx]
                
                # Determine variant
                variant = "A" if assigned_model == model_a else "B"
                
                # Log result
                ab_test.log_result(exp_id, user_id, variant, prediction, 0.8, actual)
                
                # Track performance
                if variant == "A":
                    total_a += 1
                    if prediction == actual:
                        correct_a += 1
                else:
                    total_b += 1
                    if prediction == actual:
                        correct_b += 1
            
            print(f"Model A: {correct_a}/{total_a} correct ({correct_a/total_a:.3f})")
            print(f"Model B: {correct_b}/{total_b} correct ({correct_b/total_b:.3f})")
            
            # Analyze results
            analysis = ab_test.analyze_results(exp_id)
            print("\nA/B Test Analysis:")
            if "variant_A" in analysis:
                print(f"  Variant A conversion rate: {analysis['variant_A']['conversion_rate']:.4f}")
            if "variant_B" in analysis:
                print(f"  Variant B conversion rate: {analysis['variant_B']['conversion_rate']:.4f}")
            
            # Stop experiment
            ab_test.stop_experiment(exp_id)
            print("Experiment stopped.")
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except ImportError as e:
        print(f"MLOps modules not available: {e}")


def deep_learning_example():
    """Demonstrate deep learning framework usage."""
    print("\n" + "=" * 60)
    print("DEEP LEARNING FRAMEWORK EXAMPLE")
    print("=" * 60)
    
    try:
        from deep_learning_framework import (
            Tensor, MLP, Adam, Trainer
        )
        
        # 1. Basic Tensor Operations
        print("1. Tensor Operations and Automatic Differentiation")
        print("-" * 50)
        
        # Create tensors with gradient tracking
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        
        # Perform operations
        z = x * y + x ** 2
        print(f"z = x * y + x^2 = {z.data}")
        
        # Compute gradients
        z.backward()
        print(f"dz/dx = {x.grad}")  # Should be y + 2*x = 3 + 4 = 7
        print(f"dz/dy = {y.grad}")  # Should be x = 2
        
        # 2. Neural Network Training
        print("\n2. Neural Network Training")
        print("-" * 35)
        
        # Create simple binary classification dataset
        X, y = make_classification(n_samples=200, n_features=4, n_classes=2, 
                                 n_clusters_per_class=1, random_state=42)
        
        # Normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to lists (our framework expects lists)
        X_list = X.tolist()
        y_list = y.tolist()
        
        # Split data
        split_idx = int(0.8 * len(X_list))
        X_train, X_test = X_list[:split_idx], X_list[split_idx:]
        y_train, y_test = y_list[:split_idx], y_list[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create model
        model = MLP([4, 10, 6, 1])  # 4 input -> 10 hidden -> 6 hidden -> 1 output
        print(f"Model created with {len(model.parameters())} parameters")
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Create trainer
        trainer = Trainer(model, optimizer)
        
        # Training loop
        print("\nTraining model...")
        losses = []
        
        for epoch in range(20):
            try:
                loss = trainer.train_step(X_train, y_train)
                if loss is not None:
                    loss_val = loss.data if hasattr(loss, 'data') else loss
                    losses.append(loss_val)
                    
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")
            except Exception as e:
                print(f"Training step failed at epoch {epoch}: {e}")
                break
        
        # Test the model
        if len(losses) > 0:
            print(f"\nFinal training loss: {losses[-1]:.4f}")
            
            # Make predictions on test set
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(min(20, len(X_test))):  # Test on subset
                try:
                    output = model.forward(X_test[i])
                    if output:
                        prediction = 1 if output[0].data > 0.5 else 0
                        actual = y_test[i]
                        
                        if prediction == actual:
                            correct_predictions += 1
                        total_predictions += 1
                except Exception:
                    continue
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"Test accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # 3. Advanced Tensor Operations
        print("\n3. Advanced Tensor Operations")
        print("-" * 40)
        
        # Chain rule example
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        
        # Complex expression: (a + b) * (a - b) * exp(a)
        result = (a + b) * (a - b) * a.exp()
        print(f"Complex expression result: {result.data:.4f}")
        
        result.backward()
        print(f"da = {a.grad:.4f}")
        print(f"db = {b.grad:.4f}")
        
        print("\nDeep Learning Framework demonstration completed!")
    
    except ImportError as e:
        print(f"Deep Learning modules not available: {e}")
    except Exception as e:
        print(f"Deep Learning example failed: {e}")


def main():
    """Run all examples."""
    print("ADVANCED ML IMPLEMENTATION - USAGE EXAMPLES")
    print("=" * 80)
    print("This script demonstrates the usage of all enhanced ML modules.")
    print("=" * 80)
    
    # Run all examples
    ensemble_methods_example()
    interpretability_example()
    mlops_example()
    deep_learning_example()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nFor more detailed usage, check the individual module files:")
    print("- ensemble_methods.py")
    print("- model_interpretability.py") 
    print("- mlops_toolkit.py")
    print("- deep_learning_framework.py")


if __name__ == "__main__":
    main()
