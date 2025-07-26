"""
Advanced Evaluation Metrics for Machine Learning
Comprehensive collection of evaluation metrics beyond standard sklearn offerings
Enhanced with uncertainty quantification and fairness metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize


class UncertaintyMetrics:
    """Metrics for evaluating prediction uncertainty and calibration."""
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between accuracy and confidence.
        Lower ECE indicates better calibration.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for reliability diagram.
        
        Returns:
            bin_centers: Center of each confidence bin
            accuracies: Accuracy in each bin
            confidences: Average confidence in each bin
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = 0
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        
        return bin_centers, np.array(accuracies), np.array(confidences)
    
    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Brier Score for probability predictions.
        
        Brier Score measures the mean squared difference between
        predicted probabilities and actual outcomes.
        """
        return np.mean((y_prob - y_true) ** 2)
    
    @staticmethod
    def entropy_based_uncertainty(predictions: np.ndarray) -> np.ndarray:
        """
        Calculate entropy-based uncertainty for multi-class predictions.
        
        Args:
            predictions: Array of shape (n_samples, n_classes) with probabilities
        
        Returns:
            uncertainties: Array of uncertainties for each sample
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Calculate entropy
        uncertainties = -np.sum(predictions * np.log(predictions), axis=1)
        return uncertainties


class FairnessMetrics:
    """Metrics for evaluating algorithmic fairness."""
    
    @staticmethod
    def demographic_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                                    sensitive_features: np.ndarray) -> float:
        """
        Calculate demographic parity difference.
        
        Measures the difference in positive prediction rates between groups.
        """
        unique_groups = np.unique(sensitive_features)
        
        if len(unique_groups) != 2:
            raise ValueError("Currently only supports binary sensitive features")
        
        group_0_mask = sensitive_features == unique_groups[0]
        group_1_mask = sensitive_features == unique_groups[1]
        
        positive_rate_0 = y_pred[group_0_mask].mean()
        positive_rate_1 = y_pred[group_1_mask].mean()
        
        return abs(positive_rate_1 - positive_rate_0)
    
    @staticmethod
    def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                                sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate equalized odds difference.
        
        Measures the difference in TPR and FPR between groups.
        """
        unique_groups = np.unique(sensitive_features)
        
        if len(unique_groups) != 2:
            raise ValueError("Currently only supports binary sensitive features")
        
        results = {}
        
        for outcome in [0, 1]:
            outcome_mask = y_true == outcome
            
            group_0_mask = (sensitive_features == unique_groups[0]) & outcome_mask
            group_1_mask = (sensitive_features == unique_groups[1]) & outcome_mask
            
            if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
                rate_0 = y_pred[group_0_mask].mean()
                rate_1 = y_pred[group_1_mask].mean()
                
                rate_name = "tpr_difference" if outcome == 1 else "fpr_difference"
                results[rate_name] = abs(rate_1 - rate_0)
            else:
                rate_name = "tpr_difference" if outcome == 1 else "fpr_difference"
                results[rate_name] = 0.0
        
        return results
    
    @staticmethod
    def disparate_impact_ratio(y_pred: np.ndarray, sensitive_features: np.ndarray) -> float:
        """
        Calculate disparate impact ratio.
        
        Ratio of positive prediction rates between groups.
        A ratio close to 1.0 indicates fairness.
        """
        unique_groups = np.unique(sensitive_features)
        
        if len(unique_groups) != 2:
            raise ValueError("Currently only supports binary sensitive features")
        
        group_0_mask = sensitive_features == unique_groups[0]
        group_1_mask = sensitive_features == unique_groups[1]
        
        positive_rate_0 = y_pred[group_0_mask].mean()
        positive_rate_1 = y_pred[group_1_mask].mean()
        
        if positive_rate_0 == 0:
            return float('inf') if positive_rate_1 > 0 else 1.0
        
        return positive_rate_1 / positive_rate_0


class RobustnessMetrics:
    """Metrics for evaluating model robustness and stability."""
    
    @staticmethod
    def prediction_stability(model, X: np.ndarray, n_perturbations: int = 100, 
                           noise_std: float = 0.01) -> float:
        """
        Measure prediction stability under small perturbations.
        
        Args:
            model: Trained model with predict method
            X: Input data
            n_perturbations: Number of noise perturbations to apply
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            stability_score: Average agreement between original and perturbed predictions
        """
        original_predictions = model.predict(X)
        agreements = []
        
        for _ in range(n_perturbations):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, X.shape)
            X_perturbed = X + noise
            
            perturbed_predictions = model.predict(X_perturbed)
            agreement = (original_predictions == perturbed_predictions).mean()
            agreements.append(agreement)
        
        return np.mean(agreements)
    
    @staticmethod
    def adversarial_robustness_approx(model, X: np.ndarray, y: np.ndarray, 
                                    epsilon: float = 0.1) -> float:
        """
        Approximate adversarial robustness using random perturbations.
        
        Args:
            model: Trained model
            X: Input data
            y: True labels
            epsilon: Maximum perturbation magnitude
        
        Returns:
            robustness_score: Fraction of samples that maintain correct prediction
        """
        original_predictions = model.predict(X)
        correct_mask = original_predictions == y
        
        if correct_mask.sum() == 0:
            return 0.0
        
        # Generate random perturbations within epsilon ball
        perturbations = np.random.uniform(-epsilon, epsilon, X.shape)
        X_adversarial = np.clip(X + perturbations, 0, 1)  # Assume normalized inputs
        
        adversarial_predictions = model.predict(X_adversarial)
        
        # Check how many originally correct predictions remain correct
        robust_predictions = (adversarial_predictions == y) & correct_mask
        
        return robust_predictions.sum() / correct_mask.sum()


class AdvancedClassificationMetrics:
    """Advanced metrics for classification problems"""
    
    @staticmethod
    def matthews_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient (MCC)
        
        MCC is a correlation coefficient between observed and predicted binary classifications.
        It ranges from -1 to +1, where +1 represents perfect prediction, 0 no better than random,
        and -1 indicates total disagreement between prediction and observation.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate balanced accuracy
        
        Balanced accuracy is the average of recall obtained on each class.
        It's useful for imbalanced datasets.
        """
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        return np.mean(per_class_accuracy)
    
    @staticmethod
    def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
        """
        Calculate top-k accuracy
        
        Parameters:
        -----------
        y_true : ndarray
            True labels
        y_pred_proba : ndarray
            Predicted probabilities for each class
        k : int, default=5
            Number of top predictions to consider
        
        Returns:
        --------
        float
            Top-k accuracy score
        """
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    @staticmethod
    def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Cohen's Kappa score
        
        Kappa measures inter-rater agreement for categorical items.
        It's generally thought to be a more robust measure than simple percent agreement.
        """
        cm = confusion_matrix(y_true, y_pred)
        # n_classes = cm.shape[0]  # Not used in this calculation
        n_samples = np.sum(cm)
        
        # Observed agreement
        po = np.trace(cm) / n_samples
        
        # Expected agreement
        pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n_samples ** 2)
        
        if pe == 1:
            return 0.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa
    
    @staticmethod
    def macro_averaged_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate macro-averaged recall"""
        cm = confusion_matrix(y_true, y_pred)
        recalls = cm.diagonal() / cm.sum(axis=1)
        return np.mean(recalls)
    
    @staticmethod
    def micro_averaged_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate micro-averaged precision"""
        cm = confusion_matrix(y_true, y_pred)
        tp = cm.diagonal().sum()
        fp = cm.sum() - cm.diagonal().sum() - (cm.sum(axis=0) - cm.diagonal()).sum()
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def geometric_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate geometric mean of recall scores (G-mean)
        
        G-mean is useful for imbalanced datasets as it considers both sensitivity and specificity.
        """
        cm = confusion_matrix(y_true, y_pred)
        recalls = cm.diagonal() / cm.sum(axis=1)
        
        # Handle zero recalls
        recalls[recalls == 0] = 1e-10
        
        return np.power(np.prod(recalls), 1.0 / len(recalls))
    
    @staticmethod
    def class_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate comprehensive class-wise metrics
        
        Parameters:
        -----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted labels
        class_names : list, optional
            Names of classes
        
        Returns:
        --------
        DataFrame
            Class-wise metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(n_classes)]
        
        metrics = []
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append({
                'Class': class_names[i],
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1_Score': f1,
                'Support': cm[i, :].sum()
            })
        
        return pd.DataFrame(metrics)


class AdvancedRegressionMetrics:
    """Advanced metrics for regression problems"""
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        MAPE is a scale-independent measure of prediction accuracy.
        """
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
        
        SMAPE is a variation of MAPE that is symmetric and bounded between 0 and 200%.
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    @staticmethod
    def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_train: np.ndarray) -> float:
        """
        Calculate Mean Absolute Scaled Error (MASE)
        
        MASE is a scale-free error metric that compares the forecast error
        to the error of a naive forecast.
        """
        # Calculate naive forecast error (seasonal naive)
        naive_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        
        if naive_error == 0:
            return np.inf
        
        mae = np.mean(np.abs(y_true - y_pred))
        return mae / naive_error
    
    @staticmethod
    def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Normalized Root Mean Squared Error (NRMSE)
        
        NRMSE normalizes RMSE by the range of the true values.
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        y_range = np.max(y_true) - np.min(y_true)
        
        if y_range == 0:
            return 0.0
        
        return rmse / y_range
    
    @staticmethod
    def coefficient_of_determination_adjusted(y_true: np.ndarray, y_pred: np.ndarray, 
                                            n_features: int) -> float:
        """
        Calculate Adjusted R-squared
        
        Adjusted R-squared penalizes the addition of irrelevant features.
        """
        n_samples = len(y_true)
        
        # Regular R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R-squared
        if n_samples - n_features - 1 <= 0:
            return r2
        
        adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        return adj_r2
    
    @staticmethod
    def index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Willmott's Index of Agreement
        
        The index of agreement is a standardized measure of the degree of model prediction error.
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        
        y_mean = np.mean(y_true)
        denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean)) ** 2)
        
        if denominator == 0:
            return 1.0
        
        return 1 - (numerator / denominator)
    
    @staticmethod
    def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Nash-Sutcliffe Efficiency (NSE)
        
        NSE is a normalized statistic that determines the relative magnitude
        of the residual variance compared to the measured data variance.
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if denominator == 0:
            return 1.0
        
        return 1 - (numerator / denominator)


class RobustMetrics:
    """Robust evaluation metrics less sensitive to outliers"""
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error (MedAE)"""
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def trimmed_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                   trim_percent: float = 0.1) -> float:
        """
        Calculate Trimmed Mean Absolute Error
        
        Parameters:
        -----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        trim_percent : float, default=0.1
            Percentage of extreme values to trim from each end
        
        Returns:
        --------
        float
            Trimmed MAE
        """
        errors = np.abs(y_true - y_pred)
        n_trim = int(len(errors) * trim_percent)
        
        if n_trim == 0:
            return np.mean(errors)
        
        sorted_errors = np.sort(errors)
        trimmed_errors = sorted_errors[n_trim:-n_trim]
        
        return np.mean(trimmed_errors)
    
    @staticmethod
    def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """
        Calculate Huber Loss
        
        Huber loss is less sensitive to outliers than squared error loss.
        
        Parameters:
        -----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        delta : float, default=1.0
            Threshold for switching between quadratic and linear loss
        
        Returns:
        --------
        float
            Huber loss
        """
        errors = y_true - y_pred
        is_small_error = np.abs(errors) <= delta
        
        squared_loss = 0.5 * errors ** 2
        linear_loss = delta * (np.abs(errors) - 0.5 * delta)
        
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))


class TimeSeriesMetrics:
    """Specialized metrics for time series forecasting"""
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Directional Accuracy
        
        Measures the percentage of times the forecast correctly predicts
        the direction of change.
        """
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return np.mean(true_direction == pred_direction)
    
    @staticmethod
    def theil_inequality_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Theil's Inequality Coefficient
        
        Theil's U ranges from 0 to 1, where 0 indicates perfect forecast
        and 1 indicates forecast no better than naive forecast.
        """
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Forecast Bias
        
        Positive values indicate over-forecasting, negative values indicate under-forecasting.
        """
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def forecast_accuracy_measures(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive time series accuracy measures
        
        Returns:
        --------
        dict
            Dictionary containing various time series metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
        metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        
        # Percentage errors
        try:
            metrics['MAPE'] = TimeSeriesMetrics.mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            metrics['MAPE'] = np.inf
        
        # Directional accuracy
        metrics['DA'] = TimeSeriesMetrics.directional_accuracy(y_true, y_pred)
        
        # Theil's U
        metrics['Theil_U'] = TimeSeriesMetrics.theil_inequality_coefficient(y_true, y_pred)
        
        # Bias
        metrics['Bias'] = TimeSeriesMetrics.forecast_bias(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE for time series (handles zero values better)"""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class ProbabilisticMetrics:
    """Metrics for probabilistic predictions"""
    
    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate Brier Score
        
        Brier score measures the accuracy of probabilistic predictions for binary outcomes.
        Lower scores indicate better predictions.
        """
        return np.mean((y_pred_proba - y_true) ** 2)
    
    @staticmethod
    def log_loss_stable(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       epsilon: float = 1e-15) -> float:
        """
        Calculate stable log loss
        
        Uses clipping to prevent overflow/underflow issues.
        """
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    @staticmethod
    def calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        ECE measures how well the predicted probabilities match the actual outcomes.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class MetricsVisualizer:
    """Visualization utilities for evaluation metrics"""
    
    @staticmethod
    def plot_confusion_matrix_advanced(y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     normalize: bool = False,
                                     figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot advanced confusion matrix with detailed annotations"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set ticks and labels
        if class_names:
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45)
            ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiclass_roc(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot ROC curves for multiclass classification"""
        n_classes = y_pred_proba.shape[1]
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            class_name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr[i], tpr[i], linewidth=2,
                   label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multiclass ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   figsize: Tuple[int, int] = (8, 6)) -> None:
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, 
                label=f'Average Precision = {average_precision:.2f}')
        plt.fill_between(recall, precision, alpha=0.2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Plot calibration curve (reliability diagram)"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=figsize)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2,
                label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    print("Testing Advanced Evaluation Metrics")
    print("=" * 50)
    
    # Test classification metrics
    print("\nTesting Classification Metrics:")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                      n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Advanced classification metrics
    clf_metrics = AdvancedClassificationMetrics()
    
    mcc = clf_metrics.matthews_correlation_coefficient(y_test, y_pred)
    balanced_acc = clf_metrics.balanced_accuracy(y_test, y_pred)
    kappa = clf_metrics.cohen_kappa_score(y_test, y_pred)
    gmean = clf_metrics.geometric_mean_score(y_test, y_pred)
    top_2_acc = clf_metrics.top_k_accuracy(y_test, y_pred_proba, k=2)
    
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Geometric Mean: {gmean:.4f}")
    print(f"Top-2 Accuracy: {top_2_acc:.4f}")
    
    # Class-wise metrics
    class_metrics = clf_metrics.class_wise_metrics(y_test, y_pred)
    print("\nClass-wise Metrics:")
    print(class_metrics)
    
    # Test regression metrics
    print("\n" + "="*50)
    print("Testing Regression Metrics:")
    
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    
    # Advanced regression metrics
    reg_metrics = AdvancedRegressionMetrics()
    
    mape = reg_metrics.mean_absolute_percentage_error(y_test_reg, y_pred_reg)
    smape = reg_metrics.symmetric_mean_absolute_percentage_error(y_test_reg, y_pred_reg)
    nrmse = reg_metrics.normalized_root_mean_squared_error(y_test_reg, y_pred_reg)
    adj_r2 = reg_metrics.coefficient_of_determination_adjusted(y_test_reg, y_pred_reg, X_test_reg.shape[1])
    nse = reg_metrics.nash_sutcliffe_efficiency(y_test_reg, y_pred_reg)
    
    print(f"MAPE: {mape:.4f}%")
    print(f"SMAPE: {smape:.4f}%")
    print(f"NRMSE: {nrmse:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    print(f"Nash-Sutcliffe Efficiency: {nse:.4f}")
    
    # Test robust metrics
    print("\nTesting Robust Metrics:")
    robust_metrics = RobustMetrics()
    
    medae = robust_metrics.median_absolute_error(y_test_reg, y_pred_reg)
    trimmed_mae = robust_metrics.trimmed_mean_absolute_error(y_test_reg, y_pred_reg)
    huber = robust_metrics.huber_loss(y_test_reg, y_pred_reg)
    
    print(f"Median Absolute Error: {medae:.4f}")
    print(f"Trimmed MAE: {trimmed_mae:.4f}")
    print(f"Huber Loss: {huber:.4f}")
    
    # Test probabilistic metrics
    print("\nTesting Probabilistic Metrics:")
    prob_metrics = ProbabilisticMetrics()
    
    # For binary classification
    X_bin, y_bin = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.3, random_state=42
    )
    
    clf_bin = RandomForestClassifier(random_state=42)
    clf_bin.fit(X_train_bin, y_train_bin)
    y_pred_proba_bin = clf_bin.predict_proba(X_test_bin)[:, 1]
    
    brier = prob_metrics.brier_score(y_test_bin, y_pred_proba_bin)
    log_loss = prob_metrics.log_loss_stable(y_test_bin, y_pred_proba_bin)
    cal_error = prob_metrics.calibration_error(y_test_bin, y_pred_proba_bin)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {log_loss:.4f}")
    print(f"Calibration Error: {cal_error:.4f}")
    
    print("\nAll advanced metrics tests completed successfully!")