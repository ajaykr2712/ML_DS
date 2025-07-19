"""
Advanced Evaluation Metrics for Machine Learning
===============================================

Comprehensive evaluation metrics for classification, regression, clustering,
and ranking tasks with advanced statistical analysis.

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Any
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    cohen_kappa_score, matthews_corrcoef
)
import warnings

warnings.filterwarnings('ignore')


class ClassificationMetrics:
    """Advanced classification evaluation metrics."""
    
    @staticmethod
    def comprehensive_evaluation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation with all relevant metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like  
            Predicted labels
        y_proba : array-like, optional
            Predicted probabilities
        class_names : list, optional
            Names of classes
        average : str, default='weighted'
            Averaging strategy for multi-class metrics
            
        Returns:
        --------
        metrics : dict
            Comprehensive metrics dictionary
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Per-class metrics
        if len(np.unique(y_true)) > 2:
            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Advanced metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix statistics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique, counts))
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if y_proba.ndim == 2:
                        y_proba_positive = y_proba[:, 1]
                    else:
                        y_proba_positive = y_proba
                    
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba_positive)
                    metrics['average_precision'] = average_precision_score(y_true, y_proba_positive)
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                    
                    # ROC curve data
                    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
                    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                    
                    # Precision-Recall curve data
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba_positive)
                    metrics['pr_curve'] = {'precision': precision_curve, 'recall': recall_curve}
                    
                else:
                    # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                    
            except Exception as e:
                print(f"Warning: Could not compute probability-based metrics: {e}")
        
        # Classification report
        if class_names:
            target_names = class_names
        else:
            target_names = [f'Class_{i}' for i in np.unique(y_true)]
        
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        
        return metrics
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate balanced accuracy (average of recall for each class).
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        balanced_acc : float
            Balanced accuracy score
        """
        return np.mean(recall_score(y_true, y_pred, average=None, zero_division=0))
    
    @staticmethod
    def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
        """
        Calculate top-k accuracy.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities
        k : int, default=5
            Number of top predictions to consider
            
        Returns:
        --------
        top_k_acc : float
            Top-k accuracy score
        """
        top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        return correct / len(y_true)
    
    @staticmethod
    def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate sensitivity (recall) and specificity for binary classification.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted binary labels
            
        Returns:
        --------
        sensitivity : float
            Sensitivity (True Positive Rate)
        specificity : float
            Specificity (True Negative Rate)
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return sensitivity, specificity


class RegressionMetrics:
    """Advanced regression evaluation metrics."""
    
    @staticmethod
    def comprehensive_evaluation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive regression evaluation with all relevant metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        sample_weight : array-like, optional
            Sample weights
            
        Returns:
        --------
        metrics : dict
            Comprehensive metrics dictionary
        """
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2_score'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Additional metrics
        metrics['mean_absolute_percentage_error'] = RegressionMetrics.mape(y_true, y_pred)
        metrics['symmetric_mape'] = RegressionMetrics.smape(y_true, y_pred)
        metrics['mean_absolute_scaled_error'] = RegressionMetrics.mase(y_true, y_pred)
        metrics['explained_variance'] = RegressionMetrics.explained_variance_score(y_true, y_pred)
        
        # Statistical metrics
        metrics['pearson_correlation'] = stats.pearsonr(y_true, y_pred)[0]
        metrics['spearman_correlation'] = stats.spearmanr(y_true, y_pred)[0]
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Percentage metrics
        metrics['mean_percentage_error'] = np.mean((y_true - y_pred) / y_true) * 100
        metrics['max_error'] = np.max(np.abs(residuals))
        
        return metrics
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(numerator / np.where(denominator != 0, denominator, 1)) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Scaled Error."""
        # Using naive forecast (previous value) as baseline
        naive_forecast = np.roll(y_true, 1)[1:]
        y_true_scaled = y_true[1:]
        mae_naive = np.mean(np.abs(y_true_scaled - naive_forecast))
        mae_pred = np.mean(np.abs(y_true - y_pred))
        return mae_pred / mae_naive if mae_naive != 0 else np.inf
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Explained variance score."""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)


class ClusteringMetrics:
    """Advanced clustering evaluation metrics."""
    
    @staticmethod
    def comprehensive_evaluation(
        X: np.ndarray,
        labels_pred: np.ndarray,
        labels_true: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        labels_pred : array-like
            Predicted cluster labels
        labels_true : array-like, optional
            True cluster labels (if available)
            
        Returns:
        --------
        metrics : dict
            Comprehensive clustering metrics
        """
        metrics = {}
        
        # Internal metrics (don't require true labels)
        if len(np.unique(labels_pred)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, labels_pred)
            metrics['calinski_harabasz_score'] = ClusteringMetrics.calinski_harabasz_score(X, labels_pred)
            metrics['davies_bouldin_score'] = ClusteringMetrics.davies_bouldin_score(X, labels_pred)
        
        # External metrics (require true labels)
        if labels_true is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(labels_true, labels_pred)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(labels_true, labels_pred)
            metrics['v_measure_score'] = ClusteringMetrics.v_measure_score(labels_true, labels_pred)
            metrics['fowlkes_mallows_score'] = ClusteringMetrics.fowlkes_mallows_score(labels_true, labels_pred)
        
        # Cluster statistics
        unique_labels, counts = np.unique(labels_pred, return_counts=True)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
        metrics['largest_cluster_size'] = np.max(counts)
        metrics['smallest_cluster_size'] = np.min(counts)
        metrics['cluster_size_std'] = np.std(counts)
        
        return metrics
    
    @staticmethod
    def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz Index (Variance Ratio Criterion)."""
        from sklearn.metrics import calinski_harabasz_score as ch_score
        return ch_score(X, labels)
    
    @staticmethod
    def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
        """Davies-Bouldin Index."""
        from sklearn.metrics import davies_bouldin_score as db_score
        return db_score(X, labels)
    
    @staticmethod
    def v_measure_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """V-measure clustering scoring."""
        from sklearn.metrics import v_measure_score
        return v_measure_score(labels_true, labels_pred)
    
    @staticmethod
    def fowlkes_mallows_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """Fowlkes-Mallows Index."""
        from sklearn.metrics import fowlkes_mallows_score
        return fowlkes_mallows_score(labels_true, labels_pred)


class StatisticalTests:
    """Statistical significance tests for model comparison."""
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, float]:
        """
        McNemar's test for comparing two classifiers.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred1 : array-like
            Predictions from first classifier
        y_pred2 : array-like
            Predictions from second classifier
            
        Returns:
        --------
        result : dict
            Test statistic and p-value
        """
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # McNemar's table
        # |          | Model 2 Correct | Model 2 Wrong |
        # |----------|-----------------|---------------|
        # | Model 1 Correct |      a      |      b        |
        # | Model 1 Wrong   |      c      |      d        |
        
        a = np.sum(correct1 & correct2)
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)
        d = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if b + c == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'message': 'No disagreement between models'}
        
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'contingency_table': {'a': a, 'b': b, 'c': c, 'd': d},
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """
        Paired t-test for comparing model performance across folds.
        
        Parameters:
        -----------
        scores1 : array-like
            Performance scores from first model
        scores2 : array-like
            Performance scores from second model
            
        Returns:
        --------
        result : dict
            Test statistic and p-value
        """
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'mean_difference': np.mean(scores1 - scores2),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def wilcoxon_signed_rank_test(scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Parameters:
        -----------
        scores1 : array-like
            Performance scores from first model
        scores2 : array-like
            Performance scores from second model
            
        Returns:
        --------
        result : dict
            Test statistic and p-value
        """
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


class VisualizationMetrics:
    """Advanced visualization utilities for metrics."""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot an enhanced confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Class names for labeling
        normalize : bool, default=False
            Whether to normalize the confusion matrix
        figsize : tuple, default=(8, 6)
            Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names or np.unique(y_true),
            yticklabels=class_names or np.unique(y_true)
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curves(
        y_true_list: List[np.ndarray],
        y_proba_list: List[np.ndarray],
        model_names: List[str],
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot ROC curves for multiple models.
        
        Parameters:
        -----------
        y_true_list : list of arrays
            True labels for each model
        y_proba_list : list of arrays
            Predicted probabilities for each model
        model_names : list of str
            Names of models
        figsize : tuple, default=(10, 8)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        for y_true, y_proba, name in zip(y_true_list, y_proba_list, model_names):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curves(
        y_true_list: List[np.ndarray],
        y_proba_list: List[np.ndarray],
        model_names: List[str],
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        
        Parameters:
        -----------
        y_true_list : list of arrays
            True labels for each model
        y_proba_list : list of arrays
            Predicted probabilities for each model
        model_names : list of str
            Names of models
        figsize : tuple, default=(10, 8)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        for y_true, y_proba, name in zip(y_true_list, y_proba_list, model_names):
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap_score = average_precision_score(y_true, y_proba)
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (12, 4)
    ) -> None:
        """
        Plot residual analysis for regression.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        figsize : tuple, default=(12, 4)
            Figure size
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[2].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Distribution of Residuals')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification, make_regression, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans
    
    print("="*60)
    print("Advanced Evaluation Metrics Demo")
    print("="*60)
    
    # Classification example
    print("\n1. Classification Metrics Demo")
    print("-" * 40)
    
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    clf_metrics = ClassificationMetrics.comprehensive_evaluation(
        y_test, y_pred, y_proba, class_names=['Class A', 'Class B', 'Class C']
    )
    
    print(f"Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"F1 Score: {clf_metrics['f1_score']:.4f}")
    print(f"Cohen's Kappa: {clf_metrics['cohen_kappa']:.4f}")
    print(f"Matthews Correlation: {clf_metrics['matthews_corrcoef']:.4f}")
    
    # Regression example
    print("\n2. Regression Metrics Demo")
    print("-" * 40)
    
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    reg_metrics = RegressionMetrics.comprehensive_evaluation(y_test, y_pred)
    
    print(f"R² Score: {reg_metrics['r2_score']:.4f}")
    print(f"RMSE: {reg_metrics['rmse']:.4f}")
    print(f"MAE: {reg_metrics['mae']:.4f}")
    print(f"MAPE: {reg_metrics['mean_absolute_percentage_error']:.4f}%")
    
    # Clustering example
    print("\n3. Clustering Metrics Demo")
    print("-" * 40)
    
    X_cluster, y_cluster = make_blobs(n_samples=300, centers=4, random_state=42)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels_pred = kmeans.fit_predict(X_cluster)
    
    cluster_metrics = ClusteringMetrics.comprehensive_evaluation(X_cluster, labels_pred, y_cluster)
    
    print(f"Silhouette Score: {cluster_metrics['silhouette_score']:.4f}")
    print(f"Adjusted Rand Score: {cluster_metrics['adjusted_rand_score']:.4f}")
    print(f"Number of Clusters: {cluster_metrics['n_clusters']}")
    
    print("\nDemo completed successfully!")
