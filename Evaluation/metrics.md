# 📊 ML Metrics Quick Reference Guide

## 🎯 Classification Metrics

### Basic Metrics

#### **Accuracy**
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Use**: Balanced datasets
- **Avoid**: Imbalanced datasets

#### **Precision**
```python
precision = TP / (TP + FP)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Meaning**: "Of all positive predictions, how many were correct?"
- **Use**: When false positives are costly

#### **Recall (Sensitivity)**
```python
recall = TP / (TP + FN)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Meaning**: "Of all actual positives, how many were found?"
- **Use**: When false negatives are costly

#### **F1-Score**
```python
f1 = 2 * (precision * recall) / (precision + recall)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Use**: Balance between precision and recall

#### **Specificity**
```python
specificity = TN / (TN + FP)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Meaning**: "Of all actual negatives, how many were correctly identified?"

### Advanced Classification Metrics

#### **Cohen's Kappa**
```python
kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
```
- **Range**: [-1, 1]
- **Interpretation**:
  - < 0: Less than chance agreement
  - 0.01-0.20: Slight agreement
  - 0.21-0.40: Fair agreement
  - 0.41-0.60: Moderate agreement
  - 0.61-0.80: Substantial agreement
  - 0.81-1.00: Almost perfect agreement

#### **Matthews Correlation Coefficient (MCC)**
```python
mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
```
- **Range**: [-1, 1]
- **Best**: 1.0
- **Use**: Balanced measure for all four confusion matrix categories

#### **ROC-AUC**
```python
# Area Under the Receiver Operating Characteristic Curve
auc = ∫ TPR d(FPR)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Interpretation**:
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - 0.6-0.7: Poor
  - 0.5-0.6: Fail

#### **Precision-Recall AUC**
```python
# Area Under the Precision-Recall Curve
pr_auc = ∫ Precision d(Recall)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Use**: Imbalanced datasets (better than ROC-AUC)

#### **Log Loss (Cross-Entropy)**
```python
log_loss = -1/N * Σ[y*log(p) + (1-y)*log(1-p)]
```
- **Range**: [0, ∞)
- **Best**: 0.0
- **Use**: Probability calibration assessment

---

## 📈 Regression Metrics

### Error-Based Metrics

#### **Mean Absolute Error (MAE)**
```python
mae = Σ|y_true - y_pred| / n
```
- **Range**: [0, ∞)
- **Best**: 0.0
- **Units**: Same as target variable
- **Robust**: To outliers

#### **Mean Squared Error (MSE)**
```python
mse = Σ(y_true - y_pred)² / n
```
- **Range**: [0, ∞)
- **Best**: 0.0
- **Units**: Squared target units
- **Sensitive**: To outliers

#### **Root Mean Squared Error (RMSE)**
```python
rmse = sqrt(Σ(y_true - y_pred)² / n)
```
- **Range**: [0, ∞)
- **Best**: 0.0
- **Units**: Same as target variable
- **Common**: Most widely used

#### **Mean Absolute Percentage Error (MAPE)**
```python
mape = 100 * Σ|y_true - y_pred| / |y_true| / n
```
- **Range**: [0, ∞)
- **Best**: 0.0
- **Units**: Percentage
- **Issue**: Division by zero when y_true = 0

#### **Symmetric MAPE (SMAPE)**
```python
smape = 100 * Σ|y_true - y_pred| / ((|y_true| + |y_pred|)/2) / n
```
- **Range**: [0, 200]
- **Best**: 0.0
- **Advantage**: Symmetric, handles zeros better

### Correlation-Based Metrics

#### **R-squared (Coefficient of Determination)**
```python
r2 = 1 - SS_res / SS_tot
where SS_res = Σ(y_true - y_pred)²
      SS_tot = Σ(y_true - y_mean)²
```
- **Range**: (-∞, 1]
- **Best**: 1.0
- **Interpretation**: Proportion of variance explained
- **Note**: Can be negative if model is worse than mean

#### **Adjusted R-squared**
```python
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
```
- **Range**: (-∞, 1]
- **Best**: 1.0
- **Use**: Comparing models with different numbers of features

#### **Pearson Correlation**
```python
r = Σ(x - x_mean)(y - y_mean) / sqrt(Σ(x - x_mean)² * Σ(y - y_mean)²)
```
- **Range**: [-1, 1]
- **Best**: ±1.0
- **Measures**: Linear relationship strength

---

## 🎯 Clustering Metrics

### Internal Metrics (No Ground Truth Required)

#### **Silhouette Score**
```python
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
- **Range**: [-1, 1]
- **Best**: 1.0
- **Interpretation**:
  - > 0.7: Strong structure
  - 0.5-0.7: Medium structure
  - 0.25-0.5: Weak structure
  - < 0.25: No structure

#### **Calinski-Harabasz Index**
```python
ch = (trace(B) / (k-1)) / (trace(W) / (n-k))
```
- **Range**: [0, ∞)
- **Best**: Higher values
- **Measures**: Ratio of between/within cluster dispersion

#### **Davies-Bouldin Index**
```python
db = (1/k) * Σ max(R_ij) for j ≠ i
```
- **Range**: [0, ∞)
- **Best**: Lower values
- **Measures**: Average similarity between clusters

### External Metrics (Ground Truth Required)

#### **Adjusted Rand Index (ARI)**
```python
ari = (RI - Expected_RI) / (max(RI) - Expected_RI)
```
- **Range**: [-1, 1]
- **Best**: 1.0
- **Adjusted**: For chance

#### **Normalized Mutual Information (NMI)**
```python
nmi = MI(U,V) / sqrt(H(U) * H(V))
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Measures**: Information shared between clusterings

#### **V-Measure**
```python
v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
```
- **Range**: [0, 1]
- **Best**: 1.0
- **Balance**: Between homogeneity and completeness

---

## 🔀 Multi-Class Strategies

### Averaging Methods

#### **Macro Average**
```python
macro_avg = Σ metric_per_class / n_classes
```
- **Use**: When all classes are equally important
- **Characteristic**: Treats all classes equally

#### **Micro Average**
```python
micro_avg = metric(all_TP, all_FP, all_FN)
```
- **Use**: When dataset is imbalanced
- **Characteristic**: Dominated by frequent classes

#### **Weighted Average**
```python
weighted_avg = Σ (metric_per_class * support_per_class) / total_support
```
- **Use**: Most common choice
- **Characteristic**: Weighted by class frequency

---

## 📊 Metric Selection Guide

### Classification Tasks

| Scenario | Recommended Metrics |
|----------|-------------------|
| **Balanced Dataset** | Accuracy, F1-Score, ROC-AUC |
| **Imbalanced Dataset** | Precision, Recall, F1-Score, PR-AUC |
| **Cost-sensitive** | Precision (high FP cost), Recall (high FN cost) |
| **Multi-class** | Macro F1, Weighted F1, Cohen's Kappa |
| **Probability Quality** | Log Loss, Brier Score |

### Regression Tasks

| Scenario | Recommended Metrics |
|----------|-------------------|
| **General Purpose** | RMSE, MAE, R² |
| **Outlier Robust** | MAE, Huber Loss |
| **Percentage Errors** | MAPE, SMAPE |
| **Comparing Models** | Adjusted R², Cross-validated RMSE |
| **Time Series** | MASE, sMAPE |

### Clustering Tasks

| Scenario | Recommended Metrics |
|----------|-------------------|
| **No Ground Truth** | Silhouette Score, Calinski-Harabasz |
| **With Ground Truth** | ARI, NMI, V-Measure |
| **Hierarchical** | Cophenetic Correlation |
| **Density-based** | Modified Silhouette |

---

## ⚠️ Common Pitfalls

### ❌ **Accuracy Paradox**
- High accuracy doesn't mean good model
- Example: 99% accuracy on 1% positive class

### ❌ **Data Leakage in Metrics**
- Don't use test data for metric selection
- Use validation set for hyperparameter tuning

### ❌ **Inappropriate Metrics**
- Using RMSE for classification
- Using accuracy for regression
- Using ROC-AUC for multi-class without proper averaging

### ❌ **Statistical Significance**
- Report confidence intervals
- Use appropriate statistical tests
- Consider multiple runs/cross-validation

---

## 🛠️ Implementation Examples

### Python with Scikit-learn
```python
from sklearn.metrics import *

# Classification
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
roc_auc = roc_auc_score(y_true, y_proba[:, 1])

# Regression  
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Clustering
silhouette = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, y_pred)
```

### Custom Metrics
```python
def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy = average of recall for each class"""
    return recall_score(y_true, y_pred, average='macro')

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

## 📚 Additional Resources

- **Scikit-learn Metrics**: [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Paper**: "A systematic analysis of performance measures for classification tasks" (Sokolova & Lapalme, 2009)
- **Book**: "Evaluating Machine Learning Models" (Alice Zheng, 2015)

---

*Quick reference for ML practitioners | Last updated: July 2025*
