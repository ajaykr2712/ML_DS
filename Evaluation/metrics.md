# Top Evaluation Metrics
Classification Metrics.
## Accuracy
Definition: The ratio of correctly predicted instances to the total instances to be prepared and done simultaneously 
Formula:
 

 Python Calculation:
 ```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)

 ```


 ## Precision
Definition: The ratio of true positive predictions to the total predicted positives.
Formula:
 

Python Calculation:

```
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)

```

## Recall (Sensitivity)

Definition: The ratio of true positive predictions to all actual positives.

Python Calculation:
```

from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)

```

## F1 Score
Definition: The harmonic mean of precision and recall, providing a balance between the two.


```
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)

```

## Confusion Matrix
Definition: A table that summarizes the performance of a classification algorithm by showing true positives, false positives, true negatives, and false negatives.

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

```

## Area Under Curve (AUC-ROC)
Definition: Measures the ability of a classifier to distinguish between classes; it is particularly useful for binary classification problems

```
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_scores)  # y_scores are the probabilities of the positive class

```

# Regression Metrics

## Mean Absolute Error (MAE)

Definition: The average absolute difference between predicted and actual values.


```
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)

```