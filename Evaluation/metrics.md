```mermaid
graph TD;
  A[Evaluation Metrics] --> B[Classification Metrics];
  A --> C[Regression Metrics];

  B --> B1[Accuracy];
  B1 -->|Formula| B1a((Correct Predictions / Total Predictions))
  B1 -->|Python Code| B1b["accuracy_score(y_true, y_pred)"];

  B --> B2[Precision];
  B2 -->|Formula| B2a((TP / (TP + FP)))
  B2 -->|Python Code| B2b["precision_score(y_true, y_pred)"];

  B --> B3[Recall];
  B3 -->|Formula| B3a((TP / (TP + FN)))
  B3 -->|Python Code| B3b["recall_score(y_true, y_pred)"];

  B --> B4[F1 Score];
  B4 -->|Formula| B4a((2 * (Precision * Recall) / (Precision + Recall)))
  B4 -->|Python Code| B4b["f1_score(y_true, y_pred)"];

  B --> B5[Confusion Matrix];
  B5 -->|Definition| B5a["Summarizes classification results"]
  B5 -->|Python Code| B5b["confusion_matrix(y_true, y_pred)"];

  B --> B6[AUC-ROC];
  B6 -->|Definition| B6a["Measures classifier's ability to separate classes"]
  B6 -->|Python Code| B6b["roc_auc_score(y_true, y_scores)"];

  C --> C1[Mean Absolute Error (MAE)];
  C1 -->|Formula| C1a(("Mean(|y_true - y_pred|)"));
  C1 -->|Python Code| C1b["mean_absolute_error(y_true, y_pred)"];
