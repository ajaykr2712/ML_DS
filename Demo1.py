import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, mean_squared_error,
                             mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class ModelEvaluation:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def perform_kfold_cv(self, model, k=5):
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy')
        return scores.mean()

    def evaluate_classification(self, model):
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        metrics = {
            'Accuracy': accuracy_score(self.y, predictions),
            'Precision': precision_score(self.y, predictions, average="weighted"),
            'Recall': recall_score(self.y, predictions, average="weighted"),
            'F1 Score': f1_score(self.y, predictions, average="weighted"),
            'ROC AUC': roc_auc_score(self.y, model.predict_proba(self.X), multi_class='ovr'),
            'Matthews Correlation Coefficient': matthews_corrcoef(self.y, predictions),
            'Cohen Kappa': cohen_kappa_score(self.y, predictions)
        }
        return metrics

class ModelSelection:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500)
        }

    def create_pipeline(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in available models.")

        steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', self.models[model_name])
        ]
        return Pipeline(steps)

if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    evaluator = ModelEvaluation(X, y)
    model_selector = ModelSelection()

    # Evaluate each model
    for model_name in model_selector.models:
        pipeline = model_selector.create_pipeline(model_name)
        print(f"Evaluating {model_name}:")
        metrics = evaluator.evaluate_classification(pipeline)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 30)
