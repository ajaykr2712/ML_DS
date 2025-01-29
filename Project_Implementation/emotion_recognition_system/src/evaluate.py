import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    log_loss,
    hamming_loss
)
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

def load_data(data_path):
    """Load and preprocess dataset"""
    df = pd.read_csv(data_path)
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
    return df, le

def evaluate_model(model, X_test, y_test, le):
    """Calculate and return all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'hamming_loss': hamming_loss(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics.update({
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr'),
            'log_loss': log_loss(y_test, y_proba)
        })
    
    # Confusion matrix data
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Class names
    class_names = le.classes_.tolist()
    
    return metrics, cm, class_names

def save_plots(cm, class_names, output_dir):
    """Save visualization plots"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix plot
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curve plot (example for multi-class)
    # Note: This might need adaptation based on specific model
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def main(data_path, output_dir):
    # Load and prepare data
    df, le = load_data(data_path)
    X = df['content']  # Replace with actual features
    y = df['sentiment_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model (replace with actual model)
    model = DummyClassifier(strategy='stratified')
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics, cm, class_names = evaluate_model(model, X_test, y_test, le)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save plots
    save_plots(cm, class_names, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to input data CSV file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    main(args.data_path, args.output_dir) 