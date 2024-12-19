# Install Necessary Libraries
pip install numpy pandas scikit-learn


# Import and use the classes as needed:

# Create instances
evaluator = ModelEvaluation(X, y)
model_selection = ModelSelection()

# Perform evaluation
results = evaluator.perform_kfold_cv(model_selection.models['rf'])


## This implementation provides a solid foundation for machine learning model evaluation and selection. You can extend it further based on your specific needs.



### The code in Demo.py contains
This code includes:
 Cross-Validation Techniques:
        K-Fold CV
        Leave-One-Out CV
        Stratified K-Fold CV


 Evaluation Metrics:
    Classification metrics (accuracy, precision, recall, F1, AUC-ROC, MCC, Kappa)
    Regression metrics (R², MAE, MSE, RMSE)


 Model Selection:
    Different model implementations
    Regularization options
    Pipeline creation