

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def forward_propagation(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def compute_cost(self, y_true, y_pred):
        m = len(y_true)
        cost = (-1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def compute_gradients(self, X, y_true, y_pred):
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)
        return dw, db

    def update_parameters(self, dw, db):
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_parameters(n)
        y = y.reshape(m, 1)

        for _ in range(self.num_iterations):
            y_pred = self.forward_propagation(X)
            cost = self.compute_cost(y, y_pred)
            dw, db = self.compute_gradients(X, y, y_pred)
            self.update_parameters(dw, db)

    def predict_proba(self, X):
        return self.forward_propagation(X)

    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        return (y_pred >= threshold).astype(int)

# Evaluation metrics
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

# Sample dataset and model evaluation
if __name__ == "__main__":
    # Generate a sample dataset
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    # Split the data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train the logistic regression model
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")