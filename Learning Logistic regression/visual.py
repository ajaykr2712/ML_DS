import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Visualization functions
def visualize_data(X, y):
    """Scatter plot of the dataset with different classes"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
    plt.title("Scatter Plot of the Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_decision_boundary(model, X, y):
    """Visualize the decision boundary of the logistic regression model"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.8)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', edgecolor='k', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', edgecolor='k', label='Class 1')
    plt.title("Decision Boundary of Logistic Regression")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_metrics(accuracy, precision, recall):
    """Bar graph of evaluation metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall']
    values = [accuracy, precision, recall]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()

if __name__ == "__main__":
    # Generate a sample dataset
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    # Visualize the dataset
    visualize_data(X, y)

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

    # Visualize the decision boundary
    visualize_decision_boundary(model, X, y)

    # Visualize the metrics
    visualize_metrics(accuracy, precision, recall)
