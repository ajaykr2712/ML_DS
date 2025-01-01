import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weights, bias):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights) + bias)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        if self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(weights ** 2)

        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(linear_model)

            # Gradient Calculation with Regularization
            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)

            if self.regularization == 'l1':
                dw += (self.lambda_reg / m) * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += (self.lambda_reg / m) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


if __name__ == "__main__":
    # Generate a sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # No Regularization
    model_no_reg = LogisticRegression()
    model_no_reg.fit(X_train, y_train)
    y_pred_no_reg = model_no_reg.predict(X_test)
    print(f'Accuracy (No Regularization): {accuracy_score(y_test, y_pred_no_reg):.4f}')

    # L1 Regularization (Lasso)
    model_l1 = LogisticRegression(regularization='l1', lambda_reg=0.1)
    model_l1.fit(X_train, y_train)
    y_pred_l1 = model_l1.predict(X_test)
    print(f'Accuracy (L1 Regularization): {accuracy_score(y_test, y_pred_l1):.4f}')

    # L2 Regularization (Ridge)
    model_l2 = LogisticRegression(regularization='l2', lambda_reg=0.1)
    model_l2.fit(X_train, y_train)
    y_pred_l2 = model_l2.predict(X_test)
    print(f'Accuracy (L2 Regularization): {accuracy_score(y_test, y_pred_l2):.4f}')