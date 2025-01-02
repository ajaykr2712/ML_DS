import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_reg=0.01):
        """
        Initializes the Logistic Regression model.

        Parameters:
        - learning_rate (float): The step size for updating weights during training.
        - num_iterations (int): Number of iterations for training the model.
        - regularization (str or None): Type of regularization ('l1' for Lasso, 'l2' for Ridge, or None).
        - lambda_reg (float): Regularization strength (applicable if regularization is not None).
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Applies the sigmoid function element-wise to the input.

        Parameters:
        - z (numpy array): The input array.

        Returns:
        - numpy array: The result of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weights, bias):
        """
        Computes the cost (log-loss) for the current weights and bias.

        Parameters:
        - X (numpy array): Feature matrix of shape (m, n).
        - y (numpy array): Labels of shape (m,).
        - weights (numpy array): Current weight vector of shape (n,).
        - bias (float): Current bias.

        Returns:
        - float: The computed cost.
        """
        m = len(y)  # Number of samples
        h = self.sigmoid(np.dot(X, weights) + bias)  # Predicted probabilities
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # Log-loss

        # Add regularization term if applicable
        if self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(weights ** 2)

        return cost

    def fit(self, X, y):
        """
        Trains the Logistic Regression model using gradient descent.

        Parameters:
        - X (numpy array): Training feature matrix of shape (m, n).
        - y (numpy array): Training labels of shape (m,).
        """
        m, n = X.shape  # m: number of samples, n: number of features
        self.weights = np.zeros(n)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero

        for _ in range(self.num_iterations):
            # Linear model calculation
            linear_model = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(linear_model)  # Predicted probabilities

            # Gradient calculation
            dw = (1 / m) * np.dot(X.T, (h - y))  # Gradient for weights
            db = (1 / m) * np.sum(h - y)  # Gradient for bias

            # Add regularization to the gradients
            if self.regularization == 'l1':
                dw += (self.lambda_reg / m) * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += (self.lambda_reg / m) * self.weights

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts class labels for the input features.

        Parameters:
        - X (numpy array): Feature matrix of shape (m, n).

        Returns:
        - numpy array: Predicted class labels (0 or 1) for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)  # Predicted probabilities
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Classify as 0 or 1
        return np.array(y_predicted_cls)


if __name__ == "__main__":
    # Generate a sample dataset for binary classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model without regularization
    model_no_reg = LogisticRegression()
    model_no_reg.fit(X_train, y_train)
    y_pred_no_reg = model_no_reg.predict(X_test)
    print(f'Accuracy (No Regularization): {accuracy_score(y_test, y_pred_no_reg):.4f}')

    # Train and evaluate the model with L1 regularization
    model_l1 = LogisticRegression(regularization='l1', lambda_reg=0.1)
    model_l1.fit(X_train, y_train)
    y_pred_l1 = model_l1.predict(X_test)
    print(f'Accuracy (L1 Regularization): {accuracy_score(y_test, y_pred_l1):.4f}')

    # Train and evaluate the model with L2 regularization
    model_l2 = LogisticRegression(regularization='l2', lambda_reg=0.1)
    model_l2.fit(X_train, y_train)
    y_pred_l2 = model_l2.predict(X_test)
    print(f'Accuracy (L2 Regularization): {accuracy_score(y_test, y_pred_l2):.4f}')
