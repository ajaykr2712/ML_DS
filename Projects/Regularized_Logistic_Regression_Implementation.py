import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RegularizedLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, reg_type='l2', reg_strength=1.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.reg_type = reg_type
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Calculate gradients
            dz = predictions - y
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz)
            
            # Add regularization gradient
            if self.reg_type == 'l1':
                dw += (self.reg_strength * np.sign(self.weights))
            elif self.reg_type == 'l2':
                dw += (self.reg_strength * self.weights)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                         n_redundant=5, random_state=42)

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models with different regularization
models = {
    'No Regularization': RegularizedLogisticRegression(reg_strength=0),
    'L1 (Lasso)': RegularizedLogisticRegression(reg_type='l1', reg_strength=0.1),
    'L2 (Ridge)': RegularizedLogisticRegression(reg_type='l2', reg_strength=0.1)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    results[name] = {'train': train_score, 'test': test_score}
    print(f"\n{name}:")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")

# Plot feature weights comparison
plt.figure(figsize=(12, 6))
for i, (name, model) in enumerate(models.items()):
    plt.plot(model.weights, label=name, marker='o', linestyle='-', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.title('Feature Weights Comparison Across Different Regularization Types')
plt.legend()
plt.grid(True)
plt.tight_layout()