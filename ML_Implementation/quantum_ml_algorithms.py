"""
Quantum-Inspired Machine Learning Algorithms
Advanced quantum computing concepts applied to classical ML problems

Features:
- Quantum-inspired optimization algorithms
- Variational quantum classifiers
- Quantum feature maps and kernel methods
- Quantum approximate optimization algorithm (QAOA)
- Quantum neural networks
- Entanglement-based feature selection
- Quantum state preparation for data encoding
- Hybrid classical-quantum algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

@dataclass
class QuantumConfig:
    """Configuration for quantum-inspired algorithms"""
    n_qubits: int = 4
    n_layers: int = 3
    max_iterations: int = 100
    learning_rate: float = 0.01
    optimizer: str = 'COBYLA'  # COBYLA, SLSQP, Nelder-Mead
    shot_noise: bool = True
    n_shots: int = 1024
    backend: str = 'statevector'  # statevector, qasm_simulator
    
class QuantumStateGenerator:
    """Generate quantum states for data encoding"""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_params = 2 ** n_qubits
        
    def amplitude_encoding(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum amplitudes"""
        # Normalize data to unit vector
        norm = np.linalg.norm(data)
        if norm == 0:
            return np.zeros(self.n_params)
        
        normalized_data = data / norm
        
        # Pad or truncate to match quantum state size
        if len(normalized_data) > self.n_params:
            quantum_state = normalized_data[:self.n_params]
        else:
            quantum_state = np.zeros(self.n_params)
            quantum_state[:len(normalized_data)] = normalized_data
        
        # Ensure normalization
        quantum_state = quantum_state / np.linalg.norm(quantum_state)
        
        return quantum_state
    
    def angle_encoding(self, data: np.ndarray) -> List[float]:
        """Encode data using rotation angles"""
        # Scale data to [0, 2π]
        scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 * np.pi
        
        # Create rotation angles for each qubit
        angles = []
        for i in range(min(len(scaled_data), self.n_qubits)):
            angles.append(scaled_data[i])
        
        # Pad with zeros if needed
        while len(angles) < self.n_qubits:
            angles.append(0.0)
        
        return angles
    
    def basis_encoding(self, data: np.ndarray) -> int:
        """Encode data as computational basis state"""
        # Convert to binary representation
        binary_data = np.where(data > np.median(data), 1, 0)
        
        # Convert binary array to integer
        state_index = 0
        for i, bit in enumerate(binary_data[:self.n_qubits]):
            state_index += bit * (2 ** (self.n_qubits - 1 - i))
        
        return state_index

class QuantumGates:
    """Quantum gate operations using matrix representations"""
    
    @staticmethod
    def pauli_x():
        """Pauli-X (NOT) gate"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y():
        """Pauli-Y gate"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z():
        """Pauli-Z gate"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard():
        """Hadamard gate"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def rotation_x(theta: float):
        """Rotation around X-axis"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -1j * sin_half], 
                        [-1j * sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def rotation_y(theta: float):
        """Rotation around Y-axis"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -sin_half], 
                        [sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float):
        """Rotation around Z-axis"""
        exp_pos = np.exp(1j * theta / 2)
        exp_neg = np.exp(-1j * theta / 2)
        return np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)
    
    @staticmethod
    def cnot():
        """Controlled-NOT gate"""
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)

class QuantumCircuit:
    """Quantum circuit simulator"""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0  # Initialize to |00...0⟩
        
    def reset(self):
        """Reset circuit to initial state"""
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
    
    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate to specified qubit"""
        if qubit >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range")
        
        # Create the full gate matrix
        full_gate = np.eye(1, dtype=complex)
        
        for i in range(self.n_qubits):
            if i == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=complex))
        
        # Apply gate to state
        self.state = full_gate @ self.state
    
    def apply_two_qubit_gate(self, gate: np.ndarray, control: int, target: int):
        """Apply two-qubit gate"""
        if control >= self.n_qubits or target >= self.n_qubits:
            raise ValueError("Qubit indices out of range")
        
        # For simplicity, implement CNOT specifically
        if np.allclose(gate, QuantumGates.cnot()):
            self._apply_cnot(control, target)
    
    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        new_state = np.zeros_like(self.state)
        
        for i in range(self.n_states):
            # Convert state index to binary
            binary_state = format(i, f'0{self.n_qubits}b')
            binary_list = list(binary_state)
            
            # Apply CNOT logic
            if binary_list[control] == '1':
                # Flip target bit
                binary_list[target] = '1' if binary_list[target] == '0' else '0'
            
            # Convert back to index
            new_index = int(''.join(binary_list), 2)
            new_state[new_index] = self.state[i]
        
        self.state = new_state
    
    def measure(self, qubit: int = None) -> int:
        """Measure specified qubit or entire system"""
        probabilities = np.abs(self.state) ** 2
        
        if qubit is None:
            # Measure entire system
            return np.random.choice(self.n_states, p=probabilities)
        else:
            # Measure specific qubit
            prob_0 = 0
            prob_1 = 0
            
            for i in range(self.n_states):
                bit_value = (i >> (self.n_qubits - 1 - qubit)) & 1
                if bit_value == 0:
                    prob_0 += probabilities[i]
                else:
                    prob_1 += probabilities[i]
            
            return np.random.choice([0, 1], p=[prob_0, prob_1])
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.state) ** 2
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """Calculate expectation value of observable"""
        return np.real(np.conj(self.state) @ observable @ self.state)

class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier implementation"""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.params = None
        self.scaler = StandardScaler()
        self.classes_ = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum components
        self.state_generator = QuantumStateGenerator(self.config.n_qubits)
        
    def _create_feature_map(self, x: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        """Create quantum feature map circuit"""
        circuit = QuantumCircuit(self.config.n_qubits)
        
        # Encode data
        angles = self.state_generator.angle_encoding(x)
        
        # Apply feature map layers
        param_idx = 0
        for layer in range(self.config.n_layers):
            # Data encoding layer
            for i in range(self.config.n_qubits):
                if i < len(angles):
                    circuit.apply_single_qubit_gate(
                        QuantumGates.rotation_y(angles[i]), i
                    )
            
            # Parameterized layer
            for i in range(self.config.n_qubits):
                if param_idx < len(params):
                    circuit.apply_single_qubit_gate(
                        QuantumGates.rotation_z(params[param_idx]), i
                    )
                    param_idx += 1
            
            # Entangling layer
            for i in range(self.config.n_qubits - 1):
                circuit.apply_two_qubit_gate(
                    QuantumGates.cnot(), i, i + 1
                )
        
        return circuit
    
    def _measure_expectation(self, x: np.ndarray, params: np.ndarray) -> float:
        """Measure expectation value for classification"""
        circuit = self._create_feature_map(x, params)
        
        # Create observable (Pauli-Z on first qubit)
        observable = np.zeros((2**self.config.n_qubits, 2**self.config.n_qubits))
        for i in range(2**self.config.n_qubits):
            # Check if first qubit is 0 or 1
            first_qubit = (i >> (self.config.n_qubits - 1)) & 1
            observable[i, i] = 1 if first_qubit == 0 else -1
        
        return circuit.expectation_value(observable)
    
    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Cost function for optimization"""
        total_cost = 0
        
        for i in range(len(X)):
            expectation = self._measure_expectation(X[i], params)
            prediction = 1 if expectation > 0 else 0
            
            # Simple classification loss
            if prediction != y[i]:
                total_cost += 1
        
        return total_cost / len(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the variational quantum classifier"""
        self.logger.info("Training Variational Quantum Classifier...")
        
        # Store classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert labels to 0/1
        y_binary = np.where(y == self.classes_[0], 0, 1)
        
        # Initialize parameters
        n_params = self.config.n_qubits * self.config.n_layers
        self.params = np.random.uniform(0, 2*np.pi, n_params)
        
        # Optimize parameters
        self.logger.info("Optimizing quantum circuit parameters...")
        
        result = minimize(
            self._cost_function,
            self.params,
            args=(X_scaled, y_binary),
            method=self.config.optimizer,
            options={'maxiter': self.config.max_iterations}
        )
        
        self.params = result.x
        
        self.logger.info(f"Optimization completed. Final cost: {result.fun:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.params is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for x in X_scaled:
            expectation = self._measure_expectation(x, self.params)
            prediction = 0 if expectation > 0 else 1
            predictions.append(self.classes_[prediction])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.params is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = []
        
        for x in X_scaled:
            expectation = self._measure_expectation(x, self.params)
            # Convert expectation value to probability
            prob_0 = (1 + expectation) / 2
            prob_1 = 1 - prob_0
            probabilities.append([prob_0, prob_1])
        
        return np.array(probabilities)

class QuantumKernelMethod:
    """Quantum kernel methods for machine learning"""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.state_generator = QuantumStateGenerator(self.config.n_qubits)
        
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel between two data points"""
        # Create quantum states
        circuit1 = QuantumCircuit(self.config.n_qubits)
        circuit2 = QuantumCircuit(self.config.n_qubits)
        
        # Encode data points
        angles1 = self.state_generator.angle_encoding(x1)
        angles2 = self.state_generator.angle_encoding(x2)
        
        # Apply feature maps
        for i in range(self.config.n_qubits):
            if i < len(angles1):
                circuit1.apply_single_qubit_gate(
                    QuantumGates.rotation_y(angles1[i]), i
                )
            if i < len(angles2):
                circuit2.apply_single_qubit_gate(
                    QuantumGates.rotation_y(angles2[i]), i
                )
        
        # Compute inner product (fidelity)
        state1 = circuit1.state
        state2 = circuit2.state
        
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        return fidelity
    
    def compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute full kernel matrix"""
        n = len(X)
        kernel_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                kernel_value = self.quantum_kernel(X[i], X[j])
                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value
        
        return kernel_matrix

class QuantumApproximateOptimizationAlgorithm:
    """Quantum Approximate Optimization Algorithm (QAOA)"""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.optimal_params = None
        
    def cost_hamiltonian(self, solution: np.ndarray, problem_matrix: np.ndarray) -> float:
        """Evaluate cost function for optimization problem"""
        return solution.T @ problem_matrix @ solution
    
    def mixer_hamiltonian(self, circuit: QuantumCircuit, beta: float):
        """Apply mixer Hamiltonian (X rotations)"""
        for qubit in range(circuit.n_qubits):
            circuit.apply_single_qubit_gate(QuantumGates.rotation_x(2 * beta), qubit)
    
    def cost_hamiltonian_evolution(self, circuit: QuantumCircuit, gamma: float, 
                                 problem_matrix: np.ndarray):
        """Apply cost Hamiltonian evolution"""
        # Simplified implementation - apply Z rotations based on problem matrix
        for i in range(circuit.n_qubits):
            for j in range(circuit.n_qubits):
                if i != j and problem_matrix[i, j] != 0:
                    # Apply ZZ interaction (simplified)
                    angle = gamma * problem_matrix[i, j]
                    circuit.apply_single_qubit_gate(
                        QuantumGates.rotation_z(angle), i
                    )
    
    def qaoa_circuit(self, params: np.ndarray, problem_matrix: np.ndarray) -> QuantumCircuit:
        """Create QAOA circuit"""
        circuit = QuantumCircuit(self.config.n_qubits)
        
        # Initialize in superposition
        for qubit in range(circuit.n_qubits):
            circuit.apply_single_qubit_gate(QuantumGates.hadamard(), qubit)
        
        # Apply QAOA layers
        n_layers = len(params) // 2
        for layer in range(n_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            # Cost Hamiltonian
            self.cost_hamiltonian_evolution(circuit, gamma, problem_matrix)
            
            # Mixer Hamiltonian
            self.mixer_hamiltonian(circuit, beta)
        
        return circuit
    
    def optimize(self, problem_matrix: np.ndarray, p: int = 1) -> Dict[str, Any]:
        """Run QAOA optimization"""
        # Initialize parameters
        initial_params = np.random.uniform(0, 2*np.pi, 2*p)
        
        def objective_function(params):
            circuit = self.qaoa_circuit(params, problem_matrix)
            
            # Measure expectation value of cost Hamiltonian
            total_expectation = 0
            n_measurements = self.config.n_shots
            
            for _ in range(n_measurements):
                measurement = circuit.measure()
                # Convert measurement to binary solution
                solution = np.array([int(b) for b in format(measurement, f'0{self.config.n_qubits}b')])
                cost = self.cost_hamiltonian(solution, problem_matrix)
                total_expectation += cost
            
            return total_expectation / n_measurements
        
        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            method=self.config.optimizer,
            options={'maxiter': self.config.max_iterations}
        )
        
        self.optimal_params = result.x
        
        return {
            'optimal_params': self.optimal_params,
            'optimal_cost': result.fun,
            'optimization_result': result
        }

class QuantumNeuralNetwork:
    """Quantum Neural Network implementation"""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.weights = None
        self.scaler = StandardScaler()
        
    def quantum_layer(self, circuit: QuantumCircuit, weights: np.ndarray, 
                     start_idx: int) -> int:
        """Apply a quantum neural network layer"""
        idx = start_idx
        
        # Parameterized single-qubit rotations
        for qubit in range(circuit.n_qubits):
            if idx < len(weights):
                circuit.apply_single_qubit_gate(
                    QuantumGates.rotation_y(weights[idx]), qubit
                )
                idx += 1
            if idx < len(weights):
                circuit.apply_single_qubit_gate(
                    QuantumGates.rotation_z(weights[idx]), qubit
                )
                idx += 1
        
        # Entangling gates
        for qubit in range(circuit.n_qubits - 1):
            circuit.apply_two_qubit_gate(
                QuantumGates.cnot(), qubit, (qubit + 1) % circuit.n_qubits
            )
        
        return idx
    
    def forward_pass(self, x: np.ndarray, weights: np.ndarray) -> float:
        """Forward pass through quantum neural network"""
        circuit = QuantumCircuit(self.config.n_qubits)
        
        # Data encoding
        state_gen = QuantumStateGenerator(self.config.n_qubits)
        angles = state_gen.angle_encoding(x)
        
        for i, angle in enumerate(angles):
            if i < circuit.n_qubits:
                circuit.apply_single_qubit_gate(
                    QuantumGates.rotation_y(angle), i
                )
        
        # Apply quantum layers
        weight_idx = 0
        for layer in range(self.config.n_layers):
            weight_idx = self.quantum_layer(circuit, weights, weight_idx)
        
        # Measure expectation value
        observable = np.zeros((2**self.config.n_qubits, 2**self.config.n_qubits))
        for i in range(2**self.config.n_qubits):
            first_qubit = (i >> (self.config.n_qubits - 1)) & 1
            observable[i, i] = 1 if first_qubit == 0 else -1
        
        return circuit.expectation_value(observable)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train quantum neural network"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize weights
        n_weights = self.config.n_layers * self.config.n_qubits * 2
        self.weights = np.random.uniform(0, 2*np.pi, n_weights)
        
        training_history = {'loss': [], 'accuracy': []}
        
        for epoch in range(self.config.max_iterations):
            epoch_loss = 0
            correct_predictions = 0
            
            for i in range(len(X_scaled)):
                # Forward pass
                prediction = self.forward_pass(X_scaled[i], self.weights)
                predicted_class = 1 if prediction > 0 else 0
                
                # Calculate loss (simplified)
                target = 1 if y[i] == 1 else -1
                loss = (prediction - target) ** 2
                epoch_loss += loss
                
                # Count correct predictions
                if predicted_class == y[i]:
                    correct_predictions += 1
                
                # Simple gradient update (approximated)
                for j in range(len(self.weights)):
                    # Finite difference gradient
                    eps = 0.01
                    weights_plus = self.weights.copy()
                    weights_minus = self.weights.copy()
                    weights_plus[j] += eps
                    weights_minus[j] -= eps
                    
                    pred_plus = self.forward_pass(X_scaled[i], weights_plus)
                    pred_minus = self.forward_pass(X_scaled[i], weights_minus)
                    
                    gradient = (pred_plus - pred_minus) / (2 * eps)
                    
                    # Update weight
                    self.weights[j] -= self.config.learning_rate * gradient * (prediction - target)
            
            # Record metrics
            avg_loss = epoch_loss / len(X_scaled)
            accuracy = correct_predictions / len(X_scaled)
            
            training_history['loss'].append(avg_loss)
            training_history['accuracy'].append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return training_history

# Example usage and testing
if __name__ == "__main__":
    # Generate sample quantum classification data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print("Quantum Machine Learning Demo")
    print("=" * 40)
    
    # Test Variational Quantum Classifier
    print("\n1. Variational Quantum Classifier")
    print("-" * 30)
    
    config = QuantumConfig(n_qubits=4, n_layers=2, max_iterations=50)
    vqc = VariationalQuantumClassifier(config)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train and evaluate
    vqc.fit(X_train, y_train)
    predictions = vqc.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"VQC Accuracy: {accuracy:.4f}")
    
    # Test Quantum Kernel Method
    print("\n2. Quantum Kernel Method")
    print("-" * 30)
    
    qkm = QuantumKernelMethod(config)
    kernel_matrix = qkm.compute_kernel_matrix(X_train[:10])  # Small sample for demo
    
    print(f"Kernel matrix shape: {kernel_matrix.shape}")
    print(f"Average kernel value: {np.mean(kernel_matrix):.4f}")
    
    # Test QAOA
    print("\n3. Quantum Approximate Optimization Algorithm")
    print("-" * 30)
    
    # Create a simple optimization problem (Max-Cut)
    problem_size = 4
    problem_matrix = np.random.rand(problem_size, problem_size)
    problem_matrix = (problem_matrix + problem_matrix.T) / 2  # Make symmetric
    
    qaoa = QuantumApproximateOptimizationAlgorithm(config)
    result = qaoa.optimize(problem_matrix, p=1)
    
    print(f"QAOA optimal cost: {result['optimal_cost']:.4f}")
    print(f"Optimal parameters: {result['optimal_params']}")
    
    # Test Quantum Neural Network
    print("\n4. Quantum Neural Network")
    print("-" * 30)
    
    qnn = QuantumNeuralNetwork(config)
    history = qnn.train(X_train[:20], y_train[:20])  # Small sample for demo
    
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    
    print("\nQuantum ML demo completed!")
