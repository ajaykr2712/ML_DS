"""
Federated Learning with Differential Privacy
Implementation of privacy-preserving federated learning algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
import copy

class DifferentialPrivacyMechanism:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def add_gaussian_noise(self, tensor: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """Add Gaussian noise for differential privacy."""
        if sigma is None:
            # Calculate sigma based on (epsilon, delta)-DP
            sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        
        noise = torch.normal(0, sigma, size=tensor.shape).to(tensor.device)
        return tensor + noise
    
    def add_laplacian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Laplacian noise for differential privacy."""
        scale = self.sensitivity / self.epsilon
        noise = torch.tensor(np.random.laplace(0, scale, tensor.shape)).float().to(tensor.device)
        return tensor + noise
    
    def clip_gradients(self, gradients: List[torch.Tensor], max_norm: float = 1.0) -> List[torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        clipped = []
        for grad in gradients:
            grad_norm = torch.norm(grad)
            if grad_norm > max_norm:
                clipped.append(grad * max_norm / grad_norm)
            else:
                clipped.append(grad)
        return clipped

class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.threshold = max(1, num_clients // 2)
    
    def add_random_masks(self, model_updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add random masks for secure aggregation."""
        masked_updates = {}
        for name, param in model_updates.items():
            # In practice, these would be cryptographically secure masks
            mask = torch.randn_like(param) * 0.01
            masked_updates[name] = param + mask
        return masked_updates
    
    def aggregate_with_dropout_resilience(self, 
                                        client_updates: List[Dict[str, torch.Tensor]],
                                        active_clients: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate updates with resilience to client dropouts."""
        if len(active_clients) < self.threshold:
            raise ValueError("Not enough active clients for secure aggregation")
        
        aggregated = {}
        for name in client_updates[0].keys():
            param_sum = torch.zeros_like(client_updates[0][name])
            for i in active_clients:
                param_sum += client_updates[i][name]
            aggregated[name] = param_sum / len(active_clients)
        
        return aggregated

class FederatedClient:
    """Federated learning client with privacy protection."""
    
    def __init__(self, 
                 client_id: int,
                 model: nn.Module,
                 privacy_mechanism: DifferentialPrivacyMechanism,
                 local_epochs: int = 5,
                 learning_rate: float = 0.01):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.privacy_mechanism = privacy_mechanism
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def local_training(self, train_loader: Any) -> Dict[str, torch.Tensor]:
        """Perform local training with differential privacy."""
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                
                # Clip gradients for privacy
                gradients = [param.grad for param in self.model.parameters() if param.grad is not None]
                clipped_grads = self.privacy_mechanism.clip_gradients(gradients)
                
                # Apply clipped gradients
                for param, clipped_grad in zip(self.model.parameters(), clipped_grads):
                    if param.grad is not None:
                        param.grad.data = clipped_grad
                
                self.optimizer.step()
        
        # Add noise to model updates for privacy
        model_update = {}
        for name, param in self.model.named_parameters():
            noisy_param = self.privacy_mechanism.add_gaussian_noise(param.data)
            model_update[name] = noisy_param
        
        return model_update
    
    def update_model(self, global_model_state: Dict[str, torch.Tensor]):
        """Update local model with global model state."""
        self.model.load_state_dict(global_model_state)

class FederatedServer:
    """Federated learning server with secure aggregation."""
    
    def __init__(self, 
                 global_model: nn.Module,
                 num_clients: int,
                 client_fraction: float = 1.0):
        self.global_model = global_model
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.secure_aggregation = SecureAggregation(num_clients)
        self.round_num = 0
    
    def select_clients(self) -> List[int]:
        """Select a fraction of clients for the current round."""
        num_selected = max(1, int(self.client_fraction * self.num_clients))
        return np.random.choice(self.num_clients, num_selected, replace=False).tolist()
    
    def aggregate_models(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates."""
        if not client_updates:
            return self.global_model.state_dict()
        
        # Simple federated averaging
        aggregated = {}
        for name in client_updates[0].keys():
            param_sum = torch.zeros_like(client_updates[0][name])
            for update in client_updates:
                param_sum += update[name]
            aggregated[name] = param_sum / len(client_updates)
        
        return aggregated
    
    def byzantine_robust_aggregation(self, 
                                   client_updates: List[Dict[str, torch.Tensor]],
                                   num_byzantine: int = 0) -> Dict[str, torch.Tensor]:
        """Byzantine-robust aggregation using coordinate-wise median."""
        if num_byzantine == 0:
            return self.aggregate_models(client_updates)
        
        aggregated = {}
        for name in client_updates[0].keys():
            # Stack all client parameters for this layer
            stacked_params = torch.stack([update[name] for update in client_updates])
            
            # Use coordinate-wise median for robustness
            median_params, _ = torch.median(stacked_params, dim=0)
            aggregated[name] = median_params
        
        return aggregated
    
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """Update the global model with aggregated updates."""
        self.global_model.load_state_dict(aggregated_update)
        self.round_num += 1

class PersonalizedFederatedLearning:
    """Personalized federated learning with local adaptation."""
    
    def __init__(self, base_model: nn.Module, adaptation_lr: float = 0.01):
        self.base_model = base_model
        self.adaptation_lr = adaptation_lr
        self.client_models = {}
    
    def create_personalized_model(self, client_id: int, local_data: Any) -> nn.Module:
        """Create a personalized model for a specific client."""
        # Start with global model
        personalized_model = copy.deepcopy(self.base_model)
        
        # Fine-tune on local data
        optimizer = torch.optim.SGD(personalized_model.parameters(), lr=self.adaptation_lr)
        
        personalized_model.train()
        for epoch in range(5):  # Few epochs of local adaptation
            for batch_idx, (data, target) in enumerate(local_data):
                optimizer.zero_grad()
                output = personalized_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        
        self.client_models[client_id] = personalized_model
        return personalized_model
    
    def meta_update(self, client_gradients: List[Dict[str, torch.Tensor]]):
        """Meta-update using client gradients (MAML-style)."""
        # Aggregate gradients
        meta_gradients = {}
        for name in client_gradients[0].keys():
            grad_sum = torch.zeros_like(client_gradients[0][name])
            for grads in client_gradients:
                grad_sum += grads[name]
            meta_gradients[name] = grad_sum / len(client_gradients)
        
        # Apply meta-gradients to base model
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in meta_gradients:
                    param.data -= self.adaptation_lr * meta_gradients[name]

class FederatedLearningFramework:
    """Main federated learning framework."""
    
    def __init__(self, 
                 global_model: nn.Module,
                 num_clients: int,
                 privacy_config: Dict[str, float] = None,
                 federated_config: Dict[str, Any] = None):
        
        self.global_model = global_model
        self.num_clients = num_clients
        
        # Privacy configuration
        privacy_config = privacy_config or {}
        self.privacy_mechanism = DifferentialPrivacyMechanism(
            epsilon=privacy_config.get('epsilon', 1.0),
            delta=privacy_config.get('delta', 1e-5),
            sensitivity=privacy_config.get('sensitivity', 1.0)
        )
        
        # Federated learning configuration
        federated_config = federated_config or {}
        self.server = FederatedServer(
            global_model=global_model,
            num_clients=num_clients,
            client_fraction=federated_config.get('client_fraction', 0.1)
        )
        
        # Initialize clients
        self.clients = []
        for i in range(num_clients):
            client = FederatedClient(
                client_id=i,
                model=global_model,
                privacy_mechanism=self.privacy_mechanism,
                local_epochs=federated_config.get('local_epochs', 5),
                learning_rate=federated_config.get('learning_rate', 0.01)
            )
            self.clients.append(client)
    
    def train_round(self, client_data_loaders: List[Any]) -> float:
        """Execute one round of federated training."""
        # Select clients for this round
        selected_clients = self.server.select_clients()
        
        # Collect client updates
        client_updates = []
        for client_id in selected_clients:
            if client_id < len(client_data_loaders) and client_data_loaders[client_id] is not None:
                # Update client model with current global model
                self.clients[client_id].update_model(self.global_model.state_dict())
                
                # Perform local training
                update = self.clients[client_id].local_training(client_data_loaders[client_id])
                client_updates.append(update)
        
        # Aggregate updates
        if client_updates:
            aggregated_update = self.server.aggregate_models(client_updates)
            self.server.update_global_model(aggregated_update)
        
        # Return average client participation
        return len(client_updates) / len(selected_clients) if selected_clients else 0.0
    
    def evaluate_global_model(self, test_loader: Any) -> float:
        """Evaluate the global model."""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0

# Example usage
if __name__ == "__main__":
    print("Testing Federated Learning with Differential Privacy...")
    
    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Initialize framework
    model = SimpleNet()
    
    privacy_config = {
        'epsilon': 1.0,
        'delta': 1e-5,
        'sensitivity': 1.0
    }
    
    federated_config = {
        'client_fraction': 0.1,
        'local_epochs': 3,
        'learning_rate': 0.01
    }
    
    fl_framework = FederatedLearningFramework(
        global_model=model,
        num_clients=100,
        privacy_config=privacy_config,
        federated_config=federated_config
    )
    
    print(f"Initialized federated learning with {fl_framework.num_clients} clients")
    print(f"Privacy budget: ε={privacy_config['epsilon']}, δ={privacy_config['delta']}")
    print(f"Server will select {int(federated_config['client_fraction'] * 100)}% of clients per round")
    
    # Test differential privacy mechanism
    dp_mechanism = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
    test_tensor = torch.randn(10, 10)
    noisy_tensor = dp_mechanism.add_gaussian_noise(test_tensor)
    
    print(f"\nOriginal tensor norm: {torch.norm(test_tensor):.4f}")
    print(f"Noisy tensor norm: {torch.norm(noisy_tensor):.4f}")
    print(f"Noise magnitude: {torch.norm(noisy_tensor - test_tensor):.4f}")
    
    print("\nFederated Learning with Differential Privacy implemented successfully! 🚀")
