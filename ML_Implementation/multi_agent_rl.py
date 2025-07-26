"""
Multi-Agent Reinforcement Learning Framework
Advanced MARL implementation with various algorithms and environments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
import random
import logging
from collections import deque, namedtuple
from dataclasses import dataclass
import json

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class MARLConfig:
    """Configuration for Multi-Agent RL"""
    num_agents: int = 2
    state_dim: int = 64
    action_dim: int = 4
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 64
    update_frequency: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

class ReplayBuffer:
    """Replay buffer for experience storage"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for policy-based methods"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class AttentionCritic(nn.Module):
    """Attention-based critic for multi-agent settings"""
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 256):
        super(AttentionCritic, self).__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Individual agent encoders
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, states, actions):
        # Encode states and actions
        state_encoded = F.relu(self.state_encoder(states))
        action_encoded = F.relu(self.action_encoder(actions))
        
        # Combine state and action representations
        combined = state_encoded + action_encoded
        
        # Apply attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Global representation (mean pooling)
        global_repr = torch.mean(attended, dim=0)
        
        # Compute value
        value = self.value_head(global_repr)
        return value

class DDPGAgent:
    """DDPG Agent for continuous action spaces"""
    
    def __init__(self, config: MARLConfig, agent_id: int):
        self.config = config
        self.agent_id = agent_id
        self.epsilon = config.epsilon_start
        
        # Networks
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim)
        self.actor_target = Actor(config.state_dim, config.action_dim, config.hidden_dim)
        self.critic = Critic(config.state_dim, config.action_dim, config.hidden_dim)
        self.critic_target = Critic(config.state_dim, config.action_dim, config.hidden_dim)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Noise for exploration
        self.noise_scale = 0.1
        
    def select_action(self, state, add_noise=True):
        """Select action using actor network"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        return action
    
    def update(self, experiences: List[Experience]):
        """Update agent networks"""
        if len(experiences) < self.config.batch_size:
            return
            
        # Sample batch
        batch = random.sample(experiences, self.config.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch]).unsqueeze(1)
        
        # Update critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def _soft_update(self, local_model, target_model):
        """Soft update of target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data
            )

class MADDPGAgent:
    """Multi-Agent DDPG Agent"""
    
    def __init__(self, config: MARLConfig, agent_id: int):
        self.config = config
        self.agent_id = agent_id
        
        # Individual actor
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim)
        self.actor_target = Actor(config.state_dim, config.action_dim, config.hidden_dim)
        
        # Centralized critic (sees all agents' states and actions)
        total_state_dim = config.state_dim * config.num_agents
        total_action_dim = config.action_dim * config.num_agents
        self.critic = Critic(total_state_dim, total_action_dim, config.hidden_dim)
        self.critic_target = Critic(total_state_dim, total_action_dim, config.hidden_dim)
        
        # Copy to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        self.noise_scale = 0.1
    
    def select_action(self, state, add_noise=True):
        """Select action using individual actor"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        return action

class MARLEnvironment:
    """Multi-Agent RL Environment"""
    
    def __init__(self, config: MARLConfig):
        self.config = config
        self.agents = [MADDPGAgent(config, i) for i in range(config.num_agents)]
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.episode_rewards = []
        self.training_step = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial states"""
        states = []
        for _ in range(self.config.num_agents):
            state = np.random.randn(self.config.state_dim)
            states.append(state)
        return states
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Environment step function"""
        # Simple cooperative task: agents try to reach target positions
        next_states = []
        rewards = []
        dones = []
        
        for i, action in enumerate(actions):
            # Simple dynamics: next_state = current_state + action + noise
            next_state = np.random.randn(self.config.state_dim) + action[:self.config.state_dim] * 0.1
            next_states.append(next_state)
            
            # Reward based on distance to target (cooperative)
            target = np.zeros(self.config.state_dim)
            distance = np.linalg.norm(next_state - target)
            reward = -distance + np.random.normal(0, 0.1)  # Add noise
            rewards.append(reward)
            
            # Episode ends randomly or when close to target
            done = distance < 0.5 or np.random.random() < 0.01
            dones.append(done)
        
        info = {'episode_step': self.training_step}
        return next_states, rewards, dones, info
    
    def train_episode(self, max_steps: int = 1000) -> Dict[str, float]:
        """Train for one episode"""
        states = self.reset()
        episode_rewards = [0.0] * self.config.num_agents
        step_count = 0
        
        while step_count < max_steps:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i])
                actions.append(action)
            
            # Environment step
            next_states, rewards, dones, _ = self.step(actions)
            
            # Store experiences
            for i in range(self.config.num_agents):
                experience = Experience(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=dones[i]
                )
                self.replay_buffer.push(experience)
                episode_rewards[i] += rewards[i]
            
            # Update agents
            if len(self.replay_buffer) > self.config.batch_size and \
               self.training_step % self.config.update_frequency == 0:
                
                for agent in self.agents:
                    agent.update(list(self.replay_buffer.buffer))
            
            states = next_states
            step_count += 1
            self.training_step += 1
            
            # Check if episode is done
            if any(dones):
                break
        
        avg_reward = np.mean(episode_rewards)
        self.episode_rewards.append(avg_reward)
        
        return {
            'episode_reward': avg_reward,
            'episode_length': step_count,
            'buffer_size': len(self.replay_buffer)
        }
    
    def train(self, num_episodes: int = 1000) -> List[Dict[str, float]]:
        """Train the multi-agent system"""
        training_history = []
        
        for episode in range(num_episodes):
            episode_info = self.train_episode()
            training_history.append(episode_info)
            
            if episode % 100 == 0:
                avg_reward = np.mean([info['episode_reward'] for info in training_history[-100:]])
                self.logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}")
        
        return training_history
    
    def save_models(self, filepath_prefix: str):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }, f"{filepath_prefix}_agent_{i}.pth")
        
        self.logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            checkpoint = torch.load(f"{filepath_prefix}_agent_{i}.pth")
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.logger.info(f"Models loaded with prefix: {filepath_prefix}")

# Example usage and experiments
if __name__ == "__main__":
    # Configuration
    config = MARLConfig(
        num_agents=3,
        state_dim=32,
        action_dim=8,
        hidden_dim=128,
        learning_rate=0.001,
        buffer_size=50000,
        batch_size=32
    )
    
    # Initialize environment
    env = MARLEnvironment(config)
    
    # Train
    training_history = env.train(num_episodes=500)
    
    # Save results
    with open('marl_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save models
    env.save_models('marl_models')
    
    print(f"Training completed. Final average reward: {np.mean(env.episode_rewards[-100:]):.4f}")
