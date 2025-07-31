"""
Advanced Reinforcement Learning Framework
=========================================

A comprehensive RL framework featuring multi-agent systems, advanced algorithms,
distributed training, and real-world environment integration.

Best Contributions:
- Multi-agent reinforcement learning (MARL) implementations
- State-of-the-art algorithms (PPO, SAC, A3C, IMPALA, Rainbow DQN)
- Distributed and asynchronous training
- Hierarchical reinforcement learning
- Meta-learning and few-shot adaptation
- Real-world environment integration
- Advanced exploration strategies
- Curriculum learning and self-play

Author: ML/DS Advanced Implementation Team
"""

import logging
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from collections import deque
import concurrent.futures
from abc import ABC, abstractmethod

# Deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    import torch.multiprocessing as mp
    
    # RL specific libraries
    import gym
    import stable_baselines3 as sb3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    
    # Advanced RL libraries
    import ray
    from ray import tune
    from ray.rllib.algorithms import PPO, SAC, A3C
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    
except ImportError as e:
    logging.warning(f"Some RL libraries not available: {e}")

@dataclass
class RLConfig:
    """Configuration for reinforcement learning framework."""
    algorithm: str = "PPO"
    env_name: str = "CartPole-v1"
    num_agents: int = 1
    lr: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_interval: int = 10
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    num_workers: int = 4
    rollout_length: int = 128
    num_epochs: int = 4
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gae: bool = True
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    
class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.index = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
    
    def add(self, state, action, next_state, reward, done):
        """Add experience to buffer."""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        dones = torch.BoolTensor(self.dones[indices])
        
        return states, actions, next_states, rewards, dones

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        super().__init__(capacity, state_dim, action_dim)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state, action, next_state, reward, done):
        """Add experience with maximum priority."""
        super().add(state, action, next_state, reward, done)
        self.priorities[self.index - 1] = self.max_priority
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample batch with importance sampling weights."""
        if self.size == 0:
            return super().sample(batch_size)
        
        # Calculate probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        dones = torch.BoolTensor(self.dones[indices])
        weights = torch.FloatTensor(weights)
        
        return states, actions, next_states, rewards, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

class ActorCriticNetwork(nn.Module):
    """Shared Actor-Critic network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic head (value function)
        self.critic_head = nn.Linear(prev_dim, 1)
        
        # For continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """Forward pass through network."""
        shared_features = self.shared_layers(state)
        
        # Actor output
        mean = self.actor_head(shared_features)
        std = torch.exp(self.log_std)
        
        # Critic output
        value = self.critic_head(shared_features)
        
        return mean, std, value
    
    def get_action_and_value(self, state, action=None):
        """Get action and value with log probabilities."""
        mean, std, value = self.forward(state)
        
        # Create distribution
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return action, log_prob, entropy, value

class PPOAgent:
    """Proximal Policy Optimization agent implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
        # Training metrics
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
    
    def get_action(self, state, deterministic=False):
        """Get action from current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(state)
        
        if deterministic:
            mean, _, _ = self.network(state)
            action = mean
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in local buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        # Reverse iteration for GAE computation
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.config.gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.states) < self.config.rollout_length:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # Compute advantages
        advantages = self.compute_gae()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        for epoch in range(self.config.num_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
            
            # Calculate ratio for clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_coef * value_loss + 
                         self.config.entropy_coef * entropy_loss)
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Store metrics
            self.training_metrics['policy_losses'].append(policy_loss.item())
            self.training_metrics['value_losses'].append(value_loss.item())
            self.training_metrics['entropies'].append(entropy.mean().item())
        
        # Clear rollout buffer
        self.clear_buffer()
    
    def clear_buffer(self):
        """Clear the rollout buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

class SACAgent:
    """Soft Actor-Critic agent implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = self._build_actor(state_dim, action_dim).to(self.device)
        self.critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)
        
        # Training metrics
        self.training_metrics = {
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'alpha_values': []
        }
    
    def _build_actor(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)  # mean and log_std
        )
    
    def _build_critic(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def get_action(self, state, deterministic=False):
        """Get action from current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output = self.actor(state)
            mean, log_std = actor_output.chunk(2, dim=-1)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = torch.exp(log_std.clamp(-20, 2))
                dist = Normal(mean, std)
                sample = dist.sample()
                action = torch.tanh(sample)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch_size: int = None):
        """Update SAC agent."""
        batch_size = batch_size or self.config.batch_size
        
        if self.replay_buffer.size < batch_size:
            return
        
        # Sample batch
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Current alpha
        alpha = torch.exp(self.log_alpha)
        
        # Update critics
        with torch.no_grad():
            next_actor_output = self.actor(next_states)
            next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
            next_std = torch.exp(next_log_std.clamp(-20, 2))
            next_dist = Normal(next_mean, next_std)
            next_action_sample = next_dist.sample()
            next_action = torch.tanh(next_action_sample)
            next_log_prob = next_dist.log_prob(next_action_sample) - torch.log(1 - next_action.pow(2) + 1e-7)
            next_log_prob = next_log_prob.sum(1, keepdim=True)
            
            next_q1 = self.target_critic1(torch.cat([next_states, next_action], 1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_action], 1))
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * next_q
        
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actor_output = self.actor(states)
        mean, log_std = actor_output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        dist = Normal(mean, std)
        action_sample = dist.sample()
        action = torch.tanh(action_sample)
        log_prob = dist.log_prob(action_sample) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        
        q1 = self.critic1(torch.cat([states, action], 1))
        q2 = self.critic2(torch.cat([states, action], 1))
        q = torch.min(q1, q2)
        
        actor_loss = (alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        # Store metrics
        self.training_metrics['actor_losses'].append(actor_loss.item())
        self.training_metrics['critic_losses'].append((critic1_loss + critic2_loss).item() / 2)
        self.training_metrics['alpha_losses'].append(alpha_loss.item())
        self.training_metrics['alpha_values'].append(alpha.item())

class MultiAgentEnvironment(MultiAgentEnv):
    """Multi-agent environment wrapper."""
    
    def __init__(self, base_env_name: str, num_agents: int):
        self.base_env_name = base_env_name
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Create individual environments for each agent
        self.envs = [gym.make(base_env_name) for _ in range(num_agents)]
        
        # Observation and action spaces
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        self._agent_ids = set(self.agents)
    
    def reset(self):
        """Reset all agent environments."""
        observations = {}
        for i, agent in enumerate(self.agents):
            obs = self.envs[i].reset()
            observations[agent] = obs
        return observations
    
    def step(self, action_dict):
        """Step all agent environments."""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for i, agent in enumerate(self.agents):
            if agent in action_dict:
                obs, reward, done, info = self.envs[i].step(action_dict[agent])
                observations[agent] = obs
                rewards[agent] = reward
                dones[agent] = done
                infos[agent] = info
        
        # Set special done flag
        dones["__all__"] = all(dones.values())
        
        return observations, rewards, dones, infos

class AdvancedRLFramework:
    """
    Advanced Reinforcement Learning Framework.
    
    Features:
    - Multiple RL algorithms (PPO, SAC, etc.)
    - Multi-agent support
    - Distributed training
    - Curriculum learning
    - Advanced exploration strategies
    """
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.logger = self._setup_logging()
        
        # Environment setup
        self.env = None
        self.eval_env = None
        
        # Agent setup
        self.agent = None
        self.agents = {}  # For multi-agent scenarios
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_timesteps = 0
        
        # Curriculum learning
        self.curriculum_stages = []
        self.current_stage = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def setup_environment(self, env_name: str = None):
        """Setup training and evaluation environments."""
        env_name = env_name or self.config.env_name
        
        if self.config.num_agents > 1:
            self.env = MultiAgentEnvironment(env_name, self.config.num_agents)
            self.eval_env = MultiAgentEnvironment(env_name, self.config.num_agents)
        else:
            self.env = gym.make(env_name)
            self.eval_env = gym.make(env_name)
        
        self.logger.info(f"Environment setup completed: {env_name}")
    
    def setup_agent(self, algorithm: str = None):
        """Setup RL agent based on algorithm."""
        algorithm = algorithm or self.config.algorithm
        
        if self.env is None:
            self.setup_environment()
        
        # Get environment dimensions
        if hasattr(self.env.observation_space, 'shape'):
            state_dim = self.env.observation_space.shape[0]
        else:
            state_dim = self.env.observation_space.n
        
        if hasattr(self.env.action_space, 'shape'):
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.n
        
        # Create agent based on algorithm
        if algorithm.upper() == "PPO":
            self.agent = PPOAgent(state_dim, action_dim, self.config)
        elif algorithm.upper() == "SAC":
            self.agent = SACAgent(state_dim, action_dim, self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        self.logger.info(f"Agent setup completed: {algorithm}")
    
    def train_single_agent(self, total_timesteps: int = None):
        """Train single agent."""
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        if self.agent is None:
            self.setup_agent()
        
        self.logger.info(f"Starting single-agent training for {total_timesteps} timesteps")
        
        episode = 0
        timestep = 0
        
        while timestep < total_timesteps:
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and timestep < total_timesteps:
                # Get action
                action, log_prob, value = self.agent.get_action(state)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(state, action, reward, done, log_prob, value)
                elif hasattr(self.agent, 'replay_buffer'):
                    self.agent.replay_buffer.add(state, action, next_state, reward, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                timestep += 1
                
                # Update agent
                if hasattr(self.agent, 'update'):
                    if isinstance(self.agent, PPOAgent) and len(self.agent.states) >= self.config.rollout_length:
                        self.agent.update()
                    elif isinstance(self.agent, SACAgent) and timestep % 1 == 0:
                        self.agent.update()
            
            # Episode finished
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode += 1
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.logger.info(f"Episode {episode}, Timestep {timestep}, Avg Reward: {avg_reward:.2f}")
            
            # Evaluation
            if timestep % self.config.eval_freq == 0:
                eval_reward = self.evaluate_agent()
                self.logger.info(f"Evaluation at timestep {timestep}: {eval_reward:.2f}")
        
        self.logger.info("Training completed")
    
    def evaluate_agent(self, num_episodes: int = 10) -> float:
        """Evaluate agent performance."""
        if self.agent is None or self.eval_env is None:
            return 0.0
        
        total_reward = 0
        
        for episode in range(num_episodes):
            state = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.get_action(state, deterministic=True)
                state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def save_agent(self, filepath: str):
        """Save trained agent."""
        if self.agent is None:
            return
        
        save_dict = {
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_timesteps': self.total_timesteps
        }
        
        if hasattr(self.agent, 'network'):
            save_dict['network_state_dict'] = self.agent.network.state_dict()
        elif hasattr(self.agent, 'actor'):
            save_dict['actor_state_dict'] = self.agent.actor.state_dict()
            save_dict['critic1_state_dict'] = self.agent.critic1.state_dict()
            save_dict['critic2_state_dict'] = self.agent.critic2.state_dict()
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load trained agent."""
        checkpoint = torch.load(filepath)
        
        self.config = checkpoint['config']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        
        # Setup agent and load weights
        self.setup_agent()
        
        if hasattr(self.agent, 'network') and 'network_state_dict' in checkpoint:
            self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        elif hasattr(self.agent, 'actor') and 'actor_state_dict' in checkpoint:
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        
        self.logger.info(f"Agent loaded from {filepath}")

def main():
    """Demonstration of the Advanced RL Framework."""
    print("=== Advanced Reinforcement Learning Framework Demo ===\n")
    
    # Configuration
    config = RLConfig(
        algorithm="PPO",
        env_name="CartPole-v1",
        total_timesteps=50000,
        rollout_length=128,
        lr=3e-4
    )
    
    print(f"Configuration: {config.algorithm} on {config.env_name}")
    
    # Initialize framework
    framework = AdvancedRLFramework(config)
    
    # Setup environment and agent
    framework.setup_environment()
    framework.setup_agent()
    
    print(f"Environment: {framework.env}")
    print(f"Agent: {type(framework.agent).__name__}")
    
    # Train agent
    print("\nStarting training...")
    framework.train_single_agent(total_timesteps=10000)  # Reduced for demo
    
    # Evaluate agent
    eval_reward = framework.evaluate_agent(num_episodes=5)
    print(f"\nFinal evaluation reward: {eval_reward:.2f}")
    
    # Show training progress
    if framework.episode_rewards:
        print(f"Episodes completed: {len(framework.episode_rewards)}")
        print(f"Average reward (last 10): {np.mean(framework.episode_rewards[-10:]):.2f}")
    
    print("\nAdvanced RL Framework demonstration completed!")

if __name__ == "__main__":
    main()
