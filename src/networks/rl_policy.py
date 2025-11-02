#!/usr/bin/env python3
"""
Reinforcement Learning Policy Network

This module implements the RL policy network that provides compensating
torques based on tracking errors and meta-PID predictions.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3


class RLPolicyNetwork(nn.Module):
    """
    RL Policy Network for Torque Compensation
    
    Takes augmented state (robot state + PID gains + tracking error) and
    outputs compensating torque adjustments.
    
    Args:
        state_dim (int): Dimension of augmented state
        action_dim (int): Dimension of action space (number of joints)
        hidden_layers (list): List of hidden layer dimensions
        activation (str): Activation function ('relu', 'tanh', 'elu')
    """
    
    def __init__(self, state_dim, action_dim, 
                 hidden_layers=[256, 128], activation='relu'):
        super(RLPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build actor network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer with tanh to bound actions
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.actor = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state (torch.Tensor): Augmented state [batch_size, state_dim]
        
        Returns:
            torch.Tensor: Compensating torque [batch_size, action_dim]
        """
        return self.actor(state)


class ValueNetwork(nn.Module):
    """
    Value Network for Critic
    
    Estimates the value of a given state for policy optimization.
    """
    
    def __init__(self, state_dim, hidden_layers=[256, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
    
    def forward(self, state):
        """Compute state value"""
        return self.value_net(state)


def create_rl_agent(env, algorithm='PPO', learning_rate=3e-4, **kwargs):
    """
    Create RL agent using Stable-Baselines3
    
    Args:
        env: Gymnasium environment
        algorithm (str): RL algorithm ('PPO', 'SAC', 'TD3')
        learning_rate (float): Learning rate
        **kwargs: Additional algorithm-specific arguments
    
    Returns:
        Agent instance from Stable-Baselines3
    """
    if algorithm == 'PPO':
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=kwargs.get('n_steps', 2048),
            batch_size=kwargs.get('batch_size', 64),
            n_epochs=kwargs.get('n_epochs', 10),
            gamma=kwargs.get('gamma', 0.99),
            gae_lambda=kwargs.get('gae_lambda', 0.95),
            clip_range=kwargs.get('clip_range', 0.2),
            ent_coef=kwargs.get('ent_coef', 0.01),
            vf_coef=kwargs.get('vf_coef', 0.5),
            max_grad_norm=kwargs.get('max_grad_norm', 0.5),
            verbose=kwargs.get('verbose', 1),
            tensorboard_log=kwargs.get('tensorboard_log', './logs/')
        )
    
    elif algorithm == 'SAC':
        agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=kwargs.get('buffer_size', 1000000),
            learning_starts=kwargs.get('learning_starts', 10000),
            batch_size=kwargs.get('batch_size', 256),
            tau=kwargs.get('tau', 0.005),
            gamma=kwargs.get('gamma', 0.99),
            verbose=kwargs.get('verbose', 1),
            tensorboard_log=kwargs.get('tensorboard_log', './logs/')
        )
    
    elif algorithm == 'TD3':
        agent = TD3(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=kwargs.get('buffer_size', 1000000),
            learning_starts=kwargs.get('learning_starts', 10000),
            batch_size=kwargs.get('batch_size', 256),
            tau=kwargs.get('tau', 0.005),
            gamma=kwargs.get('gamma', 0.99),
            policy_delay=kwargs.get('policy_delay', 2),
            verbose=kwargs.get('verbose', 1),
            tensorboard_log=kwargs.get('tensorboard_log', './logs/')
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def load_rl_policy(model_path, env=None):
    """
    Load trained RL policy
    
    Args:
        model_path (str): Path to saved model (.zip file)
        env: Environment instance (optional, for retraining)
    
    Returns:
        Loaded RL agent
    """
    # Determine algorithm from filename or try loading
    if 'ppo' in model_path.lower():
        agent = PPO.load(model_path, env=env)
    elif 'sac' in model_path.lower():
        agent = SAC.load(model_path, env=env)
    elif 'td3' in model_path.lower():
        agent = TD3.load(model_path, env=env)
    else:
        # Try PPO as default
        try:
            agent = PPO.load(model_path, env=env)
        except:
            try:
                agent = SAC.load(model_path, env=env)
            except:
                agent = TD3.load(model_path, env=env)
    
    return agent


def evaluate_policy(agent, env, n_episodes=100, deterministic=True):
    """
    Evaluate trained policy
    
    Args:
        agent: Trained RL agent
        env: Evaluation environment
        n_episodes (int): Number of evaluation episodes
        deterministic (bool): Use deterministic policy
    
    Returns:
        dict: Evaluation metrics (mean reward, std, episode lengths)
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    import numpy as np
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    return results


if __name__ == '__main__':
    # Test the network
    print("Testing RLPolicyNetwork...")
    
    # Create model
    state_dim = 40  # Example: 9 joints * 4 features + 3 PID gains
    action_dim = 9  # 9 joints
    model = RLPolicyNetwork(state_dim, action_dim)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, state_dim)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("✓ Test passed!")

