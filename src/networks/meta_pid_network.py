#!/usr/bin/env python3
"""
Meta-Learning PID Network

This module implements the meta-learning network that predicts PID gains
based on robot state features.
"""

import torch
import torch.nn as nn


class MetaPIDNetwork(nn.Module):
    """
    Meta-Learning PID Predictor Network
    
    Predicts PID gains [K_p, K_i, K_d] for each joint based on robot state.
    
    Args:
        input_dim (int): Dimension of input features (default: 4 per joint)
                        [q, q_dot, q_ref, q_ref_dot]
        hidden_dim (int): Dimension of hidden layers (default: 64)
        output_dim (int): Dimension of output (default: 3 for [K_p, K_i, K_d])
        n_layers (int): Number of hidden layers (default: 2)
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, n_layers=2):
        super(MetaPIDNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build encoder layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Add hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
        # PID gain prediction head
        self.pid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive PID gains
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Predicted PID gains [batch_size, output_dim]
        """
        features = self.encoder(x)
        pid_gains = self.pid_head(features)
        return pid_gains


class SimplePIDPredictor(nn.Module):
    """
    Simplified PID Predictor (Backward Compatibility)
    
    This is a simplified version used in the paper for faster training.
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(SimplePIDPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


def load_meta_pid_model(model_path, device='cpu'):
    """
    Load pre-trained meta-learning PID model
    
    Args:
        model_path (str): Path to model checkpoint (.pth file)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        tuple: (model, X_mean, X_std, y_mean, y_std)
            - model: Loaded PyTorch model
            - X_mean, X_std: Input normalization parameters
            - y_mean, y_std: Output normalization parameters
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine model type from checkpoint
    if 'input_dim' in checkpoint:
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', 64)
        output_dim = checkpoint['output_dim']
        model = MetaPIDNetwork(input_dim, hidden_dim, output_dim)
    else:
        # Use simple predictor as default
        model = SimplePIDPredictor(input_dim=4, hidden_dim=64, output_dim=3)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load normalization parameters
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    
    return model, X_mean, X_std, y_mean, y_std


def predict_pid_gains(model, robot_features, X_mean, X_std, y_mean, y_std):
    """
    Predict PID gains using meta-learning model
    
    Args:
        model: Trained meta-learning model
        robot_features (np.ndarray): Robot state features [q, q_dot, q_ref, q_ref_dot]
        X_mean, X_std: Input normalization parameters
        y_mean, y_std: Output normalization parameters
    
    Returns:
        np.ndarray: Predicted PID gains [K_p, K_i, K_d]
    """
    import numpy as np
    
    # Normalize input
    x_normalized = (robot_features - X_mean) / (X_std + 1e-8)
    
    # Convert to tensor
    x_tensor = torch.FloatTensor(x_normalized).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        y_normalized = model(x_tensor).cpu().numpy()[0]
    
    # Denormalize output
    pid_gains = y_normalized * y_std + y_mean
    
    return pid_gains


if __name__ == '__main__':
    # Test the network
    print("Testing MetaPIDNetwork...")
    
    # Create model
    model = MetaPIDNetwork(input_dim=4, hidden_dim=64, output_dim=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 4)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("✓ Test passed!")

