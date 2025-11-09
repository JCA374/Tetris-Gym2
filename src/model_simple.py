"""
Simple Feature-Based DQN for Tetris

This module implements a simple feedforward DQN that uses hand-crafted features
instead of raw board pixels. This is the approach used by most successful
Tetris DQN implementations in the literature.

Architecture: 4-8 features → Dense(64) → Dense(64) → Dense(n_actions)
Parameters: ~5,000 (vs. 2.8M in hybrid CNN)

Based on successful implementations:
- nuno-faria/tetris-ai: 4 features → 32 → 32 → 1 (~1,200 params)
- ChesterHuynh/tetrisAI: features → 64 → 64 → actions (~4,500 params)

This serves as a baseline to compare against our hybrid dual-branch architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleFeatureDQN(nn.Module):
    """
    Simple feedforward DQN using hand-crafted Tetris features.

    This network takes a small feature vector as input (typically 4-8 values)
    and outputs Q-values for each action. No convolutions, no complex architecture.

    Default features (4-vector):
    - Holes: Number of holes in the board
    - Bumpiness: Surface roughness
    - Aggregate Height: Total height of all columns
    - Completable Rows: Rows close to being cleared

    Optional additional features:
    - Max Height: Tallest column
    - Min Height: Shortest non-zero column
    - Height Variance: Variance of column heights
    - Lines Cleared: Lines just cleared
    """

    def __init__(self, obs_space, action_space, hidden_dims=[64, 64], is_target=False):
        """
        Initialize simple feature-based DQN.

        Args:
            obs_space: Observation space (should be Box with shape (n_features,))
            action_space: Action space (Discrete)
            hidden_dims: List of hidden layer sizes (default: [64, 64])
            is_target: If True, suppresses initialization message
        """
        super(SimpleFeatureDQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.is_target = is_target
        self.hidden_dims = hidden_dims

        # Validate observation space
        if len(obs_space.shape) != 1:
            raise ValueError(
                f"SimpleFeatureDQN requires 1D feature vector, got shape {obs_space.shape}")

        self.n_features = obs_space.shape[0]

        # Build network layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(self.n_features, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], self.n_actions))

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())

        if not self.is_target:
            print(f"Initialized Simple Feature DQN:")
            print(f"  Architecture: {self.n_features} → {' → '.join(map(str, hidden_dims))} → {self.n_actions}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Input features: {self.n_features}")
            print(f"  Output actions: {self.n_actions}")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input feature tensor (batch_size, n_features) or (n_features,)

        Returns:
            Q-values tensor (batch_size, n_actions)
        """
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Ensure correct dtype
        if x.dtype != torch.float32:
            x = x.float()

        # Forward through hidden layers with ReLU
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Output layer (no activation)
        x = self.layers[-1](x)

        return x


class SimpleDuelingDQN(nn.Module):
    """
    Simple dueling architecture using features.

    Separates value and advantage streams for better Q-value estimation.
    Still uses simple feature vectors as input.

    Architecture:
        Input → Shared layers → Value stream (1 output)
                             → Advantage stream (n_actions outputs)
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, obs_space, action_space, hidden_dims=[64, 64],
                 value_hidden=32, advantage_hidden=32, is_target=False):
        """
        Initialize simple dueling DQN.

        Args:
            obs_space: Observation space (1D feature vector)
            action_space: Action space (Discrete)
            hidden_dims: Shared hidden layers (default: [64, 64])
            value_hidden: Value stream hidden size (default: 32)
            advantage_hidden: Advantage stream hidden size (default: 32)
            is_target: If True, suppresses initialization message
        """
        super(SimpleDuelingDQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.is_target = is_target

        if len(obs_space.shape) != 1:
            raise ValueError(
                f"SimpleDuelingDQN requires 1D feature vector, got {obs_space.shape}")

        self.n_features = obs_space.shape[0]

        # Shared feature extraction layers
        self.shared = nn.ModuleList()
        self.shared.append(nn.Linear(self.n_features, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.shared.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        shared_output = hidden_dims[-1]

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(shared_output, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1)
        )

        # Advantage stream: estimates A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_output, advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, self.n_actions)
        )

        total_params = sum(p.numel() for p in self.parameters())

        if not self.is_target:
            print(f"Initialized Simple Dueling DQN:")
            print(f"  Shared: {self.n_features} → {' → '.join(map(str, hidden_dims))}")
            print(f"  Value stream: {shared_output} → {value_hidden} → 1")
            print(f"  Advantage stream: {shared_output} → {advantage_hidden} → {self.n_actions}")
            print(f"  Total parameters: {total_params:,}")

    def forward(self, x):
        """Forward pass through dueling architecture."""
        # Ensure batch dimension and dtype
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.float()

        # Shared feature extraction
        for layer in self.shared:
            x = F.relu(layer(x))

        # Compute value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine using dueling formula
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


def create_simple_model(obs_space, action_space, model_type="simple_dqn",
                       hidden_dims=[64, 64], is_target=False):
    """
    Factory function to create simple feature-based models.

    Args:
        obs_space: Environment observation space (must be 1D feature vector)
        action_space: Environment action space
        model_type: "simple_dqn" or "simple_dueling_dqn"
        hidden_dims: Hidden layer sizes (default: [64, 64])
        is_target: If True, suppresses initialization message

    Returns:
        Initialized simple model
    """
    model_type = model_type.lower()

    if model_type == "simple_dueling_dqn":
        return SimpleDuelingDQN(obs_space, action_space, hidden_dims=hidden_dims,
                               is_target=is_target)
    else:
        return SimpleFeatureDQN(obs_space, action_space, hidden_dims=hidden_dims,
                               is_target=is_target)


# Model testing function
def test_simple_model():
    """Test simple model creation and forward pass"""
    import gymnasium as gym

    print("Testing Simple Feature-Based Models")
    print("=" * 70)

    # Test with different feature sizes
    for n_features in [4, 6, 8]:
        print(f"\nTesting with {n_features} features:")
        print("-" * 70)

        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        action_space = gym.spaces.Discrete(8)

        for model_type in ["simple_dqn", "simple_dueling_dqn"]:
            print(f"\n{model_type.upper()}:")

            model = create_simple_model(obs_space, action_space, model_type)

            # Test single input
            single_input = torch.randn(n_features)
            single_output = model(single_input)
            assert single_output.shape == (1, 8), \
                f"Single input failed: expected (1, 8), got {single_output.shape}"

            # Test batch input
            batch_input = torch.randn(32, n_features)
            batch_output = model(batch_input)
            assert batch_output.shape == (32, 8), \
                f"Batch input failed: expected (32, 8), got {batch_output.shape}"

            print(f"  ✓ Forward pass tests passed")
            print(f"    Single input: {single_input.shape} → {single_output.shape}")
            print(f"    Batch input: {batch_input.shape} → {batch_output.shape}")

    print("\n" + "=" * 70)
    print("✅ All simple model tests passed!")


if __name__ == "__main__":
    test_simple_model()
