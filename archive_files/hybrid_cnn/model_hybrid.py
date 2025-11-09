"""
Hybrid Dual-Branch DQN Architecture for Tetris

This architecture properly separates visual and feature processing:
- Visual branch (channels 0-3): Board, Active piece, Holder, Queue
- Feature branch (channels 4-7): Holes, Heights, Bumpiness, Wells

The key insight: Visual data needs different processing than pre-computed features.
Visual CNNs learn spatial patterns (edges, shapes), while feature CNNs just need
to understand spatial distribution of already-meaningful values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridDQN(nn.Module):
    """
    Dual-branch DQN that processes visual and feature channels separately

    Architecture:
        Input (20, 10, 8)
        ├─→ Visual CNN (channels 0-3) → visual features
        └─→ Feature CNN (channels 4-7) → feature features
                ↓
            Concatenate
                ↓
        Fully-Connected Layers
                ↓
            Q-Values (n_actions)

    This architecture is based on research showing that hybrid visual+feature
    approaches work best when features are processed separately before fusion.
    """

    def __init__(self, obs_space, action_space, is_target=False):
        super(HybridDQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.is_target = is_target

        # Validate observation space
        if len(obs_space.shape) != 3:
            raise ValueError(
                f"HybridDQN requires 3D observation space (H, W, C), got {obs_space.shape}")

        h, w, c = obs_space.shape

        if c != 8:
            raise ValueError(
                f"HybridDQN requires 8 channels (4 visual + 4 features), got {c}")

        self.height = h
        self.width = w

        # ===== VISUAL BRANCH (Channels 0-3) =====
        # Processes: Board, Active piece, Holder, Queue
        # Uses standard CNN architecture for spatial pattern recognition
        self.visual_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.visual_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.visual_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ===== FEATURE BRANCH (Channels 4-7) =====
        # Processes: Holes, Heights, Bumpiness, Wells
        # Simpler architecture - features are already meaningful
        self.feature_conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.feature_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        # Calculate output sizes for both branches
        visual_size = self._get_visual_output_size()
        feature_size = self._get_feature_output_size()
        combined_size = visual_size + feature_size

        # ===== COMBINED FULLY-CONNECTED LAYERS =====
        self.fc1 = nn.Linear(combined_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.n_actions)

        # Dropout for regularization (0.1 for RL, not 0.3)
        self.dropout = nn.Dropout(0.1)

        if not self.is_target:
            print(f"Initialized Hybrid Dual-Branch DQN:")
            print(f"  Visual branch:  4 channels → {visual_size} features")
            print(f"  Feature branch: 4 channels → {feature_size} features")
            print(f"  Combined: {combined_size} → 512 → 256 → {self.n_actions}")

    def _get_visual_output_size(self):
        """Calculate output size of visual CNN branch"""
        # Simulate forward pass
        x = torch.zeros(1, 4, self.height, self.width)
        x = F.relu(self.visual_conv1(x))
        x = F.relu(self.visual_conv2(x))
        x = F.relu(self.visual_conv3(x))
        return x.numel()

    def _get_feature_output_size(self):
        """Calculate output size of feature CNN branch"""
        # Simulate forward pass
        x = torch.zeros(1, 4, self.height, self.width)
        x = F.relu(self.feature_conv1(x))
        x = F.relu(self.feature_conv2(x))
        return x.numel()

    def forward(self, x):
        """
        Forward pass through dual-branch architecture

        Args:
            x: Input tensor (batch, height, width, channels) or (height, width, channels)
               Expected shape: (*, 20, 10, 8)

        Returns:
            Q-values tensor (batch, n_actions)
        """
        # Ensure batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Convert from (batch, H, W, C) to (batch, C, H, W) if needed
        if x.shape[-1] == 8:  # Channels last
            x = x.permute(0, 3, 1, 2)

        # Validate shape
        if x.shape[1] != 8:
            raise ValueError(f"Expected 8 channels, got {x.shape[1]}")

        # ===== SPLIT INTO VISUAL AND FEATURE CHANNELS =====
        visual_input = x[:, :4, :, :]    # Channels 0-3
        feature_input = x[:, 4:, :, :]   # Channels 4-7

        # ===== VISUAL BRANCH =====
        v = F.relu(self.visual_conv1(visual_input))
        v = F.relu(self.visual_conv2(v))
        v = F.relu(self.visual_conv3(v))
        v = v.contiguous().view(v.size(0), -1)  # Flatten

        # ===== FEATURE BRANCH =====
        f = F.relu(self.feature_conv1(feature_input))
        f = F.relu(self.feature_conv2(f))
        f = f.contiguous().view(f.size(0), -1)  # Flatten

        # ===== CONCATENATE BRANCHES =====
        combined = torch.cat([v, f], dim=1)

        # ===== FULLY-CONNECTED LAYERS =====
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class HybridDuelingDQN(nn.Module):
    """
    Hybrid Dual-Branch Dueling DQN Architecture

    Combines:
    - Dual-branch processing (visual + feature separation)
    - Dueling architecture (value + advantage streams)

    Best of both worlds for Tetris RL.
    """

    def __init__(self, obs_space, action_space, is_target=False):
        super(HybridDuelingDQN, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.is_target = is_target

        # Validate observation space
        if len(obs_space.shape) != 3:
            raise ValueError(
                f"HybridDuelingDQN requires 3D observation space, got {obs_space.shape}")

        h, w, c = obs_space.shape

        if c != 8:
            raise ValueError(
                f"HybridDuelingDQN requires 8 channels, got {c}")

        self.height = h
        self.width = w

        # ===== VISUAL BRANCH (Channels 0-3) =====
        self.visual_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.visual_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.visual_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ===== FEATURE BRANCH (Channels 4-7) =====
        self.feature_conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.feature_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        # Calculate combined feature size
        visual_size = self._get_visual_output_size()
        feature_size = self._get_feature_output_size()
        self.feature_size = visual_size + feature_size

        # ===== DUELING STREAMS =====
        # Value stream (scalar state value)
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Advantage stream (per-action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.n_actions)
        )

        if not self.is_target:
            print(f"Initialized Hybrid Dual-Branch Dueling DQN:")
            print(f"  Visual branch:  4 channels → {visual_size} features")
            print(f"  Feature branch: 4 channels → {feature_size} features")
            print(f"  Combined: {self.feature_size} → Value + {self.n_actions} Advantages")

    def _get_visual_output_size(self):
        """Calculate output size of visual CNN branch"""
        x = torch.zeros(1, 4, self.height, self.width)
        x = F.relu(self.visual_conv1(x))
        x = F.relu(self.visual_conv2(x))
        x = F.relu(self.visual_conv3(x))
        return x.numel()

    def _get_feature_output_size(self):
        """Calculate output size of feature CNN branch"""
        x = torch.zeros(1, 4, self.height, self.width)
        x = F.relu(self.feature_conv1(x))
        x = F.relu(self.feature_conv2(x))
        return x.numel()

    def forward(self, x):
        """Forward pass through hybrid dueling architecture"""
        # Ensure batch dimension and correct format
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] == 8:  # Channels last → channels first
            x = x.permute(0, 3, 1, 2)

        # Split channels
        visual_input = x[:, :4, :, :]
        feature_input = x[:, 4:, :, :]

        # Process visual branch
        v = F.relu(self.visual_conv1(visual_input))
        v = F.relu(self.visual_conv2(v))
        v = F.relu(self.visual_conv3(v))
        v = v.contiguous().view(v.size(0), -1)

        # Process feature branch
        f = F.relu(self.feature_conv1(feature_input))
        f = F.relu(self.feature_conv2(f))
        f = f.contiguous().view(f.size(0), -1)

        # Concatenate
        combined = torch.cat([v, f], dim=1)

        # Dueling streams
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


def create_hybrid_model(obs_space, action_space, model_type="hybrid_dqn", is_target=False):
    """
    Factory function to create hybrid dual-branch models

    Args:
        obs_space: Environment observation space (must be (H, W, 8))
        action_space: Environment action space
        model_type: "hybrid_dqn" or "hybrid_dueling_dqn"
        is_target: If True, suppresses initialization message

    Returns:
        Initialized hybrid model
    """
    if model_type.lower() == "hybrid_dueling_dqn":
        return HybridDuelingDQN(obs_space, action_space, is_target=is_target)
    else:
        return HybridDQN(obs_space, action_space, is_target=is_target)


# Model testing function
def test_hybrid_model():
    """Test hybrid model creation and forward pass"""
    import gymnasium as gym

    print("Testing Hybrid Dual-Branch Model Architectures...")
    print("=" * 70)

    # Create observation space for 8-channel Tetris
    obs_space = gym.spaces.Box(
        low=0.0, high=1.0,
        shape=(20, 10, 8),
        dtype=np.float32
    )
    action_space = gym.spaces.Discrete(8)

    print(f"\nObservation space: {obs_space.shape}")
    print(f"Action space: {action_space.n} actions\n")

    # Test both hybrid model types
    for model_type in ["hybrid_dqn", "hybrid_dueling_dqn"]:
        print(f"\nTesting {model_type.upper()}:")
        print("-" * 70)

        model = create_hybrid_model(obs_space, action_space, model_type)

        # Create dummy input (batch of 4)
        batch_size = 4
        dummy_input = torch.randn((batch_size,) + obs_space.shape)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"\n  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected:     ({batch_size}, {action_space.n})")

        # Validate output
        assert output.shape == (batch_size, action_space.n), \
            f"Expected {(batch_size, action_space.n)}, got {output.shape}"

        # Test single observation (no batch dimension)
        single_input = torch.randn(obs_space.shape)
        single_output = model(single_input)
        assert single_output.shape == (1, action_space.n), \
            f"Single input failed: expected {(1, action_space.n)}, got {single_output.shape}"

        print(f"  ✅ {model_type.upper()} tests passed!")

    print("\n" + "=" * 70)
    print("✅ All hybrid model tests passed!")
    print("=" * 70)

    # Count parameters
    print("\nModel Parameter Counts:")
    for model_type in ["hybrid_dqn", "hybrid_dueling_dqn"]:
        model = create_hybrid_model(obs_space, action_space, model_type, is_target=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {model_type:20s}: {total_params:,} parameters ({trainable_params:,} trainable)")


if __name__ == "__main__":
    test_hybrid_model()
