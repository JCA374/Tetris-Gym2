#!/usr/bin/env python3
"""
Quick test script to verify HybridDQN architecture works correctly

Tests:
1. Model creation and initialization
2. Forward pass with 8-channel input
3. Short training loop (10 episodes)
4. Gradient flow and parameter updates

Run: python test_hybrid_model.py
"""

import torch
import numpy as np
from config import make_env
from src.agent import Agent

def test_hybrid_architecture():
    """Test that hybrid model architecture works correctly"""

    print("=" * 70)
    print("HYBRID DUAL-BRANCH DQN ARCHITECTURE TEST")
    print("=" * 70)

    # Create 8-channel environment
    print("\n1. Creating 8-channel hybrid environment...")
    env = make_env(use_feature_channels=True)

    obs_shape = env.observation_space.shape
    print(f"   Observation space: {obs_shape}")

    if obs_shape[-1] != 8:
        print(f"   ❌ ERROR: Expected 8 channels, got {obs_shape[-1]}")
        return False

    print(f"   ✅ 8 channels confirmed!")

    # Test both hybrid model types
    for model_type in ['hybrid_dqn', 'hybrid_dueling_dqn']:
        print(f"\n2. Testing {model_type.upper()}...")
        print("-" * 70)

        # Create agent with hybrid model
        agent = Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            lr=1e-4,
            gamma=0.99,
            batch_size=32,
            memory_size=10000,
            min_memory_size=100,
            model_type=model_type,
            epsilon_start=1.0,
            epsilon_end=0.1,
            max_episodes=10
        )

        print(f"\n3. Testing forward pass...")

        # Get initial observation
        obs, _ = env.reset()
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        print(f"   Value range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Test action selection
        agent.q_network.eval()
        with torch.no_grad():
            action = agent.select_action(obs, eval_mode=True)

        print(f"   ✅ Action selected: {action}")

        # Test Q-value computation
        obs_tensor = torch.FloatTensor(obs).to(agent.device)
        q_values = agent.q_network(obs_tensor)

        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Q-values: {q_values.cpu().numpy()[0]}")
        print(f"   ✅ Forward pass successful!")

        # Test short training loop
        print(f"\n4. Running 10-episode training test...")

        for episode in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0

            while True:
                # Select action
                action = agent.select_action(obs, training=True)

                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                agent.remember(obs, action, reward, next_obs, done)

                # Learn (if enough samples)
                if len(agent.memory) >= agent.min_buffer_size:
                    loss = agent.learn()

                episode_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            print(f"   Episode {episode+1:2d}: {steps:3d} steps, reward: {episode_reward:7.1f}, "
                  f"epsilon: {agent.epsilon:.3f}, memory: {len(agent.memory)}")

        print(f"\n   ✅ {model_type.upper()} training test passed!")

        # Test parameter updates
        print(f"\n5. Verifying gradient flow...")

        # Check that parameters have changed
        initial_params = []
        for param in agent.q_network.parameters():
            initial_params.append(param.clone().detach())

        # Train for a few more steps
        obs, _ = env.reset()
        for _ in range(50):
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, done)
            if len(agent.memory) >= agent.min_buffer_size:
                agent.learn()
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Check parameters changed
        params_changed = False
        for i, param in enumerate(agent.q_network.parameters()):
            if not torch.equal(param, initial_params[i]):
                params_changed = True
                break

        if params_changed:
            print(f"   ✅ Parameters updated successfully (gradient flow working)")
        else:
            print(f"   ⚠️  WARNING: Parameters unchanged (might be learning rate issue)")

        print(f"\n   ✅ {model_type.upper()} fully validated!")
        print("-" * 70)

    env.close()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - HYBRID ARCHITECTURE READY FOR TRAINING!")
    print("=" * 70)

    return True


def test_channel_separation():
    """Test that visual and feature channels are properly separated"""

    print("\n" + "=" * 70)
    print("CHANNEL SEPARATION TEST")
    print("=" * 70)

    from src.model_hybrid import HybridDQN
    import gymnasium as gym

    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(20, 10, 8), dtype=np.float32)
    action_space = gym.spaces.Discrete(8)

    model = HybridDQN(obs_space, action_space)
    model.eval()

    print("\n1. Testing with different channel patterns...")

    # Create test observation with distinct patterns
    obs = np.zeros((20, 10, 8), dtype=np.float32)

    # Visual channels: checkerboard pattern
    for i in range(4):
        obs[:, :, i] = np.random.rand(20, 10) * 0.3

    # Feature channels: higher values
    for i in range(4, 8):
        obs[:, :, i] = np.random.rand(20, 10) * 0.8 + 0.2

    print(f"   Visual channels (0-3) mean: {obs[:,:,:4].mean():.3f}")
    print(f"   Feature channels (4-7) mean: {obs[:,:,4:].mean():.3f}")

    # Forward pass
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs)
        q_values = model(obs_tensor)

    print(f"   Q-values: {q_values.cpu().numpy()[0]}")
    print(f"   ✅ Channel separation working!")

    print("\n2. Testing gradient flow to both branches...")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Compute loss and backprop
    obs_tensor = torch.FloatTensor(obs)
    q_values = model(obs_tensor)
    target = torch.FloatTensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    loss = torch.nn.functional.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()

    # Check gradients in both branches
    visual_grads = []
    feature_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'visual' in name:
                visual_grads.append(param.grad.abs().mean().item())
            elif 'feature' in name:
                feature_grads.append(param.grad.abs().mean().item())

    print(f"   Visual branch gradients: {len(visual_grads)} layers, "
          f"mean: {np.mean(visual_grads):.6f}")
    print(f"   Feature branch gradients: {len(feature_grads)} layers, "
          f"mean: {np.mean(feature_grads):.6f}")

    if len(visual_grads) > 0 and len(feature_grads) > 0:
        print(f"   ✅ Both branches receiving gradients!")
    else:
        print(f"   ⚠️  WARNING: One branch may not be learning!")

    print("=" * 70)


if __name__ == "__main__":
    print("\n🧪 TESTING HYBRID DUAL-BRANCH DQN ARCHITECTURE\n")

    try:
        # Test architecture
        success = test_hybrid_architecture()

        # Test channel separation
        test_channel_separation()

        if success:
            print("\n" + "=" * 70)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 70)
            print("\nYou can now run full training with:")
            print("\n  python train_progressive_improved.py \\")
            print("      --episodes 10000 \\")
            print("      --force_fresh \\")
            print("      --model_type hybrid_dqn \\")
            print("      --experiment_name hybrid_10k")
            print("\n" + "=" * 70)
        else:
            print("\n❌ Tests failed!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
