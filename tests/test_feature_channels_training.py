"""
Quick test to verify feature channels work with training loop.

This script tests that:
1. Environment creates with correct observation shape
2. Agent accepts the new observation space
3. Training loop runs without errors
4. Feature channels are actually being computed

Run with: python tests/test_feature_channels_training.py

Author: Claude Code
Date: 2025-11-05
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import make_env
from src.agent import Agent
from src.progressive_reward_improved import ImprovedProgressiveRewardShaper


def test_4_channel_mode():
    """Test original 4-channel visual-only mode."""
    print("\n" + "="*80)
    print("Test 1: 4-Channel Visual-Only Mode")
    print("="*80)

    env = make_env(use_complete_vision=True, use_feature_channels=False)

    # Verify observation space
    assert env.observation_space.shape == (20, 10, 4), \
        f"Expected (20, 10, 4), got {env.observation_space.shape}"
    assert env.observation_space.dtype == np.float32, \
        f"Expected float32, got {env.observation_space.dtype}"

    # Test reset
    obs, info = env.reset()
    assert obs.shape == (20, 10, 4), f"Observation shape incorrect: {obs.shape}"
    assert obs.min() >= 0 and obs.max() <= 1, \
        f"Values out of range: [{obs.min()}, {obs.max()}]"

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape == (20, 10, 4), f"Step observation shape incorrect: {obs.shape}"

    env.close()

    print("✅ 4-channel mode: PASSED")
    return True


def test_8_channel_mode():
    """Test new 8-channel hybrid mode."""
    print("\n" + "="*80)
    print("Test 2: 8-Channel Hybrid Mode")
    print("="*80)

    env = make_env(use_complete_vision=True, use_feature_channels=True)

    # Verify observation space
    assert env.observation_space.shape == (20, 10, 8), \
        f"Expected (20, 10, 8), got {env.observation_space.shape}"
    assert env.observation_space.dtype == np.float32, \
        f"Expected float32, got {env.observation_space.dtype}"

    # Test reset
    obs, info = env.reset()
    assert obs.shape == (20, 10, 8), f"Observation shape incorrect: {obs.shape}"
    assert obs.min() >= 0 and obs.max() <= 1, \
        f"Values out of range: [{obs.min()}, {obs.max()}]"

    # Verify visual channels (0-3) look reasonable
    board_channel = obs[:, :, 0]
    active_channel = obs[:, :, 1]
    print(f"   Board channel: {board_channel.sum():.0f} filled cells")
    print(f"   Active piece: {active_channel.sum():.0f} cells")

    # Verify feature channels (4-7) exist
    holes_channel = obs[:, :, 4]
    height_channel = obs[:, :, 5]
    bump_channel = obs[:, :, 6]
    wells_channel = obs[:, :, 7]

    print(f"   Holes heatmap: min={holes_channel.min():.3f}, max={holes_channel.max():.3f}")
    print(f"   Height map: min={height_channel.min():.3f}, max={height_channel.max():.3f}")
    print(f"   Bumpiness map: min={bump_channel.min():.3f}, max={bump_channel.max():.3f}")
    print(f"   Wells map: min={wells_channel.min():.3f}, max={wells_channel.max():.3f}")

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape == (20, 10, 8), f"Step observation shape incorrect: {obs.shape}"

    env.close()

    print("✅ 8-channel mode: PASSED")
    return True


def test_agent_compatibility():
    """Test that agent works with both observation spaces."""
    print("\n" + "="*80)
    print("Test 3: Agent Compatibility")
    print("="*80)

    # Test with 4-channel
    print("\n📍 Testing agent with 4-channel observations...")
    env4 = make_env(use_feature_channels=False)
    agent4 = Agent(
        obs_space=env4.observation_space,
        action_space=env4.action_space,
        lr=5e-4,
        batch_size=32,
    )

    obs, _ = env4.reset()
    action = agent4.select_action(obs)
    assert action in range(env4.action_space.n), f"Invalid action: {action}"
    print(f"   ✅ Agent selected valid action: {action}")

    env4.close()

    # Test with 8-channel
    print("\n📍 Testing agent with 8-channel observations...")
    env8 = make_env(use_feature_channels=True)
    agent8 = Agent(
        obs_space=env8.observation_space,
        action_space=env8.action_space,
        lr=5e-4,
        batch_size=32,
    )

    obs, _ = env8.reset()
    action = agent8.select_action(obs)
    assert action in range(env8.action_space.n), f"Invalid action: {action}"
    print(f"   ✅ Agent selected valid action: {action}")

    env8.close()

    print("\n✅ Agent compatibility: PASSED")
    return True


def test_training_loop():
    """Test that training loop works with feature channels."""
    print("\n" + "="*80)
    print("Test 4: Training Loop (10 episodes)")
    print("="*80)

    env = make_env(use_feature_channels=True)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=5e-4,
        batch_size=32,
        memory_size=10000,
        min_memory_size=100,
    )

    reward_shaper = ImprovedProgressiveRewardShaper()

    print(f"\nTraining with observation space: {env.observation_space.shape}")

    for episode in range(10):
        reward_shaper.update_episode(episode)

        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        pieces_placed = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            if action == 5:  # Hard drop
                pieces_placed += 1

            info['pieces_placed'] = pieces_placed
            info['steps'] = episode_steps

            # Apply reward shaping
            shaped_reward = reward_shaper.calculate_reward(obs, action, reward, done, info)

            # Store experience
            agent.remember(obs, action, shaped_reward, next_obs, done, info, reward)

            # Learn
            if len(agent.memory) >= agent.min_buffer_size and episode_steps % 4 == 0:
                agent.learn()

            episode_reward += shaped_reward
            episode_steps += 1
            obs = next_obs

        print(f"   Episode {episode+1:2d}: {episode_steps:3d} steps, "
              f"reward={episode_reward:7.1f}, ε={agent.epsilon:.3f}")

    env.close()

    print("\n✅ Training loop: PASSED")
    print(f"   Memory size: {len(agent.memory)}")
    print(f"   Final epsilon: {agent.epsilon:.3f}")

    return True


def test_feature_channels_actually_used():
    """Test that feature channels contain meaningful data."""
    print("\n" + "="*80)
    print("Test 5: Feature Channels Contain Data")
    print("="*80)

    env = make_env(use_feature_channels=True)

    # Play until we get some pieces on board
    obs, _ = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    # Check that feature channels have non-zero values
    holes_sum = obs[:, :, 4].sum()
    height_sum = obs[:, :, 5].sum()
    bump_sum = obs[:, :, 6].sum()
    wells_sum = obs[:, :, 7].sum()

    print(f"   Holes channel sum: {holes_sum:.2f}")
    print(f"   Height channel sum: {height_sum:.2f}")
    print(f"   Bumpiness channel sum: {bump_sum:.2f}")
    print(f"   Wells channel sum: {wells_sum:.2f}")

    # At least one should be non-zero
    assert (holes_sum + height_sum + bump_sum + wells_sum) > 0, \
        "All feature channels are zero - features not being computed!"

    env.close()

    print("\n✅ Feature channels contain data: PASSED")
    return True


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*80)
    print("🧪 FEATURE CHANNELS - INTEGRATION TESTS")
    print("="*80)
    print("Testing that feature channels work with training pipeline...")

    tests = [
        ("4-Channel Mode", test_4_channel_mode),
        ("8-Channel Mode", test_8_channel_mode),
        ("Agent Compatibility", test_agent_compatibility),
        ("Training Loop", test_training_loop),
        ("Feature Channels Data", test_feature_channels_actually_used),
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                failed_tests.append(name)
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            failed += 1
            failed_tests.append(name)

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed > 0:
        print(f"\n❌ Failed tests:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
        print("\n" + "="*80)
        return 1
    else:
        print("\n✅ ALL TESTS PASSED!")
        print("\n" + "="*80)
        print("🎉 Feature channels are ready for training!")
        print("\nNext steps:")
        print("  1. Visualize: python visualize_features.py")
        print("  2. Quick train: python train_progressive_improved.py --episodes 10")
        print("  3. Full comparison: Run 2000-episode experiments")
        print("="*80)
        return 0


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
