"""
Test 2: Environment Wrapper - Verify info dict passes through correctly
========================================================================

This test checks if the FeatureVectorWrapper preserves the info dict,
especially the 'number_of_lines' field that's critical for rewards.
"""

import sys
import numpy as np
import tetris_gymnasium.envs  # Required to register environment
sys.path.insert(0, '/home/jonas/Code/Tetris-Gym2')

from src.env_feature_vector import make_feature_vector_env

def test_wrapper_info_passthrough():
    """Test if wrapper preserves info dict"""
    print("=" * 70)
    print("TEST 2A: Wrapper Info Dict Passthrough")
    print("=" * 70)

    env = make_feature_vector_env()
    print(f"✓ Created wrapped environment")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\n✓ Reset environment")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation type: {type(obs)}")
    print(f"  Info dict keys: {list(info.keys())}")
    print(f"  Info dict contents:")
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"    {key}: array shape {value.shape}")
        else:
            print(f"    {key}: {value}")

    # Take steps and look for line clears
    print(f"\n🎮 Taking 1000 steps looking for line clears...")
    lines_cleared_episodes = []
    total_lines = 0

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        lines = info.get('number_of_lines', 0)
        if lines > 0:
            total_lines += lines
            print(f"\n  🎉 Step {step}: CLEARED {lines} LINE(S)!")
            print(f"     Reward from env: {reward}")
            print(f"     Info dict: {info}")
            print(f"     Observation shape: {obs.shape}")
            print(f"     Observation sample: {obs[:5]}")

        if terminated or truncated:
            lines_cleared_episodes.append(total_lines)
            print(f"\n  Episode ended at step {step}")
            print(f"    Total lines this episode: {total_lines}")
            obs, info = env.reset()
            total_lines = 0

    print(f"\n📊 Summary:")
    print(f"  Total episodes completed: {len(lines_cleared_episodes)}")
    print(f"  Episodes with lines: {sum(1 for x in lines_cleared_episodes if x > 0)}")
    print(f"  Total lines cleared: {sum(lines_cleared_episodes)}")

    if sum(lines_cleared_episodes) == 0:
        print(f"  ⚠️  WARNING: No lines cleared through wrapper!")
    else:
        print(f"  ✅ Wrapper preserves line clearing!")

    env.close()
    return sum(lines_cleared_episodes) > 0


def test_wrapper_observation_correctness():
    """Test if wrapper extracts features correctly"""
    print("\n" + "=" * 70)
    print("TEST 2B: Wrapper Feature Extraction")
    print("=" * 70)

    env = make_feature_vector_env()
    obs, info = env.reset()

    print(f"Feature vector (17 dimensions):")
    feature_names = [
        'aggregate_height', 'holes', 'bumpiness', 'wells',
        'col_0', 'col_1', 'col_2', 'col_3', 'col_4',
        'col_5', 'col_6', 'col_7', 'col_8', 'col_9',
        'max_height', 'min_height', 'std_height'
    ]

    for i, (name, value) in enumerate(zip(feature_names, obs)):
        print(f"  [{i:2d}] {name:16s}: {value:.4f}")

    # Check if values are in [0, 1] range (normalized)
    print(f"\n✓ Feature range check:")
    print(f"  Min value: {obs.min():.4f}")
    print(f"  Max value: {obs.max():.4f}")
    print(f"  Mean value: {obs.mean():.4f}")

    if obs.min() < -0.01 or obs.max() > 1.01:
        print(f"  ⚠️  WARNING: Features not properly normalized!")
    else:
        print(f"  ✅ Features properly normalized to [0, 1]")

    # Take several steps and check features change
    print(f"\n✓ Feature dynamics check (taking 10 steps):")
    prev_obs = obs.copy()
    changes = []

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        change = np.abs(obs - prev_obs).sum()
        changes.append(change)
        print(f"  Step {step+1}: Total change = {change:.4f}")

        prev_obs = obs.copy()

        if terminated or truncated:
            obs, info = env.reset()
            prev_obs = obs.copy()

    avg_change = np.mean(changes)
    print(f"\n  Average change per step: {avg_change:.4f}")

    if avg_change < 0.01:
        print(f"  ⚠️  WARNING: Features barely changing! Possible extraction bug.")
    else:
        print(f"  ✅ Features changing normally")

    env.close()


def test_wrapper_vs_base_env():
    """Compare wrapped env to base env side-by-side"""
    print("\n" + "=" * 70)
    print("TEST 2C: Wrapped vs Base Environment Comparison")
    print("=" * 70)

    import gymnasium as gym

    base_env = gym.make('tetris_gymnasium/Tetris', render_mode=None)
    wrapped_env = make_feature_vector_env()

    # Same seed
    base_env.reset(seed=42)
    wrapped_env.reset(seed=42)

    print("Taking 100 steps in parallel...")
    base_lines = 0
    wrapped_lines = 0

    for step in range(100):
        action = base_env.action_space.sample()

        # Step both environments
        _, base_reward, base_term, base_trunc, base_info = base_env.step(action)
        _, wrap_reward, wrap_term, wrap_trunc, wrap_info = wrapped_env.step(action)

        base_lines += base_info.get('number_of_lines', 0)
        wrapped_lines += wrap_info.get('number_of_lines', 0)

        if base_info.get('number_of_lines', 0) > 0:
            print(f"\n  Step {step}: Base env cleared {base_info['number_of_lines']} line(s)")
            print(f"             Wrapped env: {wrap_info.get('number_of_lines', 'MISSING')}")

        if base_term or base_trunc or wrap_term or wrap_trunc:
            break

    print(f"\n📊 Comparison:")
    print(f"  Base env lines: {base_lines}")
    print(f"  Wrapped env lines: {wrapped_lines}")

    if base_lines != wrapped_lines:
        print(f"  ⚠️  WARNING: Line counts don't match! Wrapper may be broken.")
    else:
        print(f"  ✅ Line counts match!")

    base_env.close()
    wrapped_env.close()


if __name__ == "__main__":
    print("🔍 DEBUGGING PLAN - TEST 2: Environment Wrapper")
    print("=" * 70)
    print("This test verifies that the wrapper preserves info dict correctly.\n")

    # Run tests
    wrapper_works = test_wrapper_info_passthrough()
    test_wrapper_observation_correctness()
    test_wrapper_vs_base_env()

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if wrapper_works:
        print("✅ PASS: Wrapper preserves info dict correctly")
        print("   The wrapper is working correctly.")
        print("   Issue likely in agent or reward function.")
    else:
        print("❌ FAIL: Wrapper breaks info dict")
        print("   This is a critical issue with the wrapper.")
        print("   Check FeatureVectorWrapper implementation.")
    print("=" * 70)
