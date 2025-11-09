"""
Test 4: Reward Function - Verify lines_cleared info reaches reward calculation
===============================================================================

This test checks if the reward function receives correct info and calculates
rewards properly, especially for line clears.
"""

import sys
import numpy as np
import tetris_gymnasium.envs  # Required to register environment
sys.path.insert(0, '/home/jonas/Code/Tetris-Gym2')

from src.env_feature_vector import make_feature_vector_env

def simple_reward(env_reward, info):
    """
    Copy of the reward function from train_feature_vector.py
    """
    lines = info.get('number_of_lines', 0)

    # Base reward: positive for surviving
    reward = 1.0

    # Huge bonus for line clears
    if lines > 0:
        reward += lines * 100

    # Penalize bad board states (if available in info)
    if 'holes' in info:
        reward -= info['holes'] * 2.0

    if 'aggregate_height' in info:
        reward -= info['aggregate_height'] * 0.1

    return reward


def test_reward_calculation():
    """Test reward function with different scenarios"""
    print("=" * 70)
    print("TEST 4A: Reward Function Calculation")
    print("=" * 70)

    # Test case 1: No lines cleared
    info1 = {'number_of_lines': 0}
    reward1 = simple_reward(0, info1)
    print(f"\nTest 1 - No lines cleared:")
    print(f"  Info: {info1}")
    print(f"  Calculated reward: {reward1}")
    print(f"  Expected: 1.0")
    assert abs(reward1 - 1.0) < 0.01, f"Expected 1.0, got {reward1}"
    print("  ✅ Correct")

    # Test case 2: 1 line cleared
    info2 = {'number_of_lines': 1}
    reward2 = simple_reward(0, info2)
    print(f"\nTest 2 - 1 line cleared:")
    print(f"  Info: {info2}")
    print(f"  Calculated reward: {reward2}")
    print(f"  Expected: 101.0 (1.0 base + 100)")
    assert abs(reward2 - 101.0) < 0.01, f"Expected 101.0, got {reward2}"
    print("  ✅ Correct")

    # Test case 3: 4 lines cleared (Tetris!)
    info3 = {'number_of_lines': 4}
    reward3 = simple_reward(0, info3)
    print(f"\nTest 3 - 4 lines cleared (Tetris!):")
    print(f"  Info: {info3}")
    print(f"  Calculated reward: {reward3}")
    print(f"  Expected: 401.0 (1.0 base + 400)")
    assert abs(reward3 - 401.0) < 0.01, f"Expected 401.0, got {reward3}"
    print("  ✅ Correct")

    # Test case 4: No lines, but holes in info
    info4 = {'number_of_lines': 0, 'holes': 5}
    reward4 = simple_reward(0, info4)
    print(f"\nTest 4 - No lines, 5 holes:")
    print(f"  Info: {info4}")
    print(f"  Calculated reward: {reward4}")
    print(f"  Expected: -9.0 (1.0 base - 10 for holes)")
    assert abs(reward4 - (-9.0)) < 0.01, f"Expected -9.0, got {reward4}"
    print("  ✅ Correct")

    print("\n✅ All reward calculations correct!")


def test_reward_during_gameplay():
    """Test reward function with actual gameplay"""
    print("\n" + "=" * 70)
    print("TEST 4B: Reward During Actual Gameplay")
    print("=" * 70)

    env = make_feature_vector_env()
    obs, info = env.reset()

    print("Playing 1000 steps and checking rewards when lines are cleared...")

    line_clear_rewards = []
    no_line_rewards = []

    for step in range(1000):
        action = env.action_space.sample()
        obs, env_reward, terminated, truncated, info = env.step(action)

        # Calculate shaped reward
        shaped_reward = simple_reward(env_reward, info)

        lines = info.get('number_of_lines', 0)

        if lines > 0:
            line_clear_rewards.append((step, lines, env_reward, shaped_reward, info))
            print(f"\n  🎉 Step {step}: CLEARED {lines} LINE(S)!")
            print(f"     Env reward: {env_reward}")
            print(f"     Shaped reward: {shaped_reward}")
            print(f"     Expected shaped reward: {1.0 + lines * 100:.1f}")
            print(f"     Info dict keys: {list(info.keys())}")
            print(f"     'number_of_lines' in info: {'number_of_lines' in info}")
        else:
            no_line_rewards.append(shaped_reward)

        if terminated or truncated:
            obs, info = env.reset()

    print(f"\n📊 Summary:")
    print(f"  Total steps: 1000")
    print(f"  Line clear events: {len(line_clear_rewards)}")
    print(f"  No-line steps: {len(no_line_rewards)}")

    if line_clear_rewards:
        print(f"\n  Line clear rewards:")
        for step, lines, env_r, shaped_r, info in line_clear_rewards:
            print(f"    Step {step}: {lines} lines -> reward {shaped_r:.1f}")

        # Verify rewards are correct
        all_correct = True
        for step, lines, env_r, shaped_r, info in line_clear_rewards:
            expected_min = 1.0 + lines * 100 - 50  # Allow some penalty from holes/height
            if shaped_r < expected_min:
                print(f"    ⚠️  Warning: Reward {shaped_r:.1f} much lower than expected {expected_min:.1f}")
                all_correct = False

        if all_correct:
            print(f"  ✅ All line clear rewards look correct!")
    else:
        print(f"  ⚠️  No lines cleared in 1000 steps - can't verify line clear rewards")

    if no_line_rewards:
        avg_no_line = np.mean(no_line_rewards)
        print(f"\n  Average reward (no lines): {avg_no_line:.2f}")
        print(f"  Expected: ~1.0 (base survival reward)")

    env.close()

    return len(line_clear_rewards) > 0


def test_info_dict_completeness():
    """Check what info dict actually contains during gameplay"""
    print("\n" + "=" * 70)
    print("TEST 4C: Info Dict Completeness Check")
    print("=" * 70)

    env = make_feature_vector_env()
    obs, info = env.reset()

    print("Checking info dict after reset:")
    print(f"  Keys: {list(info.keys())}")
    print(f"  Has 'number_of_lines': {'number_of_lines' in info}")

    # Take some steps
    print("\nChecking info dict after 20 steps:")
    for step in range(20):
        action = env.action_space.sample()
        obs, env_reward, terminated, truncated, info = env.step(action)

        if step == 0:
            print(f"\n  Step {step} info dict:")
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: array shape {value.shape}")
                else:
                    print(f"    {key}: {value} (type: {type(value).__name__})")

        if info.get('number_of_lines', 0) > 0:
            print(f"\n  Step {step}: Lines cleared!")
            print(f"    Info dict: {info}")
            break

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_reward_in_training_loop():
    """Simulate the exact reward calculation used in training"""
    print("\n" + "=" * 70)
    print("TEST 4D: Reward in Training Loop Simulation")
    print("=" * 70)

    env = make_feature_vector_env()
    obs, info = env.reset()

    print("Simulating training loop for 500 steps...")
    print("This matches exactly what happens in train_feature_vector.py\n")

    episode_rewards = []
    episode_lines = []
    current_episode_reward = 0
    current_episode_lines = 0
    episodes_completed = 0

    for step in range(500):
        # Random action (like early exploration)
        action = env.action_space.sample()

        # Step environment
        next_obs, env_reward, terminated, truncated, info = env.step(action)

        # Calculate shaped reward (EXACT COPY from training script)
        lines = info.get('number_of_lines', 0)
        shaped_reward = 1.0
        if lines > 0:
            shaped_reward += lines * 100
        if 'holes' in info:
            shaped_reward -= info['holes'] * 2.0
        if 'aggregate_height' in info:
            shaped_reward -= info['aggregate_height'] * 0.1

        current_episode_reward += shaped_reward
        current_episode_lines += lines

        if lines > 0:
            print(f"  Step {step}: Cleared {lines} lines!")
            print(f"    Env reward: {env_reward}")
            print(f"    Shaped reward: {shaped_reward:.2f}")
            print(f"    Episode total reward: {current_episode_reward:.2f}")

        if terminated or truncated:
            episodes_completed += 1
            episode_rewards.append(current_episode_reward)
            episode_lines.append(current_episode_lines)

            print(f"\n  Episode {episodes_completed} completed:")
            print(f"    Total reward: {current_episode_reward:.2f}")
            print(f"    Total lines: {current_episode_lines}")
            print(f"    Steps: {step}")

            # Reset
            obs, info = env.reset()
            current_episode_reward = 0
            current_episode_lines = 0

        obs = next_obs

    print(f"\n📊 Training simulation summary:")
    print(f"  Episodes completed: {episodes_completed}")
    print(f"  Total lines cleared: {sum(episode_lines)}")
    print(f"  Episodes with lines: {sum(1 for x in episode_lines if x > 0)}")
    if episode_rewards:
        print(f"  Average reward/episode: {np.mean(episode_rewards):.2f}")

    env.close()

    if sum(episode_lines) == 0:
        print(f"\n  ⚠️  WARNING: No lines cleared in training simulation!")
        print(f"  This matches your actual training results.")
    else:
        print(f"\n  ✅ Lines being cleared in simulation!")


if __name__ == "__main__":
    print("🔍 DEBUGGING PLAN - TEST 4: Reward Function")
    print("=" * 70)
    print("This test verifies that the reward function works correctly.\n")

    try:
        test_reward_calculation()
        saw_lines = test_reward_during_gameplay()
        test_info_dict_completeness()
        test_reward_in_training_loop()

        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print("✅ PASS: Reward function calculations are correct")
        if saw_lines:
            print("   Reward function properly handles line clears.")
        else:
            print("   ⚠️  WARNING: Didn't observe line clears during test")
            print("   Reward function is correct, but lines aren't being cleared.")
        print("=" * 70)

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"❌ FAIL: Reward function has bugs")
        print(f"   Error: {e}")
        print("   Fix reward function before continuing.")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"❌ ERROR: Unexpected error during testing")
        print(f"   Error: {e}")
        print("=" * 70)
