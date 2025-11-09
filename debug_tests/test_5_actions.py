"""
Test 5: Action Distribution - Check which actions agent is using
=================================================================

This test analyzes which actions the trained agent actually uses.
If the agent never uses HARD_DROP, it can't clear lines efficiently.
"""

import sys
import numpy as np
import torch
import tetris_gymnasium.envs  # Required to register environment
sys.path.insert(0, '/home/jonas/Code/Tetris-Gym2')

from src.env_feature_vector import make_feature_vector_env
from src.model_fc import create_feature_vector_model
from pathlib import Path

ACTION_NAMES = ['LEFT', 'RIGHT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'HARD_DROP', 'SWAP', 'NOOP']


def test_random_action_distribution():
    """Baseline: what does random action selection look like?"""
    print("=" * 70)
    print("TEST 5A: Random Action Distribution (Baseline)")
    print("=" * 70)

    env = make_feature_vector_env()
    action_counts = {i: 0 for i in range(8)}

    obs, info = env.reset()

    for step in range(1000):
        action = env.action_space.sample()
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    print("\n🎮 Random action distribution (1000 steps):")
    for action, count in action_counts.items():
        percentage = count / 1000 * 100
        print(f"  {action} {ACTION_NAMES[action]:12s}: {count:4d} ({percentage:5.1f}%)")

    print(f"\n  Expected: ~12.5% for each action (uniform random)")
    print(f"  HARD_DROP usage: {action_counts[5]/1000*100:.1f}%")

    env.close()


def test_trained_agent_actions():
    """Test what actions the trained agent actually uses"""
    print("\n" + "=" * 70)
    print("TEST 5B: Trained Agent Action Distribution")
    print("=" * 70)

    # Find the latest model
    model_path = Path('/home/jonas/Code/Tetris-Gym2/models/best_model.pth')

    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path}")
        print(f"   Checking for any model files...")

        models_dir = Path('/home/jonas/Code/Tetris-Gym2/models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pth'))
            if model_files:
                model_path = model_files[0]
                print(f"   Using: {model_path}")
            else:
                print(f"   No model files found. Skipping test.")
                return
        else:
            print(f"   Models directory doesn't exist. Skipping test.")
            return

    # Load model
    print(f"\n📦 Loading model from {model_path}")
    model = create_feature_vector_model(model_type='fc_dqn')

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded from checkpoint (episode: {checkpoint.get('episode', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"   Loaded model state dict")
    except Exception as e:
        print(f"   ⚠️  Error loading model: {e}")
        print(f"   Skipping test.")
        return

    model.eval()

    # Test agent's action selection
    env = make_feature_vector_env()
    action_counts = {i: 0 for i in range(8)}
    q_values_per_action = {i: [] for i in range(8)}

    obs, info = env.reset()

    print(f"\n🎮 Testing trained agent for 1000 steps...")

    for step in range(1000):
        # Get Q-values from model
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)

        # Select greedy action (what agent does with epsilon=0)
        action = q_values.argmax().item()
        action_counts[action] += 1

        # Store Q-values
        for a in range(8):
            q_values_per_action[a].append(q_values[0, a].item())

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    print(f"\n📊 Trained agent action distribution (1000 steps, greedy):")
    for action, count in action_counts.items():
        percentage = count / 1000 * 100
        avg_q = np.mean(q_values_per_action[action])
        bar = '█' * int(percentage / 2)
        print(f"  {action} {ACTION_NAMES[action]:12s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print(f"     Average Q-value: {avg_q:8.2f}")

    # Analysis
    print(f"\n🔍 Analysis:")
    print(f"  HARD_DROP (action 5) usage: {action_counts[5]/1000*100:.1f}%")
    if action_counts[5] < 50:  # Less than 5%
        print(f"  ⚠️  WARNING: Agent rarely uses HARD_DROP!")
        print(f"  This makes line clearing very difficult.")

    print(f"\n  NOOP (action 7) usage: {action_counts[7]/1000*100:.1f}%")
    if action_counts[7] > 500:  # More than 50%
        print(f"  ⚠️  WARNING: Agent mostly does nothing!")
        print(f"  This suggests the agent learned to minimize actions.")

    most_used = max(action_counts, key=action_counts.get)
    print(f"\n  Most used action: {ACTION_NAMES[most_used]} ({action_counts[most_used]/1000*100:.1f}%)")

    env.close()


def test_q_value_analysis():
    """Analyze Q-values to understand agent's preferences"""
    print("\n" + "=" * 70)
    print("TEST 5C: Q-Value Analysis")
    print("=" * 70)

    model_path = Path('/home/jonas/Code/Tetris-Gym2/models/best_model.pth')

    if not model_path.exists():
        print(f"⚠️  Model not found. Skipping test.")
        return

    # Load model
    model = create_feature_vector_model(model_type='fc_dqn')

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return

    model.eval()

    env = make_feature_vector_env()
    obs, info = env.reset()

    print("Analyzing Q-values for 100 states...\n")

    all_q_values = []

    for step in range(100):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

        all_q_values.append(q_values[0].numpy())

        action = q_values.argmax().item()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    all_q_values = np.array(all_q_values)  # Shape: (100, 8)

    print("📊 Q-Value Statistics (across 100 states):")
    print(f"\n  {'Action':<12s} {'Mean Q':>10s} {'Std Q':>10s} {'Min Q':>10s} {'Max Q':>10s}")
    print("  " + "-" * 56)

    for action in range(8):
        mean_q = all_q_values[:, action].mean()
        std_q = all_q_values[:, action].std()
        min_q = all_q_values[:, action].min()
        max_q = all_q_values[:, action].max()

        print(f"  {ACTION_NAMES[action]:<12s} {mean_q:10.2f} {std_q:10.2f} {min_q:10.2f} {max_q:10.2f}")

    print(f"\n🔍 Insights:")

    # Check if Q-values are all similar (not learned)
    mean_qs = all_q_values.mean(axis=0)
    q_value_range = mean_qs.max() - mean_qs.min()

    print(f"  Q-value range: {q_value_range:.2f}")
    if q_value_range < 1.0:
        print(f"  ⚠️  WARNING: Q-values very similar across actions!")
        print(f"  Agent may not have learned meaningful action preferences.")

    # Check if Q-values are unreasonably low (all negative, very negative)
    overall_mean = all_q_values.mean()
    print(f"  Overall mean Q-value: {overall_mean:.2f}")
    if overall_mean < -100:
        print(f"  ⚠️  WARNING: Very negative Q-values!")
        print(f"  Agent expects very poor outcomes for all actions.")

    # Check which action has highest average Q
    best_action = mean_qs.argmax()
    print(f"  Highest Q-value action: {ACTION_NAMES[best_action]} (Q = {mean_qs[best_action]:.2f})")

    env.close()


def test_action_sequences():
    """Look at action sequences to find patterns"""
    print("\n" + "=" * 70)
    print("TEST 5D: Action Sequence Analysis")
    print("=" * 70)

    model_path = Path('/home/jonas/Code/Tetris-Gym2/models/best_model.pth')

    if not model_path.exists():
        print(f"⚠️  Model not found. Skipping test.")
        return

    model = create_feature_vector_model(model_type='fc_dqn')

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return

    model.eval()

    env = make_feature_vector_env()
    obs, info = env.reset()

    print("Recording action sequences for 1 episode...\n")

    actions = []
    rewards = []

    for step in range(1000):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

        action = q_values.argmax().item()
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    print(f"Episode completed in {len(actions)} steps")
    print(f"Total reward: {sum(rewards):.2f}")

    print(f"\n🔍 Action sequence analysis:")

    # Look for repeated patterns
    print(f"\n  Common action sequences:")
    for length in [2, 3, 4]:
        sequences = {}
        for i in range(len(actions) - length + 1):
            seq = tuple(actions[i:i+length])
            sequences[seq] = sequences.get(seq, 0) + 1

        # Top 3 most common
        top_sequences = sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n    Length {length}:")
        for seq, count in top_sequences:
            seq_str = '-'.join([ACTION_NAMES[a][:3] for a in seq])
            print(f"      {seq_str}: {count} times")

    # Check for stuck patterns (same action repeated)
    max_repeat = 1
    current_repeat = 1
    repeat_action = actions[0]

    for i in range(1, len(actions)):
        if actions[i] == actions[i-1]:
            current_repeat += 1
            if current_repeat > max_repeat:
                max_repeat = current_repeat
                repeat_action = actions[i]
        else:
            current_repeat = 1

    print(f"\n  Longest repeated action: {ACTION_NAMES[repeat_action]} ({max_repeat} times)")
    if max_repeat > 20:
        print(f"  ⚠️  WARNING: Agent got stuck repeating same action!")

    env.close()


if __name__ == "__main__":
    print("🔍 DEBUGGING PLAN - TEST 5: Action Distribution")
    print("=" * 70)
    print("This test analyzes which actions the agent actually uses.\n")

    test_random_action_distribution()
    test_trained_agent_actions()
    test_q_value_analysis()
    test_action_sequences()

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print("Review the action distributions above.")
    print("Key questions:")
    print("  1. Does the agent use HARD_DROP? (needed for efficient line clearing)")
    print("  2. Is the agent stuck on one action? (suggests not learning)")
    print("  3. Are Q-values meaningful? (large differences = learned preferences)")
    print("=" * 70)
