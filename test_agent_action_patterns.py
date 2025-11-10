"""
Analyze what actions our trained agent is actually using.
Key question: Is the agent using LEFT/RIGHT to spread pieces, or just HARD_DROP/NOOP?
"""

import gymnasium as gym
import numpy as np
import torch
from collections import Counter

from src.env_feature_vector import make_feature_vector_env
from src.agent import Agent

def analyze_agent_actions(model_path, num_episodes=10):
    """
    Load a trained agent and analyze its action distribution.
    """
    print("="*60)
    print(f"ANALYZING AGENT ACTION PATTERNS")
    print(f"Model: {model_path}")
    print("="*60)

    # Create environment
    env = make_feature_vector_env()

    # Create agent
    agent = Agent(
        state_size=17,
        action_size=8,
        model_type='fc_dqn',
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_method='adaptive',
        total_episodes=5000
    )

    # Load model
    try:
        agent.load(model_path)
        print(f"✅ Loaded model from {model_path}\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # Action names for readability
    action_names = {
        0: 'LEFT',
        1: 'RIGHT',
        2: 'DOWN',
        3: 'ROTATE_CW',
        4: 'ROTATE_CCW',
        5: 'HARD_DROP',
        6: 'SWAP',
        7: 'NOOP'
    }

    # Track actions across episodes
    all_actions = []
    action_sequences = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_actions = []
        episode_lines = 0

        for step in range(1000):
            # Get agent's action (no exploration)
            action = agent.select_action(obs, training=False)
            episode_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_lines += info.get('lines_cleared', 0)

            if terminated or truncated:
                break

        all_actions.extend(episode_actions)
        action_sequences.append(episode_actions)

        print(f"Episode {ep+1}: {len(episode_actions)} actions, {episode_lines} lines cleared")

    # Analyze action distribution
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)

    action_counts = Counter(all_actions)
    total_actions = len(all_actions)

    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = 100 * count / total_actions
        bar = '█' * int(pct / 2)
        print(f"  {action_names[action_id]:12s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Check for horizontal movement
    horizontal_actions = action_counts[0] + action_counts[1]  # LEFT + RIGHT
    horizontal_pct = 100 * horizontal_actions / total_actions

    print(f"\n  Horizontal movement (LEFT + RIGHT): {horizontal_pct:.1f}%")

    if horizontal_pct < 5:
        print("  ⚠️  PROBLEM: Agent barely uses LEFT/RIGHT!")
        print("     Agent won't spread pieces horizontally")
    elif horizontal_pct < 20:
        print("  ⚠️  WARNING: Low horizontal movement")
        print("     Agent may not spread pieces enough")
    else:
        print("  ✅ Agent uses horizontal movement")

    # Analyze action patterns
    print("\n" + "="*60)
    print("ACTION PATTERNS")
    print("="*60)

    # Look for common sequences
    print("\nMost common 2-action sequences:")
    bigrams = []
    for seq in action_sequences:
        for i in range(len(seq) - 1):
            bigrams.append((seq[i], seq[i+1]))

    bigram_counts = Counter(bigrams)
    for bigram, count in bigram_counts.most_common(10):
        pct = 100 * count / len(bigrams)
        action1, action2 = bigram
        print(f"  {action_names[action1]:12s} → {action_names[action2]:12s}: {count:4d} ({pct:4.1f}%)")

    # Check if agent moves before dropping
    print("\n" + "="*60)
    print("MOVE BEFORE DROP ANALYSIS")
    print("="*60)

    drops_after_movement = 0
    total_drops = 0

    for seq in action_sequences:
        for i in range(len(seq)):
            if seq[i] == 5:  # HARD_DROP
                total_drops += 1
                # Check if any LEFT/RIGHT in previous 5 actions
                lookback = seq[max(0, i-5):i]
                if 0 in lookback or 1 in lookback:
                    drops_after_movement += 1

    if total_drops > 0:
        move_before_drop_pct = 100 * drops_after_movement / total_drops
        print(f"Hard drops after LEFT/RIGHT: {drops_after_movement}/{total_drops} ({move_before_drop_pct:.1f}%)")

        if move_before_drop_pct < 10:
            print("❌ CRITICAL: Agent almost never moves pieces before dropping!")
            print("   This explains why no lines are being cleared")
        elif move_before_drop_pct < 30:
            print("⚠️  WARNING: Agent rarely moves pieces before dropping")
        else:
            print("✅ Agent moves pieces before dropping")

def analyze_feature_vector():
    """
    Check if feature vector provides information about horizontal spreading.
    """
    print("\n" + "="*60)
    print("FEATURE VECTOR ANALYSIS")
    print("="*60)

    from src.feature_vector import extract_feature_vector, normalize_features

    # Create a dummy observation with pieces in different configurations
    env = make_feature_vector_env()
    obs, info = env.reset()

    features = extract_feature_vector(obs)
    features_norm = normalize_features(features)

    print("\nFeature vector (17 values):")
    print(f"  aggregate_height: {features_norm[0]:.3f}")
    print(f"  holes: {features_norm[1]:.3f}")
    print(f"  bumpiness: {features_norm[2]:.3f}")
    print(f"  wells: {features_norm[3]:.3f}")
    print(f"  column_heights[0-9]: {features_norm[4:14]}")
    print(f"  max_height: {features_norm[14]:.3f}")
    print(f"  min_height: {features_norm[15]:.3f}")
    print(f"  std_height: {features_norm[16]:.3f}")

    print("\n✅ Feature vector includes:")
    print("   - Individual column heights (can detect uneven filling)")
    print("   - Bumpiness (difference between adjacent columns)")
    print("   - Wells (valleys between columns)")
    print("\n   Agent SHOULD have information to learn horizontal spreading")

if __name__ == "__main__":
    import sys

    # Check if model exists
    model_path = "models/best_model.pth"

    print("AGENT ACTION PATTERN ANALYSIS")
    print("="*60)
    print("This will analyze what actions the trained agent uses")
    print("to identify why it's not clearing lines\n")

    try:
        analyze_agent_actions(model_path, num_episodes=10)
    except Exception as e:
        print(f"\n⚠️  Could not analyze agent: {e}")
        print("   This is expected if no trained model exists yet\n")

    analyze_feature_vector()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("If agent barely uses LEFT/RIGHT actions:")
    print("  → Reward function may not incentivize horizontal spreading")
    print("  → Agent learned to just spam HARD_DROP/NOOP for survival")
    print("  → Need to add rewards for even column distribution")
