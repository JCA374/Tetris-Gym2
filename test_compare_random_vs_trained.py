"""
Compare action patterns: random agent vs trained agent.
Key question: Does trained agent use LEFT/RIGHT more strategically?
"""

import gymnasium as gym
import torch
import numpy as np
from collections import Counter

from src.env_feature_vector import make_feature_vector_env
from src.agent import Agent

def test_random_agent(num_episodes=5):
    """Test what actions a random agent uses"""
    print("="*60)
    print("RANDOM AGENT ACTIONS")
    print("="*60)

    env = make_feature_vector_env()
    action_counts = Counter()
    total_lines = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_actions = []

        for step in range(1000):
            action = env.action_space.sample()
            episode_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_lines += info.get('lines_cleared', 0)

            if terminated or truncated:
                break

        action_counts.update(episode_actions)
        print(f"  Episode {ep+1}: {len(episode_actions)} actions")

    print(f"\n  Total lines cleared: {total_lines}")
    return action_counts

def test_trained_agent(model_path, num_episodes=5):
    """Test what actions a trained agent uses"""
    print("\n" + "="*60)
    print(f"TRAINED AGENT ACTIONS")
    print(f"Model: {model_path}")
    print("="*60)

    env = make_feature_vector_env()

    # Create agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=0.0001,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration for testing
        epsilon_end=0.0,
        model_type='fc_dqn',
        max_episodes=5000
    )

    # Load model
    try:
        success = agent.load_checkpoint(path=model_path)
        if not success:
            print(f"❌ Could not load checkpoint")
            return None
        print(f"✅ Loaded model\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None

    action_counts = Counter()
    total_lines = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_actions = []

        for step in range(1000):
            # Use agent's select_action method with training=False (greedy)
            action = agent.select_action(obs, training=False)
            episode_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_lines += info.get('lines_cleared', 0)

            if terminated or truncated:
                break

        action_counts.update(episode_actions)
        print(f"  Episode {ep+1}: {len(episode_actions)} actions")

    print(f"\n  Total lines cleared: {total_lines}")
    return action_counts

def compare_action_distributions(random_counts, trained_counts):
    """Compare the action distributions"""
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION COMPARISON")
    print("="*60)

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

    all_actions = set(list(random_counts.keys()) + list(trained_counts.keys()))

    total_random = sum(random_counts.values())
    total_trained = sum(trained_counts.values()) if trained_counts else 1

    print(f"\n{'Action':<12} | {'Random':>12} | {'Trained':>12} | {'Difference':>12}")
    print("-" * 60)

    for action_id in sorted(all_actions):
        random_pct = 100 * random_counts.get(action_id, 0) / total_random
        trained_pct = 100 * trained_counts.get(action_id, 0) / total_trained if trained_counts else 0
        diff = trained_pct - random_pct

        print(f"{action_names[action_id]:<12} | {random_pct:11.1f}% | {trained_pct:11.1f}% | {diff:+11.1f}%")

    # Analyze horizontal movement
    print("\n" + "="*60)
    print("HORIZONTAL MOVEMENT ANALYSIS")
    print("="*60)

    random_horiz = random_counts[0] + random_counts[1]
    trained_horiz = (trained_counts[0] + trained_counts[1]) if trained_counts else 0

    random_horiz_pct = 100 * random_horiz / total_random
    trained_horiz_pct = 100 * trained_horiz / total_trained if trained_counts else 0

    print(f"Random agent:  {random_horiz_pct:.1f}% (LEFT + RIGHT)")
    print(f"Trained agent: {trained_horiz_pct:.1f}% (LEFT + RIGHT)")

    if trained_counts:
        if trained_horiz_pct < random_horiz_pct:
            print("\n❌ PROBLEM: Trained agent uses LESS horizontal movement than random!")
            print("   This explains why it's not clearing lines")
        elif trained_horiz_pct < 15:
            print("\n⚠️  WARNING: Trained agent still doesn't use much horizontal movement")
        else:
            print("\n✅ Trained agent uses horizontal movement")

if __name__ == "__main__":
    print("COMPARING RANDOM VS TRAINED AGENT ACTION PATTERNS")
    print("="*60)

    # Test random agent
    random_counts = test_random_agent(num_episodes=5)

    # Test trained agent
    model_path = "models/checkpoint_ep500.pth"
    trained_counts = test_trained_agent(model_path, num_episodes=5)

    # Compare
    if trained_counts:
        compare_action_distributions(random_counts, trained_counts)
    else:
        print("\nCould not analyze trained agent")

    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("- Random agent: ~12.5% per action (uniform distribution)")
    print("- If trained agent uses <5% LEFT+RIGHT: PROBLEM")
    print("- If trained agent uses 5-15% LEFT+RIGHT: CONCERNING")
    print("- If trained agent uses >20% LEFT+RIGHT: GOOD")
    print("\nGoal: Trained agent should learn that horizontal spreading")
    print("is necessary to clear lines and get big rewards (+100/line)")
