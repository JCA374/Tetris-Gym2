"""
Simple test: What actions does the trained agent use?
"""

import torch
import numpy as np
from collections import Counter

from src.env_feature_vector import make_feature_vector_env
from src.model_fc import create_feature_vector_model

def test_trained_actions(model_path, num_episodes=10):
    """Test trained agent's action patterns"""
    print("="*60)
    print(f"TESTING TRAINED AGENT")
    print(f"Model: {model_path}")
    print("="*60)

    # Create environment
    env = make_feature_vector_env()

    # Create model
    model = create_feature_vector_model(model_type='fc_dqn')

    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

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

    action_counts = Counter()
    total_lines = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_actions = []

        for step in range(1000):
            # Get action from model (greedy)
            with torch.no_grad():
                state = torch.FloatTensor(obs).unsqueeze(0)
                q_values = model(state)
                action = q_values.argmax().item()

            episode_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_lines += info.get('lines_cleared', 0)

            if terminated or truncated:
                break

        action_counts.update(episode_actions)
        total_steps += len(episode_actions)
        print(f"  Episode {ep+1}: {len(episode_actions)} steps, {info.get('lines_cleared', 0)} lines")

    print(f"\n  Average episode length: {total_steps/num_episodes:.1f} steps")
    print(f"  Total lines cleared: {total_lines}")

    # Analyze action distribution
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)

    total_actions = sum(action_counts.values())

    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = 100 * count / total_actions
        bar = '█' * int(pct / 2)
        print(f"  {action_names[action_id]:12s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Analyze horizontal movement
    horizontal = action_counts[0] + action_counts[1]  # LEFT + RIGHT
    horizontal_pct = 100 * horizontal / total_actions

    print(f"\n  Horizontal movement (LEFT+RIGHT): {horizontal_pct:.1f}%")

    if horizontal_pct < 5:
        print("\n  ❌ CRITICAL PROBLEM: Agent barely uses LEFT/RIGHT!")
        print("     Agent won't spread pieces → Can't clear lines")
        print("\n  ROOT CAUSE IDENTIFIED:")
        print("     - Agent learned that survival = positive reward")
        print("     - Spam HARD_DROP/NOOP → quick pieces → survive longer")
        print("     - Never learned that spreading pieces → line clears → huge rewards")
    elif horizontal_pct < 15:
        print("\n  ⚠️  WARNING: Low horizontal movement")
        print("     Agent may not spread pieces enough for consistent line clears")
    else:
        print("\n  ✅ Agent uses horizontal movement appropriately")

    return action_counts

if __name__ == "__main__":
    print("ANALYZING TRAINED AGENT ACTION PATTERNS")
    print("="*60)

    model_path = "models/checkpoint_ep500.pth"
    action_counts = test_trained_actions(model_path, num_episodes=10)

    if action_counts:
        print("\n" + "="*60)
        print("DIAGNOSIS")
        print("="*60)

        horizontal = action_counts[0] + action_counts[1]
        total = sum(action_counts.values())
        horiz_pct = 100 * horizontal / total

        if horiz_pct < 10:
            print("\nThe agent learned a SUBOPTIMAL strategy:")
            print("  1. Reward function gives +1 per step survived")
            print("  2. Agent learns: survive longer = more reward")
            print("  3. Fastest way to place pieces = just HARD_DROP")
            print("  4. Moving pieces takes more steps → less efficient")
            print("  5. Agent never discovers line clears give +100 reward")
            print("\nSOLUTION:")
            print("  - Add reward for horizontal spreading")
            print("  - Add reward for even column heights")
            print("  - Reduce per-step survival reward")
            print("  - OR: Use reward shaping to encourage exploration")
