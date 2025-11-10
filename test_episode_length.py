"""Check how long episodes actually run and why they terminate"""

import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

def test_episode_lengths():
    """Check how many steps episodes last"""
    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    episode_lengths = []
    termination_reasons = {'game_over': 0, 'truncated': 0}

    for ep in range(100):
        obs, info = env.reset()

        steps = 0
        for step in range(10000):  # Very large limit
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    termination_reasons['game_over'] += 1
                if truncated:
                    termination_reasons['truncated'] += 1
                break

        episode_lengths.append(steps)

    print("="*60)
    print(f"Episode Length Statistics ({len(episode_lengths)} episodes):")
    print("="*60)
    print(f"  Min length: {min(episode_lengths)} steps")
    print(f"  Max length: {max(episode_lengths)} steps")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Median length: {np.median(episode_lengths):.1f} steps")

    print(f"\nTermination reasons:")
    print(f"  Game over: {termination_reasons['game_over']}")
    print(f"  Truncated: {termination_reasons['truncated']}")

    # Count pieces that could have been placed
    print(f"\nWith avg {np.mean(episode_lengths):.0f} steps/episode:")
    print(f"  Estimated pieces placed: ~{np.mean(episode_lengths)/5:.0f}-{np.mean(episode_lengths)/3:.0f}")
    print(f"  (assuming 3-5 actions per piece on average)")

    return episode_lengths

if __name__ == "__main__":
    lengths = test_episode_lengths()

    print("\n" + "="*60)
    print("ANALYSIS:")
    if np.mean(lengths) < 100:
        print("⚠️  Episodes are very short!")
        print("   Not enough pieces being placed to clear lines")
    elif np.mean(lengths) < 500:
        print("Episodes are moderate length")
        print("   Should be enough to occasionally clear lines")
    else:
        print("Episodes are long")
        print("   Should definitely see some line clears")
