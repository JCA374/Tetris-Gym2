"""
Visualize what the trained agent is actually doing.
Show the board state to understand why it's not clearing lines.
"""

import torch
import numpy as np

from src.env_feature_vector import make_feature_vector_env
from src.model_fc import create_feature_vector_model

def visualize_board(board_dict):
    """Print board in readable format"""
    board = board_dict['board'] if isinstance(board_dict, dict) else board_dict
    playable = board[0:20, 4:14]  # Extract playable area

    print("\nPlayable board (20x10):")
    for i, row in enumerate(playable):
        line = f"{i:2d} |"
        for cell in row:
            if cell == 0:
                line += '·'
            elif cell == 1:
                line += '█'
            else:
                line += str(min(cell, 9))
        line += '|'
        print(line)
    print("   " + "-"*12)

    # Show column heights
    heights = []
    for col in range(10):
        col_data = playable[:, col]
        filled = np.where(col_data > 0)[0]
        if len(filled) > 0:
            height = 20 - filled[0]
        else:
            height = 0
        heights.append(height)

    print(f"Heights: {heights}")
    print(f"Max: {max(heights)}, Min: {min(heights)}, Avg: {np.mean(heights):.1f}")

    # Check for almost-complete rows
    for i, row in enumerate(playable):
        filled = np.sum(row > 0)
        if filled >= 8:
            print(f"  Row {i}: {filled}/10 filled {'⚠️ ALMOST COMPLETE!' if filled == 9 else ''}")

def play_one_episode_visualized(model_path):
    """Play one episode and show what happens"""
    print("="*60)
    print("VISUALIZING TRAINED AGENT GAMEPLAY")
    print("="*60)

    # Create environment (raw, without wrapper)
    import gymnasium as gym
    import tetris_gymnasium.envs
    from src.feature_vector import extract_feature_vector, normalize_features

    env = gym.make('tetris_gymnasium/Tetris', height=20, width=10)

    # Load model
    model = create_feature_vector_model(model_type='fc_dqn')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

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

    obs, info = env.reset()

    print("\nInitial board:")
    visualize_board(obs)

    print("\n" + "="*60)
    print("PLAYING...")
    print("="*60)

    action_sequence = []
    pieces_placed = 0

    for step in range(200):
        # Extract features for agent
        features = extract_feature_vector(obs)
        features_norm = normalize_features(features)

        # Get action
        with torch.no_grad():
            state = torch.FloatTensor(features_norm).unsqueeze(0)
            q_values = model(state)
            action = q_values.argmax().item()

        action_sequence.append(action)

        # Execute
        obs, reward, terminated, truncated, info = env.step(action)

        # Show piece placements (roughly every 10-20 actions)
        if step > 0 and step % 20 == 0:
            pieces_placed += 1
            print(f"\nAfter ~{pieces_placed} pieces ({step} actions):")
            print(f"Recent actions: {[action_names[a] for a in action_sequence[-10:]]}")
            visualize_board(obs)

            lines = info.get('lines_cleared', 0)
            if lines > 0:
                print(f"\n🎉 LINES CLEARED: {lines}")

        if terminated or truncated:
            print(f"\nGame over after {step} actions")
            break

    print("\n" + "="*60)
    print("FINAL BOARD:")
    print("="*60)
    visualize_board(obs)

    # Analyze action sequence
    from collections import Counter
    action_counts = Counter(action_sequence)

    print("\nActions used:")
    for action_id, count in sorted(action_counts.items()):
        pct = 100 * count / len(action_sequence)
        print(f"  {action_names[action_id]:12s}: {count:3d} ({pct:5.1f}%)")

if __name__ == "__main__":
    play_one_episode_visualized("models/checkpoint_ep500.pth")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("Look at the board progression to understand:")
    print("  1. Are pieces being placed evenly across columns?")
    print("  2. Are there rows that are ALMOST complete (9/10 filled)?")
    print("  3. What pattern does the agent create?")
    print("  4. Why doesn't this pattern create complete rows?")
