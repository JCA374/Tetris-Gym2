# debug_center_stacking.py
"""
Simple debugging script to identify why the agent is center-stacking
Run this to get immediate insights into the problem
"""

import numpy as np
import gymnasium as gym
import tetris_gymnasium

def get_column_heights(board):
    """Get the height of each column"""
    heights = []
    for col in range(10):
        column = board[:, col]
        non_zero = np.where(column != 0)[0]
        if len(non_zero) > 0:
            heights.append(20 - non_zero[0])
        else:
            heights.append(0)
    return heights

def visualize_board(board):
    """Print a visual representation of the board"""
    print("\nBoard State (X = filled, . = empty):")
    print("   " + "".join(str(i) for i in range(10)))
    print("   " + "-" * 10)
    
    for row in range(20):
        row_str = f"{19-row:2}|"
        for col in range(10):
            if board[row, col] > 0:
                row_str += "X"
            else:
                row_str += "."
        print(row_str + "|")
    print("   " + "-" * 10)

def test_action_space():
    """Test if pieces can physically reach outer columns"""
    print("\n" + "="*60)
    print("TEST 1: Can pieces reach outer columns?")
    print("="*60)
    
    # Import and register the environment
    import tetris_gymnasium
    env = gym.make('tetris_gymnasium/Tetris')
    
    # Track where pieces land with random actions
    column_landings = {i: 0 for i in range(10)}
    
    for episode in range(10):
        obs, _ = env.reset()
        prev_board = np.zeros((20, 10))
        
        for step in range(50):  # Take 50 random actions
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            
            # Extract board
            if isinstance(obs, dict):
                curr_board = obs['board'][10:30, :10]
            else:
                curr_board = obs[0, 10:30, :10] if len(obs.shape) == 3 else obs[10:30, :10]
            
            # Check which column got a new piece
            diff = curr_board - prev_board
            for col in range(10):
                if np.any(diff[:, col] > 0):
                    column_landings[col] += 1
            
            prev_board = curr_board.copy()
            
            if done:
                break
    
    print(f"\nRandom action results (10 episodes):")
    for col in range(10):
        bar = '█' * (column_landings[col] // 2)
        print(f"Column {col}: {column_landings[col]:3d} pieces {bar}")
    
    outer_total = sum(column_landings[c] for c in [0,1,2,7,8,9])
    center_total = sum(column_landings[c] for c in [3,4,5,6])
    
    print(f"\nOuter columns (0-2, 7-9): {outer_total} pieces")
    print(f"Center columns (3-6): {center_total} pieces")
    
    if outer_total == 0:
        print("🔴 CRITICAL: Pieces NEVER reach outer columns with random actions!")
        print("   This is likely an environment or action mapping issue.")
    elif outer_total < center_total * 0.5:
        print("⚠️ WARNING: Outer columns get significantly fewer pieces")
        print("   This suggests a bias in the action space or piece spawning.")
    else:
        print("✅ GOOD: Pieces can reach all columns with random actions")
    
    env.close()
    return column_landings

def test_reward_calculation():
    """Test if reward calculation properly rewards spreading"""
    print("\n" + "="*60)
    print("TEST 2: Reward Calculation Check")
    print("="*60)
    
    # Create test boards
    test_boards = [
        ("Perfect Center-Stack", [0, 0, 0, 15, 18, 18, 15, 0, 0, 0]),
        ("Perfect Spread", [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
        ("Partial Spread", [0, 5, 7, 9, 10, 10, 9, 7, 5, 0]),
        ("Left Heavy", [15, 15, 12, 10, 5, 3, 0, 0, 0, 0]),
        ("Right Heavy", [0, 0, 0, 3, 5, 10, 12, 15, 15, 15]),
    ]
    
    # Import the reward calculation from your training script
    # This is a mock version - replace with your actual import
    def calculate_mock_reward(heights):
        """Mock reward calculation - replace with your actual function"""
        # Penalties
        total_height = sum(heights)
        holes = 0  # Simplified - no holes in test
        
        # Spread calculation
        columns_used = sum(1 for h in heights if h > 0)
        outer_used = sum(1 for c in [0,1,2,7,8,9] if heights[c] > 0)
        
        # Mock reward
        reward = 0
        reward -= 0.1 * total_height  # Height penalty
        reward += 5.0 * columns_used  # Column usage bonus
        reward += 10.0 * (outer_used / 6.0)  # Outer column bonus
        
        # Penalty for unused outer columns
        outer_empty = sum(1 for c in [0,1,2,7,8,9] if heights[c] == 0)
        reward -= 8.0 * outer_empty
        
        return reward
    
    print("\nReward calculations for different board states:")
    print("-" * 50)
    
    results = []
    for name, heights in test_boards:
        reward = calculate_mock_reward(heights)
        columns_used = sum(1 for h in heights if h > 0)
        outer_used = sum(1 for c in [0,1,2,7,8,9] if heights[c] > 0)
        
        results.append((name, reward, columns_used, outer_used))
        print(f"{name:20s}: Reward={reward:+8.2f}, Cols={columns_used}, Outer={outer_used}")
    
    # Check if spreading is properly rewarded
    center_reward = results[0][1]  # Perfect center-stack
    spread_reward = results[1][1]   # Perfect spread
    
    print("\n" + "-" * 50)
    if spread_reward > center_reward + 20:
        print(f"✅ GOOD: Spreading rewarded {spread_reward - center_reward:+.1f} more than center-stacking")
    else:
        print(f"🔴 PROBLEM: Spread advantage only {spread_reward - center_reward:+.1f} points")
        print("   This is not enough to overcome the difficulty of spreading!")
        print("   Spreading should be at least +30 to +50 points better")

def test_with_actual_game():
    """Run actual game episodes and analyze behavior"""
    print("\n" + "="*60)
    print("TEST 3: Actual Gameplay Analysis")
    print("="*60)
    
    import tetris_gymnasium
    env = gym.make('tetris_gymnasium/Tetris')
    
    # Test different action patterns
    action_patterns = [
        ("Only DROP", [5]),  # Only hard drop
        ("LEFT + DROP", [0, 0, 0, 5]),  # Move left then drop
        ("RIGHT + DROP", [1, 1, 1, 5]),  # Move right then drop
        ("Mixed", [0, 1, 5, 0, 1, 5]),  # Mixed movements
    ]
    
    for pattern_name, actions in action_patterns:
        obs, _ = env.reset()
        
        print(f"\nTesting pattern: {pattern_name}")
        
        # Run for 10 pieces
        for piece_num in range(10):
            for action in actions:
                obs, _, done, _, _ = env.step(action)
                if done:
                    break
            
            if done:
                break
        
        # Analyze final board
        if isinstance(obs, dict):
            board = obs['board'][10:30, :10]
        else:
            board = obs[0, 10:30, :10] if len(obs.shape) == 3 else obs[10:30, :10]
        
        heights = get_column_heights(board)
        columns_used = sum(1 for h in heights if h > 0)
        outer_used = sum(1 for c in [0,1,2,7,8,9] if heights[c] > 0)
        
        print(f"  Columns used: {columns_used}/10")
        print(f"  Outer columns used: {outer_used}/6")
        print(f"  Heights: {heights}")
    
    env.close()

def diagnose_trained_agent():
    """Load and analyze a trained agent if checkpoint exists"""
    print("\n" + "="*60)
    print("TEST 4: Trained Agent Analysis")
    print("="*60)
    
    import os
    import torch
    
    checkpoint_path = "checkpoints/progressive_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Train the agent first.")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Analyze training metrics
    if 'episode' in checkpoint:
        print(f"Trained for {checkpoint['episode']} episodes")
    if 'epsilon' in checkpoint:
        print(f"Final epsilon: {checkpoint['epsilon']:.4f}")
    if 'stage_idx' in checkpoint:
        stages = ["basic_placement", "height_management", "spreading", "balanced"]
        stage_name = stages[min(checkpoint['stage_idx'], 3)]
        print(f"Final curriculum stage: {stage_name}")
    
    # Test the agent
    import tetris_gymnasium
    env = gym.make('tetris_gymnasium/Tetris')
    
    # Import your model class
    # from train_progressive import TetrisDQN
    
    # Load and test model
    # model = TetrisDQN()
    # model.load_state_dict(checkpoint['q_network_state'])
    # ... run test episodes ...
    
    print("\nTo fully test the trained agent, uncomment the model loading code above")
    env.close()

def main():
    """Run all diagnostic tests"""
    print("🔍 TETRIS CENTER-STACKING DIAGNOSTIC")
    print("=" * 60)
    
    # Run tests
    print("\nRunning diagnostic tests...\n")
    
    # Test 1: Can pieces reach outer columns?
    column_distribution = test_action_space()
    
    # Test 2: Is reward calculation correct?
    test_reward_calculation()
    
    # Test 3: Actual gameplay patterns
    test_with_actual_game()
    
    # Test 4: Trained agent analysis (if exists)
    diagnose_trained_agent()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("\n📋 Checklist of potential issues:")
    print("[ ] Pieces can't physically reach outer columns")
    print("[ ] Reward calculation doesn't favor spreading enough")
    print("[ ] Epsilon too low (not exploring)")
    print("[ ] Action mapping is incorrect")
    print("[ ] Curriculum stages not advancing")
    print("[ ] Network not learning properly")
    
    print("\n💡 Most likely cause based on tests:")
    if sum(column_distribution[c] for c in [0,1,2,7,8,9]) == 0:
        print("🔴 CRITICAL: Action space or environment issue!")
        print("   Pieces literally cannot reach outer columns.")
        print("   Check action mappings and piece spawn location.")
    else:
        print("⚠️ Reward shaping issue detected!")
        print("   The reward difference between spreading and center-stacking")
        print("   is not large enough to overcome the learning difficulty.")
        print("\n   Recommended fix:")
        print("   1. Increase outer column bonus to +15 per column")
        print("   2. Add exponential penalty for unused outer columns")
        print("   3. Use forced exploration for first 100 episodes")

if __name__ == "__main__":
    main()