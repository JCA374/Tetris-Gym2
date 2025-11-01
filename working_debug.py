# working_debug.py
"""
Working debug script with proper tetris-gymnasium initialization
This handles different versions and naming conventions
"""

import numpy as np
import gymnasium as gym
import random
import sys

def initialize_tetris_env():
    """Try different ways to initialize the Tetris environment"""
    env = None
    
    # Try different environment names
    env_names = [
        'Tetris-v0',
        'Tetris-v1', 
        'tetris_gymnasium/Tetris',
        'TetrisGymnasium-v0',
        'tetris-gymnasium/Tetris-v0'
    ]
    
    print("Trying to initialize Tetris environment...")
    
    # First, try importing the module to register environments
    try:
        import tetris_gymnasium
        import tetris_gymnasium.envs
        print("✓ tetris_gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tetris_gymnasium: {e}")
        return None
    
    # Now try different environment names
    for name in env_names:
        try:
            env = gym.make(name)
            print(f"✓ Successfully created environment with: {name}")
            return env, name
        except Exception as e:
            continue
    
    # If nothing worked, try to see what environments are available
    print("\n Available Tetris environments:")
    try:
        all_envs = gym.envs.registry.keys()
        tetris_envs = [e for e in all_envs if 'tetris' in e.lower()]
        for env_name in tetris_envs:
            print(f"  - {env_name}")
        
        if tetris_envs:
            # Try the first one found
            try:
                env = gym.make(tetris_envs[0])
                print(f"\n✓ Using found environment: {tetris_envs[0]}")
                return env, tetris_envs[0]
            except:
                pass
    except:
        pass
    
    print("\n✗ Could not create Tetris environment")
    print("\nTroubleshooting:")
    print("1. Make sure tetris-gymnasium is installed:")
    print("   pip install tetris-gymnasium")
    print("2. Try upgrading:")
    print("   pip install --upgrade tetris-gymnasium gymnasium")
    
    return None, None

def get_board_from_obs(obs):
    """Extract board from observation, handling different formats"""
    if obs is None:
        return None
        
    # Handle dict observation
    if isinstance(obs, dict):
        if 'board' in obs:
            return obs['board']
        elif 'grid' in obs:
            return obs['grid']
    
    # Handle array observation
    if isinstance(obs, np.ndarray):
        # Could be the board directly or need extraction
        if len(obs.shape) == 2:
            return obs
        elif len(obs.shape) == 3:
            # Might be (channels, height, width) or similar
            return obs[0] if obs.shape[0] < obs.shape[1] else obs
    
    return obs

def get_column_heights(board):
    """Calculate height of each column"""
    if board is None:
        return [0] * 10
    
    heights = []
    width = min(board.shape[1], 10)  # Handle different board widths
    
    for col in range(width):
        height = 0
        for row in range(board.shape[0]):
            if board[row, col] != 0:
                height = board.shape[0] - row
                break
        heights.append(height)
    
    # Pad to 10 columns if needed
    while len(heights) < 10:
        heights.append(0)
    
    return heights[:10]

def test_random_play():
    """Test what happens with random actions"""
    env, env_name = initialize_tetris_env()
    if env is None:
        return None
    
    print(f"\n{'='*50}")
    print("TEST 1: Random Action Test")
    print('='*50)
    
    try:
        obs, info = env.reset()
    except:
        obs = env.reset()
        info = {}
    
    print(f"Environment: {env_name}")
    print(f"Action space: {env.action_space}")
    print(f"Observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
    
    # Take random actions
    action_counts = {}
    column_usage = [0] * 10
    
    for step in range(200):
        action = env.action_space.sample()
        action_counts[action] = action_counts.get(action, 0) + 1
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except:
            obs, reward, done, info = env.step(action)
        
        # Every 50 steps, check board state
        if step % 50 == 49:
            board = get_board_from_obs(obs)
            if board is not None:
                heights = get_column_heights(board)
                print(f"\nStep {step+1} - Heights: {heights}")
                
                for i, h in enumerate(heights):
                    if h > 0:
                        column_usage[i] += 1
        
        if done:
            print(f"Game ended at step {step+1}")
            break
    
    # Final analysis
    board = get_board_from_obs(obs)
    if board is not None:
        final_heights = get_column_heights(board)
        print(f"\nFinal heights: {final_heights}")
    else:
        final_heights = [0] * 10
    
    print(f"\nAction distribution (200 random actions):")
    for action, count in sorted(action_counts.items()):
        print(f"  Action {action}: {count} times ({count/sum(action_counts.values())*100:.1f}%)")
    
    print(f"\nColumn usage frequency:")
    for i in range(10):
        bar = '█' * (column_usage[i] * 2)
        print(f"  Col {i}: {bar}")
    
    # Check for center-stacking
    outer_cols = [0, 1, 2, 7, 8, 9]
    outer_usage = sum(column_usage[i] for i in outer_cols)
    center_usage = sum(column_usage[i] for i in [3, 4, 5, 6])
    
    print(f"\nOuter columns total usage: {outer_usage}")
    print(f"Center columns total usage: {center_usage}")
    
    if outer_usage == 0:
        print("\n🔴 CRITICAL: Random actions NEVER used outer columns!")
        print("This means pieces physically cannot reach the outer columns,")
        print("or they spawn in center and fall too fast.")
    elif outer_usage < center_usage * 0.3:
        print("\n⚠️ WARNING: Outer columns rarely used even with random actions")
        print("This suggests a bias in the game mechanics.")
    else:
        print("\n✅ Good: Random actions can use all columns")
    
    env.close()
    return final_heights

def test_specific_movements():
    """Test if we can force pieces to outer columns"""
    env, env_name = initialize_tetris_env()
    if env is None:
        return
    
    print(f"\n{'='*50}")
    print("TEST 2: Forced Movement Test")
    print('='*50)
    
    # Map action indices - these are common mappings
    possible_mappings = [
        {'left': 0, 'right': 1, 'down': 2, 'rotate': 3, 'drop': 4},
        {'left': 3, 'right': 4, 'down': 2, 'rotate': 0, 'drop': 5},
        {'left': 6, 'right': 3, 'down': 2, 'rotate': 1, 'drop': 5},
    ]
    
    for mapping_idx, actions in enumerate(possible_mappings):
        print(f"\nTrying action mapping #{mapping_idx+1}...")
        
        # Reset environment
        try:
            obs, info = env.reset()
        except:
            obs = env.reset()
        
        # Try moving left multiple times
        print("  Moving LEFT 10 times...")
        for _ in range(10):
            if 'left' in actions:
                obs, _, done, *_ = env.step(actions['left'])
                if done:
                    break
        
        # Check where we are
        board = get_board_from_obs(obs)
        if board is not None:
            heights = get_column_heights(board)
            if any(heights[i] > 0 for i in [0, 1, 2]):
                print(f"  ✅ Reached left side! Heights: {heights}")
                print(f"  Correct mapping found: LEFT={actions['left']}")
                break
            else:
                print(f"  ✗ Did not reach left side. Heights: {heights}")
    
    env.close()

def analyze_your_training():
    """Look at your training results if available"""
    print(f"\n{'='*50}")
    print("TEST 3: Your Training Analysis")
    print('='*50)
    
    import os
    
    # Check for board_states.txt
    if os.path.exists('board_states.txt'):
        print("Found board_states.txt - analyzing...")
        
        with open('board_states.txt', 'r') as f:
            lines = f.readlines()
        
        # Parse episodes
        episodes_found = 0
        center_stacking_episodes = 0
        
        for line in lines:
            if 'heights' in line.lower() or 'column' in line.lower():
                # Try to extract height data
                if '[' in line and ']' in line:
                    try:
                        # Extract the list
                        start = line.index('[')
                        end = line.index(']') + 1
                        heights_str = line[start:end]
                        heights = eval(heights_str)
                        
                        if len(heights) >= 10:
                            episodes_found += 1
                            
                            # Check if center-stacking
                            outer_empty = sum(1 for i in [0,1,2,7,8,9] if heights[i] == 0)
                            if outer_empty >= 5:
                                center_stacking_episodes += 1
                                if episodes_found <= 5:  # Show first few
                                    print(f"  Episode {episodes_found}: {heights} ← CENTER-STACKING")
                    except:
                        pass
        
        if episodes_found > 0:
            print(f"\nAnalysis of {episodes_found} episodes:")
            print(f"  Center-stacking: {center_stacking_episodes}/{episodes_found} ({100*center_stacking_episodes/episodes_found:.1f}%)")
            
            if center_stacking_episodes / episodes_found > 0.8:
                print("\n🔴 CONFIRMED: Your agent is center-stacking!")
        else:
            print("Could not parse height data from board_states.txt")
    else:
        print("No board_states.txt found")
    
    # Check for checkpoints
    if os.path.exists('checkpoints'):
        print("\nFound checkpoints folder")
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if checkpoints:
            print(f"  Found {len(checkpoints)} checkpoint files")
            latest = max(checkpoints)
            print(f"  Latest: {latest}")

def main():
    print("="*60)
    print("TETRIS CENTER-STACKING DIAGNOSTIC - WORKING VERSION")
    print("="*60)
    
    # Run tests
    heights = test_random_play()
    
    if heights is not None:
        test_specific_movements()
        analyze_your_training()
        
        # Diagnosis
        print(f"\n{'='*60}")
        print("DIAGNOSIS")
        print('='*60)
        
        outer_heights = [heights[i] for i in [0,1,2,7,8,9] if i < len(heights)]
        if all(h == 0 for h in outer_heights):
            print("\n🔴 ROOT CAUSE: Physical/Mechanical Issue")
            print("Pieces cannot reach outer columns due to:")
            print("1. Spawn position (pieces start in center)")
            print("2. Movement limits (can't move far enough)")
            print("3. Fall speed (pieces drop before reaching edges)")
            print("\nSOLUTION:")
            print("- Use more LEFT/RIGHT actions before dropping")
            print("- Reduce soft drop usage") 
            print("- Check if you're using HARD_DROP too early")
        else:
            print("\n⚠️ ROOT CAUSE: Reward/Learning Issue")
            print("Pieces CAN reach outer columns but agent doesn't use them")
            print("\nSOLUTIONS:")
            print("1. Increase outer column reward: +20 per outer column used")
            print("2. Add harsh penalty: -50 if all outer columns empty")
            print("3. Force exploration: First 200 episodes, 30% random actions")
            print("4. Check epsilon decay: Should stay high (>0.5) for 100+ episodes")

if __name__ == "__main__":
    main()