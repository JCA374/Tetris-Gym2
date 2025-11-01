# CLAUDE_CODE_IMPLEMENTATION.md
# Fix Center-Stacking Issue with Research-Based Reward Shaping

## Problem Summary
- Agent places ALL pieces in columns 3-6, leaving outer columns empty
- Diagnostic confirmed: Pieces CAN reach all columns with random actions
- Root cause: Reward function doesn't sufficiently incentivize spreading
- Current epsilon_decay: 0.9999 (reasonable)

## Implementation Task

### Step 1: Create Enhanced Reward Shaping Module
Create a new file `reward_shaping_v2.py` with the following features:

```python
class TetrisRewardShaper:
    """
    Research-based reward shaping for Tetris
    Based on successful approaches from literature
    """
    
    def __init__(self):
        # Coefficients based on research (Dellacherie, 2003 and others)
        self.weights = {
            'lines_cleared': 1.0,      # Positive: clearing lines
            'holes': -0.35663,         # Negative: holes are bad
            'aggregate_height': -0.510066,  # Negative: lower is better
            'bumpiness': -0.184483,    # Negative: smooth is better
            'well_depth': -0.1,        # Negative: deep wells are risky
            # NEW: Anti-center-stacking weights
            'column_usage': 0.2,       # Positive: use more columns
            'outer_bonus': 0.3,        # Positive: use outer columns
            'center_penalty': -0.4,    # Negative: penalize center-only
        }
        
    def calculate_features(self, board):
        """Extract all features from board state"""
        features = {}
        
        # Get column heights
        heights = self.get_column_heights(board)
        
        # Standard Tetris features
        features['holes'] = self.count_holes(board)
        features['aggregate_height'] = sum(heights)
        features['bumpiness'] = sum(abs(heights[i] - heights[i+1]) for i in range(9))
        features['max_height'] = max(heights)
        features['well_depth'] = self.calculate_wells(board)
        
        # Anti-center-stacking features
        features['columns_used'] = sum(1 for h in heights if h > 0)
        features['outer_used'] = sum(1 for i in [0,1,2,7,8,9] if heights[i] > 0)
        features['center_concentration'] = sum(heights[3:7]) / (sum(heights) + 1)
        
        return features
    
    def shape_reward(self, obs, action, env_reward, done, info, prev_features=None):
        """
        Calculate shaped reward using research-based heuristics
        """
        board = self.extract_board(obs)
        features = self.calculate_features(board)
        
        # Base reward from environment (lines cleared)
        lines = info.get('lines_cleared', 0)
        
        # Research-based: Square lines cleared to encourage multi-line clears
        shaped_reward = (lines ** 2) * 10.0
        
        # Calculate feature-based score (Dellacherie-style)
        score = 0
        score += self.weights['holes'] * features['holes']
        score += self.weights['aggregate_height'] * features['aggregate_height']
        score += self.weights['bumpiness'] * features['bumpiness']
        
        # Anti-center-stacking bonuses/penalties
        # Progressive penalty for unused outer columns
        outer_empty = 6 - features['outer_used']
        if outer_empty >= 5:  # Almost all outer columns empty
            score -= 50.0  # Heavy penalty
        elif outer_empty >= 4:
            score -= 20.0
        elif outer_empty >= 3:
            score -= 10.0
        
        # Bonus for using outer columns
        score += features['outer_used'] * 15.0
        
        # Penalty for center concentration
        if features['center_concentration'] > 0.8:  # 80%+ pieces in center
            score -= 30.0
        
        # If we have previous features, use delta scoring
        if prev_features is not None:
            # Reward = improvement in score
            prev_score = self.calculate_score(prev_features)
            curr_score = self.calculate_score(features)
            shaped_reward += (curr_score - prev_score)
        else:
            shaped_reward += score
        
        # Terminal penalty (research-backed)
        if done:
            shaped_reward -= 100.0  # Strong penalty for dying
            # Extra penalty if died while center-stacking
            if outer_empty >= 5:
                shaped_reward -= 50.0  # Additional penalty
        
        return shaped_reward, features
```

### Step 2: Create Exploration Strategy
Add to the training loop in `train_agent.py`:

```python
class ExplorationStrategy:
    """Smart exploration to break out of center-stacking"""
    
    def __init__(self, env):
        self.env = env
        self.episode_count = 0
        
    def select_action(self, state, q_network, epsilon, episode):
        """Enhanced action selection with anti-center-stacking bias"""
        
        # For first 100 episodes, force exploration of outer columns
        if episode < 100 and random.random() < 0.3:
            # 30% chance to force outer column exploration
            if random.random() < 0.5:
                # Try to reach left side
                return random.choice([0, 0, 0, 5])  # LEFT, LEFT, LEFT, DROP
            else:
                # Try to reach right side
                return random.choice([1, 1, 1, 5])  # RIGHT, RIGHT, RIGHT, DROP
        
        # Standard epsilon-greedy with bias
        if random.random() < epsilon:
            # During exploration, bias toward horizontal movement
            if random.random() < 0.4:  # 40% of exploration
                return random.choice([0, 1])  # LEFT or RIGHT
            else:
                return self.env.action_space.sample()
        
        # Normal Q-network action
        with torch.no_grad():
            q_values = q_network(state)
            return q_values.argmax().item()
```

### Step 3: Update Training Loop
Modify the main training loop to use the new components:

```python
def train_tetris_agent():
    """Main training with anti-center-stacking measures"""
    
    # Initialize
    env = gym.make('tetris_gymnasium/Tetris')
    reward_shaper = TetrisRewardShaper()
    exploration = ExplorationStrategy(env)
    
    # Training hyperparameters
    epsilon = 1.0
    epsilon_decay = 0.9999  # Keep your current value
    epsilon_min = 0.1  # Don't go too low too fast
    
    # Metrics tracking
    metrics = {
        'episode': [],
        'reward': [],
        'lines_cleared': [],
        'columns_used': [],
        'outer_columns_used': [],
        'center_stacking_episodes': 0,
    }
    
    for episode in range(1000):
        obs, info = env.reset()
        episode_reward = 0
        prev_features = None
        action_counts = {i: 0 for i in range(8)}
        
        done = False
        while not done:
            # Select action with exploration strategy
            action = exploration.select_action(
                state=preprocess(obs),
                q_network=q_network,
                epsilon=epsilon,
                episode=episode
            )
            
            action_counts[action] += 1
            
            # Step environment
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Shape reward
            shaped_reward, features = reward_shaper.shape_reward(
                obs, action, env_reward, done, info, prev_features
            )
            prev_features = features
            
            # Store transition
            memory.push(obs, action, shaped_reward, next_obs, done)
            
            # Train
            if len(memory) > batch_size:
                train_step()
            
            episode_reward += shaped_reward
            obs = next_obs
        
        # Episode complete - analyze
        board = reward_shaper.extract_board(obs)
        heights = reward_shaper.get_column_heights(board)
        columns_used = sum(1 for h in heights if h > 0)
        outer_used = sum(1 for i in [0,1,2,7,8,9] if heights[i] > 0)
        
        # Check for center-stacking
        if outer_used == 0:
            metrics['center_stacking_episodes'] += 1
            print(f"⚠️ Episode {episode}: CENTER-STACKING DETECTED!")
            print(f"   Heights: {heights}")
            print(f"   Action distribution: {action_counts}")
            
            # Emergency intervention after 200 episodes
            if episode > 200 and metrics['center_stacking_episodes'] / (episode+1) > 0.5:
                print("🚨 EMERGENCY: >50% center-stacking after 200 episodes!")
                print("   Increasing exploration and outer column rewards...")
                epsilon = min(epsilon + 0.2, 1.0)  # Boost exploration
                reward_shaper.weights['outer_bonus'] *= 2  # Double outer bonus
        
        # Logging every 10 episodes
        if episode % 10 == 0:
            recent_center_stacking = sum(1 for e in metrics['episode'][-50:] 
                                        if e in metrics['center_stacking_episodes'])
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Columns used: {columns_used}/10")
            print(f"  Outer columns: {outer_used}/6")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Center-stacking rate (last 50): {recent_center_stacking/min(50, episode+1):.1%}")
            print(f"  Action distribution: {action_counts}")
        
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            save_checkpoint(episode, q_network, optimizer, metrics)
```

### Step 4: Create Validation Script
Create `validate_spreading.py` to verify the fix is working:

```python
def validate_spreading(model_path):
    """Test if the agent has learned to spread"""
    
    env = gym.make('tetris_gymnasium/Tetris')
    model = load_model(model_path)
    
    test_episodes = 10
    results = []
    
    for ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # Analyze final board
        board = extract_board(obs)
        heights = get_column_heights(board)
        outer_used = sum(1 for i in [0,1,2,7,8,9] if heights[i] > 0)
        
        results.append({
            'episode': ep,
            'heights': heights,
            'outer_used': outer_used,
            'is_spreading': outer_used >= 3,  # At least 3 outer columns used
        })
    
    # Report
    spreading_rate = sum(1 for r in results if r['is_spreading']) / test_episodes
    print(f"Spreading success rate: {spreading_rate:.1%}")
    
    if spreading_rate < 0.5:
        print("❌ Agent still center-stacking! Need more aggressive fixes.")
    else:
        print("✅ Agent has learned to spread!")
    
    return results
```

## Implementation Order

1. **First**: Implement `reward_shaping_v2.py` with research-based coefficients
2. **Second**: Add exploration strategy to force outer column usage in early training
3. **Third**: Update training loop with emergency interventions
4. **Fourth**: Run validation after every 100 episodes

## Key Parameters to Tune

If still center-stacking after implementation:
1. Increase `outer_bonus` weight from 0.3 to 1.0
2. Increase center-stacking penalty from -50 to -100
3. Force exploration for 200 episodes instead of 100
4. Keep epsilon at minimum 0.2 (not 0.1)

## Success Criteria

The agent should:
- Use at least 3 outer columns (0,1,2,7,8,9) in >50% of episodes by episode 200
- Use at least 4 outer columns in >75% of episodes by episode 500
- Have <20% pure center-stacking episodes after training

## Testing Command

```bash
# Train with new reward shaping
python train_agent.py --use-v2-rewards --force-exploration

# Validate every 100 episodes
python validate_spreading.py checkpoints/model_ep100.pt
```

## Emergency Fallback

If agent STILL center-stacks after all fixes:
1. Set epsilon=1.0 for episodes 0-50 (pure random to establish baseline)
2. Multiply all outer column rewards by 10x
3. Add immediate +5 reward for any LEFT/RIGHT action
4. Consider using curriculum learning with different board widths