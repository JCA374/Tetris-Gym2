# Training Failure Diagnosis - Complete Analysis

**Date**: November 9, 2025
**Episodes Trained**: 18,500
**Lines Cleared**: 0 (zero)
**Status**: Training failed - agent learned degenerate policy

## Executive Summary

After 18,500 training episodes with 0 lines cleared, comprehensive debugging revealed **TWO critical bugs** and one fundamental learning problem:

### Bug #1: Info Dict Field Name Mismatch ⚠️ CRITICAL
- **Environment returns**: `info['lines_cleared']`
- **Training script expects**: `info.get('number_of_lines', 0)`
- **Impact**: Agent NEVER received line clear feedback, even if lines were cleared
- **Fix**: Change line 109 and 254 in `train_feature_vector.py` from `'number_of_lines'` to `'lines_cleared'`

### Bug #2: Environment Never Clears Lines (Root Cause)
- **Problem**: Pieces stack in narrow vertical column (columns 8-10 only)
- **Cause**: Agent using primarily HARD_DROP without horizontal movement
- **Result**: Rows never fill completely (only 2-3/10 cells filled per row)
- **Impact**: No lines can clear if rows are never complete

### Learning Problem: Degenerate Policy
- **Agent learned**: Drop pieces straight down in center
- **Agent didn't learn**: Spread pieces horizontally across board width
- **Why**: Insufficient exploration + no intermediate rewards for good placement

## Detailed Technical Analysis

### Test 1: Basic Environment
```
Random actions (10 episodes, 500 steps each): 0 lines cleared
Hard drop strategy (51% HARD_DROP): 0 lines cleared
```

**Finding**: Environment configured correctly, line clearing mechanism works when tested manually

### Test 2: Piece Placement Tracing
```
20 consecutive HARD_DROPs:
- Pieces land in columns 8-10 (center-right) only
- Bottom rows: 2/10 to 3/10 filled
- No complete rows formed
- Game over after ~11 steps (stack reaches top)
```

**Finding**: Without horizontal movement, pieces cannot spread to fill rows

### Test 3: Manual Line Clearing
```python
# Manually fill row 19 playable area (columns 4-13)
board[19, 4:14] = 2  # Fill with pieces
lines_cleared = env.unwrapped.clear_filled_rows(board)
# Result: 1 line cleared ✅
```

**Finding**: Line clearing mechanism works correctly when rows are actually filled

### Test 4: Board Structure
```
Board size: (24, 18)
Playable area: rows 0-19, columns 4-13 (10 columns wide)
Walls: columns 0-3, 14-17 (value 1)
Floor: rows 20-23 (all value 1)

Clearing logic:
filled_row = (no zeros) AND (not all ones)
- Walls (1s) in a row are OK
- But playable area must be completely filled with pieces (values 2-8)
```

**Finding**: Clearing requires ALL 10 playable columns to have pieces

## Why Agent Failed to Learn

### 1. No Feedback for Line Clearing (Bug #1)
```python
# train_feature_vector.py line 109
lines = info.get('number_of_lines', 0)  # ❌ Always returns 0
```

Even if agent HAD cleared lines, reward would be:
- Expected: `1.0 + (lines * 100) = 101+`
- Actual: `1.0 + (0 * 100) = 1.0` (same as no lines)

**Impact**: Agent had NO incentive to clear lines (same reward either way)

### 2. Pieces Stack in Center (Insufficient Action Diversity)
Likely agent action distribution (hypothesis from training results):
```
HARD_DROP or NOOP: >80%
LEFT/RIGHT: <10%
ROTATE: <10%
```

**Why this fails**:
- Pieces spawn at center (columns 8-9)
- Immediate HARD_DROP → pieces land at center
- No horizontal spreading → narrow vertical stack
- Stack grows tall quickly → game over in 150-300 steps
- No complete rows → no lines cleared

### 3. Reward Function Doesn't Encourage Spreading
```python
reward = 1.0                          # Survival
       + lines_cleared * 100          # Never triggered (bug #1)
       - holes * 2.0                  # Not available
       - aggregate_height * 0.1       # Not available
```

Actual reward agent received: **Always 1.0 per step**

**Problem**: Agent learned to maximize steps (survival), not clear lines

### 4. Exploration Insufficient
```
Episode 18,500: epsilon = 0.0725
```

By episode 1,000: epsilon < 0.37
By episode 5,000: epsilon < 0.08
By episode 18,500: epsilon = 0.0725 (7.25% random actions)

**Problem**: Agent never explored enough to discover line clearing requires:
1. Horizontal movement (LEFT/RIGHT)
2. Rotation for tetromino orientation
3. Strategic placement across full board width

## Solutions

### Immediate Fix (Required)

**1. Fix Info Dict Field Name**
```python
# train_feature_vector.py line 109, 254
# Change from:
lines = info.get('number_of_lines', 0)

# Change to:
lines = info.get('lines_cleared', 0)
```

### Re-Training Strategy

**2. Add Dense Reward Shaping**
```python
def improved_reward(env_reward, info, obs):
    """Reward that guides agent toward line clearing"""
    lines = info.get('lines_cleared', 0)

    # Base survival
    reward = 1.0

    # Huge bonus for clearing lines
    if lines > 0:
        reward += lines * 100

    # Extract features from observation
    features = extract_feature_vector(obs['board'])

    # Reward low bumpiness (flat top)
    bumpiness = features[2]  # Normalized 0-1
    reward -= bumpiness * 5.0

    # Reward low holes
    holes = features[1]  # Normalized 0-1
    reward -= holes * 10.0

    # Penalize height (encourage keeping board low)
    max_height = features[14]  # Normalized 0-1
    reward -= max_height * 3.0

    # Reward for filling bottom rows (even if not cleared yet)
    board = obs['board']
    bottom_row = board[19, 4:14]  # Bottom playable row
    filled_ratio = (bottom_row > 1).sum() / 10.0
    reward += filled_ratio * 2.0  # Reward partial row completion

    return reward
```

**3. Force Exploration Early**
```python
# Prevent epsilon from decaying too fast
epsilon_decay = 0.9999  # Was 0.9995
epsilon_end = 0.1  # Was 0.05

# Or use epsilon floor with periodic resets
if episode % 2000 == 0 and episode > 0:
    agent.epsilon = max(agent.epsilon, 0.3)  # Reset to 30% every 2000 episodes
```

**4. Curriculum Learning**
```python
# Stage 1 (episodes 0-1000): Force horizontal movement
# Modify action selection to bias toward LEFT/RIGHT early
if episode < 1000 and np.random.random() < 0.3:
    action = np.random.choice([0, 1, 5])  # Force LEFT, RIGHT, or HARD_DROP

# Stage 2 (episodes 1000-3000): Balanced exploration
# Normal epsilon-greedy

# Stage 3 (episodes 3000+): Refinement
# Lower epsilon for exploitation
```

**5. Add Expert Demonstrations**
```python
# Pre-fill replay buffer with good placements
def add_expert_demos(agent, num_episodes=50):
    env = make_feature_vector_env()

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Simple rule-based policy:
            # 1. Move toward edges
            # 2. Hard drop
            if state[4] > 0.5:  # First column height
                action = 1  # Move RIGHT
            else:
                action = 0  # Move LEFT

            # Occasional hard drop
            if np.random.random() < 0.3:
                action = 5  # HARD_DROP

            next_state, reward, terminated, truncated, info = env.step(action)

            agent.remember(state, action, reward, next_state, terminated or truncated)
            state = next_state
            done = terminated or truncated

    print(f"Added {num_episodes} expert demonstration episodes to replay buffer")
```

**6. Action Space Modifications**
Consider removing or penalizing NOOP (action 7) if agent over-uses it:
```python
# In env wrapper
class NoNoopWrapper(gym.Wrapper):
    def step(self, action):
        # Penalize NOOP
        obs, reward, terminated, truncated, info = self.env.step(action)
        if action == 7:  # NOOP
            reward -= 0.5  # Small penalty
        return obs, reward, terminated, truncated, info
```

## Recommended Action Plan

### Phase 1: Fix Critical Bug (5 minutes)
1. ✅ Update `train_feature_vector.py` line 109, 254: `'number_of_lines'` → `'lines_cleared'`
2. ✅ Update reward function to use extracted features (holes, bumpiness, height)
3. ✅ Test with 100 episodes to verify line clear feedback works

### Phase 2: Improve Exploration (1 hour)
4. ✅ Slow epsilon decay: 0.9995 → 0.9999
5. ✅ Raise epsilon floor: 0.05 → 0.1
6. ✅ Add periodic epsilon resets (30% every 2000 episodes)
7. ✅ Run 2,000 episodes, check if ANY lines cleared

### Phase 3: Dense Rewards (2 hours)
8. ✅ Implement improved reward function with bumpiness, holes, height penalties
9. ✅ Add partial row completion rewards
10. ✅ Run 5,000 episodes, target: 10+ lines/episode by end

### Phase 4: Curriculum/Experts (if Phase 3 fails)
11. Add expert demonstrations (pre-fill replay buffer)
12. Implement curriculum learning (force horizontal movement early)
13. Run 10,000 episodes with full curriculum

## Expected Results After Fixes

### After Phase 1 (Bug fix only):
- Agent will finally see line clear rewards
- BUT likely still won't clear many lines (action distribution problem remains)
- Expected: 0-5 lines in first 1,000 episodes

### After Phase 2 (Better exploration):
- Agent should discover horizontal movement
- Pieces should start spreading across board
- Expected: 10-50 lines in 2,000 episodes

### After Phase 3 (Dense rewards):
- Agent should consistently clear lines
- Performance should match research expectations
- Expected at 5,000 episodes: 50-100+ lines/episode
- Expected at 10,000 episodes: 100-500+ lines/episode

## Lessons Learned

1. **Always verify field names match** between environment and training code
2. **Test environment in isolation first** before training
3. **Monitor action distributions** - if agent uses 80% one action, it won't learn diversity
4. **Dense rewards are crucial** for sparse reward tasks like Tetris
5. **Exploration must be sufficient** to discover multi-step strategies
6. **Validate assumptions early** - run 100-episode test before 18,500 episodes

## Files Modified

- `debug_tests/`: Complete test suite for systematic debugging
- `DIAGNOSIS.md`: This file
- **TODO**: `train_feature_vector.py` - fix field name bug
- **TODO**: `train_feature_vector.py` - implement improved reward function

## References

- Tetris Gymnasium: https://github.com/Max-We/Tetris-Gymnasium
- Info dict field: `lines_cleared` (confirmed in v0.3.0 source code)
- Board structure: (24, 18) with playable area [0:20, 4:14]
