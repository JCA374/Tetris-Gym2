# Training Review: feature_110k_fixed (47,000 Episodes)

**Review Date**: November 10, 2025
**Training Duration**: Nov 9 23:29 - Nov 10 07:13 (~8 hours)
**Total Episodes**: 47,000
**Status**: ❌ FAILED - No learning progress

## Critical Stats

| Metric | Value | Status |
|--------|-------|--------|
| **Total Lines Cleared** | **0** | ❌ CRITICAL |
| **Episodes with Lines** | 0 / 47,000 (0%) | ❌ CRITICAL |
| **Avg Reward/Episode** | ~130-180 | ⚠️ (survival only) |
| **Reward == Steps** | 47,787 / 48,000 (99.56%) | ❌ (no line clears) |
| **Epsilon (episode 47k)** | 0.351 (35% exploration) | ⚠️ (should be sufficient) |
| **Memory Buffer** | 100,000 (full) | ✅ |

## Training Progress Analysis

### Epsilon Decay
```
Episode     1: Epsilon = 1.000
Episode  5000: Epsilon = 0.922
Episode 10000: Epsilon = 0.844
Episode 15000: Epsilon = 0.766
Episode 20000: Epsilon = 0.688
Episode 25000: Epsilon = 0.610
Episode 30000: Epsilon = 0.532
Episode 35000: Epsilon = 0.455
Episode 40000: Epsilon = 0.391
Episode 45000: Epsilon = 0.363
Episode 47000: Epsilon = 0.351
```

**Analysis**: Epsilon decay is working correctly. At 35% random exploration after 47k episodes, the agent has tried millions of random actions but still hasn't cleared a single line.

### Reward Progress
```
Episode     1: Reward =  92 (92 steps)
Episode  5000: Reward =  90 (90 steps)
Episode 20000: Reward = 151 (151 steps)
Episode 35000: Reward = 204 (204 steps)
Episode 47000: Reward = 102-250 (varies)
```

**Analysis**: Rewards are EXACTLY equal to steps in 99.56% of episodes, confirming:
- Agent receives +1.0 per step (survival reward)
- NO line clear bonuses ever triggered (+100 per line)
- Agent learned to survive slightly longer (90 → 150-200 steps average)

### What the Agent Learned

Based on 47,000 episodes of training:

✅ **Agent DID learn**:
- Survive longer (100-250 steps vs initial 60-90)
- Avoid immediate game over
- Basic piece placement (pieces stack somewhat)

❌ **Agent did NOT learn**:
- Horizontal movement strategy
- Rotation for better fits
- Filling complete rows
- Clearing lines (core objective)

## Root Cause: Why No Lines After 47K Episodes?

### 1. Line Clearing Requires Complex Multi-Step Strategy

Tetris line clearing is NOT discoverable through random exploration because:

**Required actions for one line clear**:
1. Move LEFT/RIGHT to position pieces across full 10-column width
2. ROTATE pieces to fit gaps and create flat surface
3. Repeat for multiple pieces to fill a complete row
4. HARD_DROP to finalize placement

**Random exploration result**:
- Pieces mostly drop in center (spawn position)
- Occasional LEFT/RIGHT but not coordinated
- Result: Narrow vertical stack (only 2-3 columns filled)
- Rows never complete → no lines clear

### 2. Sparse Reward Problem

Current reward structure:
```python
reward = 1.0                    # Every step
       + lines_cleared * 100    # Never triggers (no lines)
       - holes * 2.0            # Not in info dict
       - height * 0.1           # Not in info dict
```

**Problem**: Agent only sees +1.0 every step, regardless of action quality
- Good placement (spreading pieces): +1.0
- Bad placement (center stack): +1.0
- Moving toward edges: +1.0
- Standing still: +1.0

**Result**: No incentive to learn strategic placement

### 3. Observation-Action Disconnect

The feature vector includes:
- Column heights (10 values)
- Holes, bumpiness, wells
- Aggregate height, max/min

**BUT** the agent hasn't learned that:
- High bumpiness → should use ROTATE/horizontal movement
- Uneven heights → should fill gaps
- Pieces at center → should spread left/right

**Why**: No reward signal guides these associations

## What Needs to Change

### Option 1: Dense Reward Shaping (RECOMMENDED)

Add intermediate rewards that guide the agent:

```python
def shaped_reward(info, obs):
    """Dense rewards that guide toward line clearing"""
    lines = info.get('lines_cleared', 0)

    # Base survival
    reward = 1.0

    # HUGE bonus for lines (keep this)
    if lines > 0:
        reward += lines * 100

    # Extract features from observation
    board = obs['board']
    playable = board[0:20, 4:14]  # Playable area

    # === NEW: Dense intermediate rewards ===

    # 1. Reward for filling bottom rows (even without clearing)
    for row_idx in range(19, 14, -1):  # Bottom 5 rows
        row = playable[row_idx, :]
        filled_ratio = (row > 1).sum() / 10.0
        weight = (20 - row_idx)  # Bottom rows more important
        reward += filled_ratio * weight * 0.5

    # 2. Reward for low bumpiness (flat surface)
    heights = get_column_heights(playable)
    bumpiness = calculate_bumpiness(heights)
    reward -= bumpiness * 0.3  # Penalize bumpy surface

    # 3. Reward for low holes
    holes = count_holes(playable)
    reward -= holes * 2.0  # Penalize holes

    # 4. Reward for spreading pieces (using more columns)
    columns_used = sum(1 for h in heights if h > 0)
    reward += columns_used * 0.2  # Reward using more columns

    # 5. Penalize extreme height
    max_height = max(heights)
    if max_height > 15:  # Danger zone
        reward -= (max_height - 15) * 1.0

    return reward
```

**Expected impact**:
- Agent learns to spread pieces (columns_used reward)
- Agent learns to keep surface flat (bumpiness penalty)
- Agent learns to fill bottom rows (partial completion reward)
- Lines clearing becomes natural extension

### Option 2: Curriculum Learning

**Phase 1 (Episodes 0-5,000)**: Force horizontal exploration
```python
if episode < 5000:
    # Bias action selection toward horizontal movement
    if random() < 0.4:
        action = choice([LEFT, RIGHT])  # Force spreading
```

**Phase 2 (Episodes 5,000-15,000)**: Normal exploration
- Standard epsilon-greedy
- Agent should have discovered horizontal movement

**Phase 3 (Episodes 15,000+)**: Refinement
- Lower epsilon (more exploitation)
- Agent optimizes learned strategy

### Option 3: Expert Demonstrations

Pre-fill replay buffer with simple rule-based plays:

```python
def simple_expert_policy(board_state):
    """Simple rule: spread pieces across width"""
    column_heights = get_column_heights(board_state)

    # Find lowest column
    lowest_col = argmin(column_heights)

    # Move toward lowest column
    if current_col < lowest_col:
        return RIGHT
    elif current_col > lowest_col:
        return LEFT
    else:
        return HARD_DROP
```

Add 100-500 episodes of this to replay buffer before training starts.

**Expected impact**: Agent sees examples of horizontal movement and spreading

### Option 4: Action Space Modification

Limit or bias action space early in training:

```python
# Remove NOOP (action 7) - agent shouldn't waste time
# Or heavily penalize it
if action == NOOP:
    reward -= 0.5

# Or create action hierarchy
MOVEMENT_ACTIONS = [LEFT, RIGHT, ROTATE_CW, ROTATE_CCW]
PLACEMENT_ACTIONS = [HARD_DROP, DOWN]

# Phase 1: Force movement before placement
if steps_this_piece < 2:
    action = choice(MOVEMENT_ACTIONS)  # Must move first
```

## Recommended Action Plan

### Immediate (2 hours)

1. **Implement Dense Reward Shaping** (Option 1)
   - Add the shaped_reward function above
   - Test with 1,000 episodes
   - Should see non-integer rewards (sign of working)

2. **Verify Rewards Working**
   ```bash
   # Look for rewards != steps
   tail -100 logs/test_shaped/episode_log.csv | awk -F',' '{if ($10 != $12) print "Episode", $3, "Reward:", $10, "Steps:", $12}'
   ```

3. **Short Test Run** (1,000 episodes, ~20 minutes)
   ```bash
   .venv/bin/python train_feature_vector.py \
       --episodes 1000 \
       --experiment_name shaped_rewards_test \
       --force_fresh
   ```

   **Success criteria**:
   - Rewards should vary (not always equal steps)
   - Some rewards should be > steps (positive from bumpiness/spreading)
   - Some rewards should be < steps (negative from holes/height)

### Short-term (1-2 days)

4. **Medium Training Run** (10,000 episodes, ~3 hours)
   - With shaped rewards
   - Monitor for first line clear
   - Should see by episode 5,000-7,000

5. **Add Curriculum if Needed**
   - If still 0 lines after 10k with shaped rewards
   - Implement forced horizontal movement (Option 2)

### Long-term (1 week)

6. **Full Training Run** (50,000-100,000 episodes)
   - With shaped rewards + any curriculum
   - Target: 10-50 lines/episode by end
   - Target: 100+ lines/episode by 100k (if trained that long)

## Key Metrics to Monitor

During next training run, watch for:

✅ **Positive signs**:
- Rewards ≠ steps (shaped rewards working)
- First line clear within 5,000 episodes
- Lines/episode increasing over time
- Epsilon decay as planned

❌ **Warning signs**:
- Reward still == steps after 1,000 episodes (shaped rewards broken)
- No lines after 10,000 episodes (need curriculum)
- Lines/episode not increasing (local optimum found)

## Comparison to Expected Performance

### Research Benchmarks (Feature Vector DQN)
| Episodes | Expected Lines/Ep | Actual Lines/Ep | Status |
|----------|------------------|-----------------|--------|
| 5,000 | 5-20 | 0 | ❌ 100% behind |
| 10,000 | 20-50 | 0 | ❌ 100% behind |
| 25,000 | 50-200 | 0 | ❌ 100% behind |
| 47,000 | 100-500 | 0 | ❌ 100% behind |

### What Good Training Looks Like

```
Episode  1000: Lines = 0-1 (random luck)
Episode  2000: Lines = 1-3 (discovering clearing)
Episode  5000: Lines = 10-30 (consistent clearing)
Episode 10000: Lines = 50-100 (strategic play)
Episode 25000: Lines = 100-300 (expert level)
```

## Files to Modify

1. **train_feature_vector.py**
   - Replace `simple_reward()` function (lines 92-127)
   - Add dense reward shaping as shown above

2. **src/feature_vector.py** (already correct)
   - Has get_column_heights(), count_holes(), etc.
   - Can be imported into shaped_reward function

## Conclusion

**Status**: Current training has completely failed to learn the task

**Root cause**: Sparse reward signal + complex multi-step requirement

**Solution**: Dense reward shaping to guide agent toward line clearing behaviors

**Next step**: Implement shaped rewards and run 1,000-episode test

**Time investment**: 2 hours to implement + test, then 3 hours for 10k episode run

**Expected outcome**: First lines cleared within 5,000 episodes, consistent clearing by 10,000 episodes

---

**Bottom Line**: The agent has run 47,000 episodes without learning anything useful. The current approach will NOT work even at 100k or 1 million episodes. Dense reward shaping is REQUIRED to make progress.
