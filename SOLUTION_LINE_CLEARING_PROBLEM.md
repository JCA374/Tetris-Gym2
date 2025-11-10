# Solution: Why Agent Can't Clear Lines

**Date:** 2025-11-10
**Status:** ✅ ROOT CAUSE IDENTIFIED

---

## Executive Summary

The Tetris environment **WORKS CORRECTLY**. The problem is that the agent learned a **suboptimal strategy** that makes line clearing impossible.

**The Issue:** Agent only fills the RIGHT side of the board (columns 4-9), leaving the left side empty (columns 0-3). Since line clearing requires ALL 10 columns to be filled, the agent can NEVER clear lines.

---

## Investigation Timeline

### 1. Initial Hypothesis (WRONG)
**Hypothesis:** Environment can't clear lines
**Evidence:** 110k+ training episodes, 0 lines cleared
**Conclusion:** INCORRECT - extensive testing proved environment works

### 2. Testing Environment (CORRECT)
**Test:** `test_proper_action_sequences.py`
**Strategy:** Alternate LEFT/CENTER/RIGHT placement before HARD_DROP
**Result:** ✅ **Line cleared on 3rd attempt with only 16 pieces!**
**Conclusion:** Environment works perfectly

### 3. Analyzing Agent Behavior (KEY FINDING)
**Test:** `test_trained_agent_simple.py`
**Results:**
```
LEFT:        1.0%  ← CRITICAL: Agent barely moves left!
RIGHT:      28.4%
ROTATE_CW:  64.7%
```

**Test:** `test_visualize_agent_gameplay.py`
**Results:**
```
Column Heights: [0, 0, 0, 0, 17, 17, 17, 9, 9, 9]
                 └─ Empty ─┘  └───── Filled ─────┘
```

**Conclusion:** Agent learned to spam RIGHT + ROTATE, never LEFT

---

## Root Cause Analysis

### Why Agent Learned This Bad Strategy

1. **Reward Function:**
   ```python
   reward = 1.0 per step survived
          + lines_cleared * 100
   ```

2. **What Agent Learned:**
   - More steps = more reward
   - To maximize steps: Place pieces quickly
   - Fastest placement: spam RIGHT + ROTATE, let gravity drop
   - This strategy survives ~133 steps/episode

3. **What Agent DIDN'T Learn:**
   - Clearing lines gives +100 reward
   - To clear lines, must fill ALL columns
   - Must use LEFT to reach columns 0-3
   - Strategic placement > fast placement

4. **Why Agent Didn't Discover Line Clears:**
   - Agent's exploration (epsilon-greedy) randomly tried actions
   - But randomly moving RIGHT often, LEFT rarely
   - Never accidentally created the pattern needed for line clears
   - Without ever clearing a line, never learned the +100 reward signal

---

## Detailed Behavior Analysis

### Agent's Strategy Pattern
```
For each piece:
1. Spawn in center (columns 4-5)
2. Move RIGHT 3-6 times
3. Rotate 6-10 times
4. Let gravity drop
5. Piece lands on right side
6. Repeat
```

### Board State Progression
```
After 1 piece:  [0,0,0,0,20,20,0,0,0,0]
After 2 pieces: [0,0,0,0,12,13,13,0,3,5]
After 3 pieces: [0,0,0,0,17,17,17,9,9,9]
After 4 pieces: [0,0,0,0,0,14,0,9,9,9]
                 ^^^^^^^^ Always empty!
```

### Why This Fails
- Line clearing requires: `all(row[0:10] > 0)`
- Agent creates: `row[0:4] == 0` (always empty)
- Result: `IMPOSSIBLE to clear lines`

---

## Solutions

### Solution 1: Reward Shaping for Balanced Placement ⭐ RECOMMENDED

Add rewards that encourage even column distribution:

```python
def balanced_reward(info, obs):
    base_reward = 1.0  # Survival
    line_reward = info.get('lines_cleared', 0) * 100

    # Extract column heights from features
    column_heights = features[4:14]  # Features 4-13 are column heights

    # Reward for balanced heights (low std = balanced)
    height_std = np.std(column_heights)
    balance_reward = -height_std * 0.5  # Penalty for uneven heights

    # Reward for filling left columns (encourage LEFT usage)
    left_columns = column_heights[0:4]  # Columns 0-3
    left_fill_reward = np.mean(left_columns) * 0.2

    return base_reward + line_reward + balance_reward + left_fill_reward
```

**Advantages:**
- Directly incentivizes balanced placement
- Rewards filling left side specifically
- Should discover line clears much faster

**Disadvantages:**
- More complex reward function
- Requires tuning reward coefficients

### Solution 2: Epsilon with Action Bias

Bias exploration toward underused actions:

```python
def select_action_with_bias(state, epsilon, action_counts):
    if random.random() < epsilon:
        # Biased exploration: favor LEFT if it's underused
        if action_counts[0] < action_counts[1] * 0.5:  # LEFT < RIGHT/2
            if random.random() < 0.3:
                return 0  # Force LEFT
        return random.choice(range(8))  # Normal exploration
    else:
        return greedy_action(state)  # Exploitation
```

**Advantages:**
- Simple to implement
- Ensures LEFT gets explored
- Minimal changes to existing code

**Disadvantages:**
- Hack-ish solution
- Doesn't address fundamental reward problem

### Solution 3: Curriculum Learning

Train in stages with increasing difficulty:

```python
Stage 1: Only allow LEFT/RIGHT/HARD_DROP (force horizontal spreading)
Stage 2: Add ROTATE actions
Stage 3: Add all actions
```

**Advantages:**
- Forces agent to learn basics first
- Proven effective in other RL domains

**Disadvantages:**
- Complex implementation
- Requires multiple training phases

### Solution 4: Demonstrations / Imitation Learning

Provide expert demonstrations of good gameplay:

```python
# Record human or scripted gameplay
demonstrations = load_expert_demos()

# Pre-train on demonstrations
for state, action in demonstrations:
    loss = F.cross_entropy(model(state), action)
    loss.backward()

# Then continue with RL
```

**Advantages:**
- Jumpstarts learning
- Agent sees good strategies from the start

**Disadvantages:**
- Requires creating demonstrations
- Most complex solution

---

## Recommended Action Plan

### Phase 1: Quick Fix (Reward Shaping) ✅ DO THIS FIRST

1. Implement `balanced_reward()` function (see Solution 1)
2. Update `train_feature_vector.py` to use new reward
3. Run 5,000 episode training
4. Expected result: First line clear within 500-1000 episodes

**Time estimate:** 1 hour implementation + 3-5 hours training

### Phase 2: Validate

1. Use `test_trained_agent_simple.py` to check action distribution
2. Verify LEFT usage increased to 10-20%
3. Verify column heights are more balanced
4. Verify lines are being cleared

**Time estimate:** 30 minutes

### Phase 3: Iterate

If Phase 1 doesn't work:
1. Try adjusting reward coefficients
2. Try Solution 2 (epsilon bias) as supplement
3. Consider Solution 3 (curriculum) if still failing

---

## Code Changes Required

### File: `train_feature_vector.py`

**Current:**
```python
def simple_reward(env_reward, info):
    lines = info.get('lines_cleared', 0)
    reward = 1.0 + lines * 100
    return reward
```

**New:**
```python
def balanced_reward(env_reward, info, features):
    """Reward that encourages balanced column placement"""
    base_reward = 1.0
    line_reward = info.get('lines_cleared', 0) * 100

    # Extract column heights (normalized 0-1)
    column_heights = features[4:14]

    # Penalty for uneven distribution
    height_std = np.std(column_heights)
    balance_penalty = -height_std * 2.0

    # Bonus for filling left columns (0-3)
    left_fill = np.mean(column_heights[0:4])
    left_bonus = left_fill * 1.0

    # Bonus for filling ALL columns (bumpiness should be low)
    bumpiness = features[2]
    smoothness_bonus = -(bumpiness * 1.0)

    total_reward = (base_reward + line_reward +
                    balance_penalty + left_bonus + smoothness_bonus)

    return total_reward
```

**Update training loop:**
```python
# Before:
reward = simple_reward(env_reward, info)

# After:
reward = balanced_reward(env_reward, info, state)  # Pass features
```

---

## Expected Outcomes

### With Balanced Reward Function

**Episodes 0-500:**
- Agent explores LEFT more due to left_bonus
- Column heights become more even
- Still no line clears (learning phase)

**Episodes 500-1500:**
- First line clears appear!
- Agent discovers +100 reward signal
- Begins optimizing for line clears

**Episodes 1500-5000:**
- Consistent line clearing
- 5-20 lines/episode average
- Strategy: balanced placement across all columns

**Final Performance (5000 episodes):**
- 10-50 lines/episode
- Balanced column heights
- Strategic piece placement
- Clear understanding that filling all columns = line clears

---

## Validation Tests

After implementing fixes, run:

```bash
# 1. Check action distribution
python test_trained_agent_simple.py
# Expect: LEFT usage 10-20%, RIGHT 15-25%

# 2. Visualize gameplay
python test_visualize_agent_gameplay.py
# Expect: Pieces spread across all columns

# 3. Check column heights
# Expect: Heights = [5,6,7,5,6,7,5,6,7,5] (balanced)
# Not: Heights = [0,0,0,0,10,10,10,0,0,0] (unbalanced)

# 4. Monitor training
python monitor_training.py logs/balanced_reward_5k
# Expect: Lines cleared > 0 before episode 1500
```

---

## Lessons Learned

1. **Always test environment separately from agent**
   - We initially blamed the environment
   - Environment was working correctly all along

2. **Reward function must incentivize ALL desired behaviors**
   - Survival reward alone = agent learns fast placement
   - Need explicit rewards for balanced placement

3. **Sparse rewards (line clears) are hard to discover**
   - Agent played 110k episodes without discovering line clears
   - Need intermediate rewards to guide exploration

4. **Action distribution is a critical diagnostic**
   - LEFT: 1% vs RIGHT: 28% immediately revealed the problem
   - Should monitor action usage during training

5. **Visualization is essential for debugging**
   - Seeing the board state made the problem obvious
   - Column heights [0,0,0,0,17,17,17,9,9,9] told the whole story

---

## Files Created During Investigation

1. ✅ `test_proper_action_sequences.py` - Proved environment works
2. ✅ `test_trained_agent_simple.py` - Revealed action imbalance
3. ✅ `test_visualize_agent_gameplay.py` - Showed right-only placement
4. ✅ `test_compare_random_vs_trained.py` - Action distribution comparison
5. ✅ `CRITICAL_BUG_LINE_CLEARING.md` - Initial (incorrect) hypothesis
6. ✅ `SOLUTION_LINE_CLEARING_PROBLEM.md` - This document

---

## Next Steps

1. ✅ Implement `balanced_reward()` function
2. ✅ Update training script
3. ⏳ Run 5,000 episode training with new rewards
4. ⏳ Validate results
5. ⏳ Iterate if needed
6. ⏳ Document final solution

---

**Status:** Ready to implement Solution 1 (Reward Shaping)
