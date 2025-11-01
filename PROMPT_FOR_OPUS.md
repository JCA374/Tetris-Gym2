# Tetris RL Agent - Center-Stacking Problem

## Problem Statement

My Tetris DQN agent **consistently center-stacks** pieces in columns 3-6, leaving outer columns (0-2, 7-9) empty. This persists after 500+ episodes despite multiple reward shaping fixes.

---

## Current Training Results (Episode 1-500)

### Pattern Analysis:
```
Center-stacking (5-6 outer columns unused): 461 episodes (92%)
  - Average holes: 28.1
  - Average reward/step: -23.89

Spreading (0-2 outer columns unused): 4 episodes (0.8%)
  - Average holes: 54.5
  - Average reward/step: -50.38

❌ Spreading gives 26.5 points WORSE reward than center-stacking!
```

### Sample Episodes:
```
Ep    | Cols | Outer | Holes | R/step  | Heights
------|------|-------|-------|---------|---------------------------
1     |  8   |   2   |  43   | -37.17  | [0, 2, 3, 10, 16, 18, 19, 15, 12, 0]
11    |  5   |   5   |  30   | -66.00  | [0, 0, 6, 7, 20, 20, 17, 0, 0, 0]
21    |  7   |   3   |  59   | -49.99  | [0, 0, 9, 18, 18, 19, 19, 20, 8, 0]
111   |  8   |   2   |  80   | -61.04  | [12, 12, 12, 19, 20, 20, 15, 14, 0, 0]
161   |  4   |   6   |  21   | -65.95  | [0, 0, 0, 6, 18, 18, 19, 0, 0, 0]
```

**Key observation:** When agent tries to spread (Ep 1, 21, 111), it creates MORE holes (43-80) and gets worse rewards!

---

## Current Reward Shaping (src/reward_shaping.py)

```python
def overnight_reward_shaping(obs, action, reward, done, info):
    board = extract_board_from_obs(obs)  # 20x10 playable area

    shaped = float(reward) * 100.0  # Amplify env reward

    # Calculate metrics
    agg_h   = calculate_aggregate_height(board)       # 0..200
    holes   = count_holes(board)                      # 0..~200
    bump    = calculate_bumpiness(board)              # 0..~100
    wells   = calculate_wells(board)                  # 0..~100
    spread  = calculate_horizontal_distribution(board) # 0..1
    heights = get_column_heights(board)

    # PENALTIES
    shaped -= 0.05 * agg_h           # Height penalty
    shaped -= 0.75 * holes           # Holes penalty (REDUCED from 2.0)
    shaped -= 0.5 * bump             # Bumpiness penalty
    shaped -= 0.10 * wells

    # ANTI-CENTER-STACKING REWARDS
    shaped += 25.0 * spread          # Spread bonus (INCREASED from 15.0)

    columns_used = sum(1 for h in heights if h > 0)
    shaped += columns_used * 6.0     # Column usage (INCREASED from 4.0)

    outer_columns = [0, 1, 2, 7, 8, 9]
    outer_unused = sum(1 for c in outer_columns if heights[c] == 0)
    shaped -= outer_unused * 8.0     # Outer penalty (INCREASED from 5.0)

    height_std = float(np.std(heights))
    shaped -= 3.0 * height_std       # Height concentration (INCREASED from 2.0)

    # SURVIVAL & LINE CLEARS
    steps = int(info.get("steps", 0))
    shaped += min(steps * 0.2, 20.0) # Survival bonus

    lines = int(info.get("lines_cleared", 0))
    if lines > 0:
        shaped += lines * 80.0
        if lines == 4:
            shaped += 120.0

    if done:
        shaped -= 5.0                # Death penalty

    return float(np.clip(shaped, -150.0, 600.0))
```

---

## What We've Tried

### Fix #1: Strengthened penalties 17-20x
- **Result:** All rewards hit -100 clamp, eliminated gradient
- **Failed:** No difference between good and bad play

### Fix #2: Calibrated penalties, removed clamp
- **Result:** Gradient restored (76 points)
- **Failed:** Still center-stacking (holes penalty too strong)

### Fix #3 (Current): Reduced holes, increased anti-center rewards
- Reduced holes penalty: -2.0 → -0.75
- Increased spread bonus: 15.0 → 25.0
- Increased column usage: 4.0 → 6.0
- Increased outer penalty: 5.0 → 8.0
- Increased height std: 2.0 → 3.0
- **Result:** Gradient 112 points in test
- **Failed:** Still center-stacking (spreading creates 2x more holes)

---

## Test Results (Clean Boards)

Running `tests/test_reward_system_complete.py` with perfect boards (no training):

```
Scenario                    | Reward/Step | Holes
----------------------------|-------------|-------
Perfect balance [8,8,8...] |   +68.19    |   0
Good spreading [2,4,7,10..] |   +40.73    |   0
Slight spread [0,0,2,10..]  |   -30.30    |   0
Center-stack [0,0,0,15,19..]|   -71.40    |   0
Empty board                 |   -35.30    |   0

Gradient (good → center): 112 points ✅
```

**Test shows spreading SHOULD be better, but in training it's not!**

---

## The Core Issue

**Training vs Test Mismatch:**

| Scenario | Test (0 holes) | Training (with holes) | Difference |
|----------|----------------|----------------------|------------|
| Center-stacking | -71.40/step | -23.89/step | **+47.51** |
| Good spreading  | +40.73/step | -50.38/step | **-91.11** |

**Diagnosis:**
1. Agent isn't skilled enough to spread cleanly yet
2. Spreading creates 2x more holes (28 → 55 holes)
3. Extra holes penalty: -0.75 × 26.4 = **-19.8**
4. Anti-center rewards: ~**+20-30** (not enough!)
5. **Net: Spreading still 26.5 points worse**

---

## Agent Setup

- **Model:** DQN with 4-channel CNN (board, active piece, holder, queue)
- **Epsilon:** Starting at 1.0, decay 0.9999
- **Episodes:** 500 completed, 0 lines cleared
- **Learning:** Batch size 32, learning every 4 steps, gamma 0.99
- **Actions:** LEFT, RIGHT, DOWN, ROTATE_CW, ROTATE_CCW, HARD_DROP, SWAP, NOOP

---

## Questions for Diagnosis

1. **Is the holes penalty still too strong?** (-0.75 per hole)
   - Should I reduce it further to -0.5 or -0.3?
   - Or is the fundamental approach wrong?

2. **Are the anti-center-stacking rewards fundamentally flawed?**
   - Should I reward based on potential, not actual state?
   - Should I ignore holes in early training?

3. **Is this a chicken-and-egg problem?**
   - Agent needs to spread to learn spreading doesn't create holes
   - But spreading creates holes, so agent won't try it
   - How to break this cycle?

4. **Should I use curriculum learning?**
   - Start with only spreading rewards, add holes penalty later?
   - Use different reward functions for early vs late training?

5. **Is the gradient calculation method wrong?**
   - Test uses perfect boards (0 holes)
   - Training has realistic holes
   - Should I calculate gradient with expected holes per strategy?

6. **Other potential issues:**
   - Q-network not learning properly?
   - Exploration not diverse enough?
   - Action space issues?
   - State representation missing information?

---

## What I Need

**A clear diagnosis of:**
1. Why spreading still gives worse rewards than center-stacking
2. Whether the reward function architecture is fundamentally flawed
3. Specific fixes to try next (with concrete numbers if penalty adjustments)
4. Whether this requires a different approach (curriculum, shaped exploration, etc.)

**Please be specific with:**
- Exact penalty multipliers if adjustments needed
- Mathematical reasoning for the values
- Expected gradient between strategies
- How to verify the fix will work before training

---

## Additional Context

- Training on 20x10 Tetris board (standard size)
- Episodes end when board fills (max height reached)
- Average episode length: 30-40 steps
- No lines cleared in 500 episodes (agent dying too fast)
- Using complete 4-channel vision (verified working)
- Action mapping verified correct
