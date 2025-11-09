# Complete Training Analysis: hybrid_15k_fixed_curriculum (13,200 Episodes)

**Date**: 2025-11-07
**Training Status**: Stopped at 13,200 episodes (88% of 15K target)
**Model**: Hybrid Dual-Branch DQN (8-channel)
**Duration**: ~18 hours

---

## Executive Summary

🚨 **CRITICAL FINDING: Training has FAILED to meet objectives**

### Final Performance (Episodes 13,100-13,200)

| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| **Lines/episode** | **0.16** | 2-3 | ❌ **92-94% below target** |
| **Holes (avg)** | 23.7 | <15 | ⚠️ 58% above target |
| **Steps/episode** | 245.6 | 300-500 | ⚠️ Below target |
| **Stage** | line_clearing_focus | Stage 5 | ✅ Correct |
| **Epsilon** | 0.09 | <0.15 | ✅ Good |

### Key Findings

❌ **FAILURES:**
1. **Catastrophically low line clearing**: 0.16 lines/ep vs 2-3 target (92% shortfall)
2. **Reward system severely imbalanced**: Hole penalty -25K to -41K, Line bonus only +136
3. **No meaningful improvement in Stage 5**: Episodes 8K-13K showed minimal gains
4. **Agent plateaued**: Performance flat for last 5K episodes

✅ **SUCCESSES:**
1. Holes decreased from 36 to 23.7 (34% improvement)
2. Survival improved: steps 65→246 (4x improvement)
3. Epsilon decay working correctly
4. Stage 5 transition successful (fallback timer at episode 8001)

🔍 **ROOT CAUSE IDENTIFIED:**
**Reward shaping is fundamentally broken** - hole penalties dominate (+200x larger than line bonuses), agent never learns that lines are valuable.

---

## Detailed Performance Analysis

### Learning Curve (by 1000-episode blocks)

| Episodes | Avg Lines/Ep | Avg Holes | Avg Steps | Stage | Learning Rate |
|----------|--------------|-----------|-----------|-------|---------------|
| 1-1000 | 0.00 | 36.0 | 65.3 | foundation/clean | Baseline |
| 1001-2000 | 0.00 | 38.0 | 88.9 | spreading | No improvement |
| 2001-3000 | 0.01 | 40.4 | 104.6 | spreading | Minimal |
| 3001-4000 | 0.02 | 38.9 | 134.6 | clean_spreading | Very slow |
| 4001-5000 | 0.05 | 35.1 | 156.0 | clean_spreading | Slow |
| 5001-6000 | 0.07 | 33.0 | 168.7 | clean_spreading | Slow |
| 6001-7000 | 0.08 | 31.9 | 187.4 | clean_spreading | Slow |
| **7001-8000** | **0.10** | **30.9** | **205.7** | clean_spreading | Slow |
| **8001-9000** | **0.10** | **28.0** | **197.1** | **Stage 5 START** | **FLAT** |
| 9001-10000 | 0.10 | 25.2 | 181.9 | line_clearing | FLAT |
| 10001-11000 | 0.12 | 24.6 | 189.0 | line_clearing | Minimal |
| 11001-12000 | 0.14 | 24.0 | 206.3 | line_clearing | Minimal |
| 12001-13000 | 0.13 | 23.6 | 228.7 | line_clearing | REGRESSION |

**Critical Observation:**
Stage 5 transition at episode 8001 had **NO POSITIVE IMPACT** on line clearing rate. Agent continued at 0.10 lines/ep and only reached 0.14 by episode 12K before regressing.

### Comparison to Expectations

**From HYBRID_DQN_GUIDE.md:**

| Episode Range | Expected Lines/Ep | Actual | Delta |
|---------------|-------------------|--------|-------|
| 0-2000 | 0-0.5 | 0.00 | -0.25 (on track) |
| 2000-5000 | 0.5-1.0 | 0.02-0.05 | **-0.75** ⚠️ |
| 5000-10000 | 1.0-2.0 | 0.07-0.10 | **-1.20** ❌ |
| 10000-15000 | 2.0-3.0 | 0.12-0.13 | **-2.00** ❌ |

**Interpretation:**
The hybrid architecture was expected to learn 10-50x faster than visual-only baseline. Instead, it's learning at approximately **the same rate or slower** than expected baseline performance.

---

## Stage 5 Transition Analysis

### Transition Details
- **Episode 8001**: Stage 5 (line_clearing_focus) unlocked
- **Method**: Fallback timer (NOT performance gate)
- **Performance at transition**: 0.10 lines/ep, 30.9 holes

### Episodes Around Transition

| Episode | Lines | Holes | Steps | Stage |
|---------|-------|-------|-------|-------|
| 7998 | 0 | 30.7 | 246 | clean_spreading |
| 7999 | 0 | 25.3 | 206 | clean_spreading |
| **8000** | **0** | **24.8** | **224** | **clean_spreading** |
| **8001** | **0** | **25.0** | **278** | **line_clearing_focus** ← TRANSITION |
| 8002 | 1 | 26.4 | 277 | line_clearing_focus |
| 8003 | 0 | 36.3 | 246 | line_clearing_focus |

### Post-Transition Performance (Episode 8001-13200 = 5,200 episodes in Stage 5)

| Metric | Episode 8001 | Episode 13000 | Change | Expected |
|--------|--------------|---------------|--------|----------|
| Lines/ep (1K avg) | 0.10 | 0.13 | +0.03 | +1.5-2.0 |
| Holes | 28.0 | 23.6 | -4.4 | -10 to -15 |
| Steps | 197.1 | 228.7 | +31.6 | +50-100 |

**Conclusion**: Stage 5 reward shaping had **minimal positive impact**. Agent learned slightly better hole avoidance but did NOT learn line clearing.

---

## Reward System Analysis (CRITICAL ISSUE)

### Reward Component Breakdown (Last 5 Episodes)

#### Episode 13200 (1 line cleared) ✅ "GOOD"
```
Lines: 1 | Holes: 21.5 | Steps: 242
─────────────────────────────────────
Base reward:             +2,900
Line bonus:                +136  ← TINY!
Hole penalty:          -25,805  ← MASSIVE!
Completable bonus:      +1,980
Survival bonus:         -4,422  ← NEGATIVE?!
Other penalties:        ~-2,500
─────────────────────────────────────
TOTAL REWARD:           +2,216
```

#### Episode 13196 (0 lines cleared) ❌ "BAD"
```
Lines: 0 | Holes: 21.1 | Steps: 289
─────────────────────────────────────
Base reward:             +2,400
Line bonus:                   0  ← NO REWARD
Hole penalty:          -34,025  ← MASSIVE!
Completable bonus:      +3,735
Survival bonus:         -5,110  ← NEGATIVE?!
Other penalties:        ~-2,500
─────────────────────────────────────
TOTAL REWARD:           +2,164
```

### THE PROBLEM: Line Clearing Barely Matters

**Reward difference between clearing 1 line vs 0 lines:**
- Episode 13200 (1 line): +2,216
- Episode 13196 (0 lines): +2,164
- **Difference: +52 (+2.4%)**

**This is catastrophically insufficient!**

The agent receives:
- **-25,000 to -40,000** penalty for holes (even with only 21-23 holes)
- **+136** bonus for clearing 1 line
- **Net result**: Line clearing is ~200x less important than hole avoidance

### Why This Breaks Learning

The Q-learning algorithm learns based on reward signals:
```
Q(state, action) ← reward + γ * max Q(next_state)
```

If rewards are:
- Clear line: +52 total reward increase
- Avoid 1 hole: +X total reward increase (hole penalty = -5 per hole * steps)

The agent rationally learns: **"Holes matter 200x more than lines, focus on holes only!"**

**Expected reward structure** (for Stage 5):
- Clear 1 line: +500 to +1000 bonus
- Clear 2 lines: +1500 to +2500 bonus (with multiplier)
- Clear 4 lines (Tetris): +5000+ bonus
- Hole penalty: -50 to -200 total (not -25,000!)

---

## What Went Wrong: Root Cause Analysis

### 1. Reward Shaping Imbalance (PRIMARY ISSUE) ⭐

**Problem**: Hole penalty is cumulative across all steps and completely dominates reward signal.

**Evidence**:
- Hole penalty: -25K to -41K per episode
- Line bonus: +136 for 1 line
- Ratio: **200:1** (should be 1:5 to 1:10, favoring lines)

**Code location**: `src/progressive_reward_improved.py:366-367, 425-430`

```python
# Current (WRONG):
hole_penalty = -5.0 * holes  # Applied EVERY STEP → accumulates to -25K+
line_bonus = lines * 150.0 * quality  # One-time → only +136

# Should be:
hole_penalty = -1.0 * holes  # Reduce by 5x
line_bonus = lines * 1000.0 * quality  # Increase by 6-7x
```

### 2. Survival Bonus is Negative (WRONG)

**Problem**: Survival bonus showing negative values (-3,600 to -5,700)

**Evidence**: See reward breakdowns above

**This makes no sense** - survival should reward the agent for staying alive longer!

**Likely bug**: Check calculation in `src/progressive_reward_improved.py:414-423`

### 3. Curriculum May Be Too Slow

**Problem**: Agent spent 7,000 episodes in Stages 1-4 focusing on hole avoidance before learning lines.

**Evidence**:
- Episodes 1-7000: Learned hole avoidance (36→31 holes)
- Episodes 7001-8000: Continued hole focus
- Episodes 8001-13200: Stage 5, but reward still emphasizes holes

**Hypothesis**: By the time agent reached Stage 5, it had deeply learned "holes are everything", making it hard to unlearn.

### 4. Possible Neural Network Issues

**Hypothesis**: Without Q-value logging, we can't verify if:
- Q-values are stable or diverging
- Network is learning correct action values
- Network has enough capacity for complex Tetris strategy

**Recommendation**: Add Q-value logging (see below)

---

## Comparison to Baseline

### Hybrid 13K vs Visual-Only 10K (from previous runs)

| Metric | Hybrid 13K | Visual 10K | Hybrid Advantage |
|--------|------------|------------|------------------|
| Lines/ep | 0.13 | 0.21 | ❌ **38% WORSE** |
| Holes | 23.6 | 48 | ✅ 51% better |
| Steps | 228.7 | ~200 | ✅ 14% better |

**Shocking conclusion**: The hybrid architecture is performing **WORSE** than visual-only for line clearing, despite having explicit feature channels for holes, heights, etc.

**Why?**
1. Reward shaping is broken (favors holes 200:1 over lines)
2. Visual-only may have better exploration (less biased by explicit features)
3. Hybrid may be overfitting to hole avoidance features

---

## Logging System Analysis

### Current Logging: Comprehensive ✅

**What's already logged (46 columns):**
- Episode: episode, timestamp, stage
- Performance: lines_cleared, total_lines, steps, epsilon
- Board quality: holes (avg/min/final), holes_at_step_X, columns_used
- Metrics: completable_rows, clean_rows, bumpiness, max_height (avg + final)
- Curriculum: gate metrics (holes/completable/clean thresholds)
- Rewards: reward, original_reward, shaped_reward_used
- **All reward components**: rc_base, rc_line_bonus, rc_hole_penalty, rc_completable_bonus, rc_survival_bonus, rc_center_penalty, rc_spread_bonus, rc_height_penalty, rc_structure_penalty, rc_outer_penalty, rc_height_std_penalty, rc_hole_reduction_bonus, rc_clean_rows_bonus, rc_efficiency_bonus, rc_death_penalty, rc_clip_delta, rc_pre_clip_reward

**This is EXCELLENT logging!** You can diagnose almost everything from the CSV.

### Critical Missing: Q-Values ⭐

**What we can't see:**
- What Q-values is the network predicting?
- Are Q-values stable or diverging?
- Does Q(HARD_DROP) increase when clearing lines?
- Are Q-values converging to correct action values?

**Example: If Q-values show:**
```
Episode 13200:
  Q(LEFT) = 250
  Q(RIGHT) = 240
  Q(HARD_DROP) = 220
  Q(SWAP) = 210
```

This tells us:
- Q-values in reasonable range (100-500) ✅
- Agent slightly prefers LEFT/RIGHT (horizontal positioning)
- HARD_DROP Q-value relatively low (agent not confident in fast placement)

**If Q-values show:**
```
Episode 13200:
  Q(LEFT) = 2500
  Q(RIGHT) = 2480
  Q(HARD_DROP) = 50
  Q(SWAP) = 10
```

This tells us:
- Q-values too high (overestimation) ❌
- HARD_DROP severely undervalued (won't clear lines) ❌
- Network may be unstable ❌

---

## Recommendations

### Priority 1: FIX REWARD SHAPING ⭐⭐⭐ CRITICAL

**Changes needed in `src/progressive_reward_improved.py`:**

#### A. Reduce Hole Penalty (5x reduction)
```python
# Line 366-367 (Stage 5)
# OLD:
hole_penalty = -5.0 * holes

# NEW:
hole_penalty = -1.0 * holes  # Reduce from -5 to -1
```

#### B. Increase Line Bonus (6-8x increase)
```python
# Line 425-440 (Stage 5)
# OLD:
if lines > 0:
    quality = max(0.3, 1.0 - (metrics['holes'] / 50.0) - (metrics['bumpiness'] / 100.0))
    line_bonus = lines * 150.0 * quality
    if lines == 2:
        line_bonus += 50.0 * quality
    elif lines == 3:
        line_bonus += 150.0 * quality
    elif lines == 4:
        line_bonus += 400.0 * quality

# NEW:
if lines > 0:
    quality = max(0.5, 1.0 - (metrics['holes'] / 80.0))  # More lenient quality
    line_bonus = lines * 1000.0 * quality  # 150 → 1000 (6.7x increase)
    if lines == 2:
        line_bonus += 500.0 * quality  # 50 → 500 (10x)
    elif lines == 3:
        line_bonus += 2000.0 * quality  # 150 → 2000 (13x)
    elif lines == 4:
        line_bonus += 5000.0 * quality  # 400 → 5000 (12x)
```

#### C. Fix Survival Bonus (should be positive!)
```python
# Line 414-423 (Stage 5)
# Check current implementation - survival_bonus should NEVER be negative
# It should reward agent for surviving longer

# Expected logic:
steps = info.get('steps', 0)
if holes < 8:
    survival_bonus = min(steps * 0.5, 50.0)  # Up to +50
elif holes < 15:
    survival_bonus = min(steps * 0.3, 30.0)  # Up to +30
elif holes < 25:
    survival_bonus = min(steps * 0.1, 15.0)  # Up to +15
else:
    survival_bonus = 0.0  # No bonus if too many holes
```

#### D. Expected Reward Structure After Fix

**Good episode (1 line, 20 holes, 250 steps):**
```
Base: +2,000
Line bonus: +1,000 (lines * 1000 * quality)
Hole penalty: -20 (20 * -1)
Survival: +30
Completable: +500
Other: +200
─────────────
TOTAL: +3,710 ✅
```

**Bad episode (0 lines, 30 holes, 150 steps):**
```
Base: +1,500
Line bonus: 0
Hole penalty: -30
Survival: +15
Completable: +200
Other: -100
─────────────
TOTAL: +1,585 ❌
```

**Difference: +2,125 (133% increase)** ← This will teach the agent!

### Priority 2: ADD Q-VALUE LOGGING ⭐⭐

**Implementation** (add to `train_progressive_improved.py`):

```python
# After action selection (around line 700-750), BEFORE env.step():

# Collect Q-values for this step
with torch.no_grad():
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    agent.q_network.eval()
    q_values = agent.q_network(state_tensor).cpu().numpy()[0]
    episode_q_values.append(q_values)

# At episode end (in logger.log_episode(), around line 850-880):
if len(episode_q_values) > 0:
    q_values_avg = np.mean(episode_q_values, axis=0)
    logger.log_episode(
        ...existing metrics...,

        # Add Q-value metrics:
        q_mean=np.mean(q_values_avg),
        q_std=np.std(q_values_avg),
        q_max=np.max(q_values_avg),
        q_min=np.min(q_values_avg),
        q_left=q_values_avg[0],
        q_right=q_values_avg[1],
        q_down=q_values_avg[2],
        q_rotate_cw=q_values_avg[3],
        q_rotate_ccw=q_values_avg[4],
        q_hard_drop=q_values_avg[5],
        q_swap=q_values_avg[6],
        q_noop=q_values_avg[7],
    )
```

**What to monitor:**
- `q_mean`: Should be 100-500 range by episode 10K
- `q_max - q_min`: Action value spread, should be 50-200
- `q_hard_drop`: Should increase as agent learns line clearing
- If Q-values > 1000: Overestimation problem
- If Q-values < 10: Network not learning

### Priority 3: RETRAIN WITH FIXED REWARDS ⭐⭐

**Recommended approach:**

**Option A: Fresh start with fixed rewards (RECOMMENDED)**
```bash
python train_progressive_improved.py \
    --episodes 20000 \
    --force_fresh \
    --model_type hybrid_dqn \
    --experiment_name hybrid_20k_fixed_rewards
```

**Why 20K?**
- 13K wasn't enough with broken rewards
- Fixed rewards should learn faster
- Need buffer for learning curve

**Option B: Fine-tune from checkpoint**
- Risky: Agent already learned "holes >> lines"
- May be stuck in local optimum
- Would need to "unlearn" bad behavior

### Priority 4: Consider Curriculum Adjustments ⭐

**Option 1: Earlier Stage 5 (Aggressive)**
```python
# In get_current_stage():
# OLD: Stage 5 at episode 5000+
# NEW: Stage 5 at episode 3000+

if self.episode_count < 500:
    return "foundation"
elif self.episode_count < 1000:
    return "clean_placement"
elif self.episode_count < 2000:
    return "spreading_foundation"
elif self.episode_count < 3000:  # Was 5000
    return "clean_spreading"
else:
    # Stage 5 logic...
```

**Rationale**: Don't let agent over-optimize on hole avoidance before learning lines.

**Option 2: Introduce Lines Earlier (Recommended)**
- Add small line bonus (+100-200) in Stages 2-4
- Gradually increase in Stage 5
- Agent learns "lines are good" throughout training

```python
# In all stage reward functions, add:
if metrics['lines_cleared'] > 0:
    early_line_bonus = metrics['lines_cleared'] * 100.0  # Small reward
    shaped += early_line_bonus
```

---

## Decision Matrix: What To Do Next

### Scenario A: Fix and Retrain (RECOMMENDED ✅)

**Steps:**
1. Fix reward shaping (Priority 1 above)
2. Add Q-value logging (Priority 2 above)
3. Train fresh 20K episodes
4. Monitor Q-values and reward balance
5. Expected result: 1.5-3.0 lines/ep by episode 15-20K

**Time investment**: ~25-30 hours training
**Success probability**: 75-85%

### Scenario B: Analyze Current Model Further

**Steps:**
1. Add Q-value logging to current checkpoint
2. Run 100 evaluation episodes with Q-value tracking
3. Analyze what agent has learned
4. Determine if fixable with fine-tuning

**Time investment**: ~2-3 hours analysis
**Success probability**: 30-40% (agent may be too committed to hole avoidance)

### Scenario C: Try Different Architecture

**Options:**
- Standard DQN (no dual-branch) with 8 channels
- Dueling DQN architecture
- Increase network capacity (more layers/neurons)

**Time investment**: ~5-10 hours implementation + 20-30 hours training
**Success probability**: 60-70%

### Scenario D: Reduce to Simpler Problem

**Try 4-channel visual-only with FIXED rewards:**
- Simpler observation space
- Proven architecture
- Fixed reward shaping

**Time investment**: ~20 hours training
**Success probability**: 65-75%

---

## Recommended Action Plan

### Phase 1: Immediate Fixes (2-3 hours)

1. **Fix reward shaping** in `src/progressive_reward_improved.py`:
   - [ ] Reduce hole penalty: -5.0 → -1.0
   - [ ] Increase line bonus: *150 → *1000
   - [ ] Increase multi-line bonuses: 10x
   - [ ] Fix survival bonus (verify it's positive)
   - [ ] Test with `pytest tests/test_reward_helpers.py`

2. **Add Q-value logging** in `train_progressive_improved.py`:
   - [ ] Collect Q-values during episode
   - [ ] Log average Q-values per action
   - [ ] Add to CSV output

3. **Test reward changes**:
   ```bash
   python train_progressive_improved.py \
       --episodes 100 \
       --model_type hybrid_dqn \
       --experiment_name quick_reward_test
   ```
   - Verify line clearing gets +1000-5000 reward
   - Verify holes get -20 to -50 penalty
   - Check last 10 episodes in CSV

### Phase 2: Full Retraining (25-30 hours)

1. **Start fresh 20K training**:
   ```bash
   python train_progressive_improved.py \
       --episodes 20000 \
       --force_fresh \
       --model_type hybrid_dqn \
       --experiment_name hybrid_20k_fixed_rewards
   ```

2. **Monitor training** (check every 5K episodes):
   - Lines/ep should increase: 0 → 0.5 → 1.5 → 2.5 → 3.5
   - Q-values should stay 100-500 range
   - Hole penalty should be -50 to -200 total (not -25K!)
   - Line bonus should dominate when lines cleared

3. **Milestones to check**:
   - Episode 5K: 0.5-1.0 lines/ep ✅
   - Episode 10K: 1.5-2.5 lines/ep ✅
   - Episode 15K: 2.5-3.5 lines/ep ✅
   - Episode 20K: 3.0-4.0 lines/ep ✅ (target exceeded!)

### Phase 3: Evaluation (2-3 hours)

1. **Evaluate best checkpoint**:
   ```bash
   python evaluate.py \
       --model_path models/hybrid_20k_best.pth \
       --episodes 100 \
       --render
   ```

2. **Analyze results**:
   - Avg lines/episode
   - Max lines in single episode
   - Board quality (holes, height)
   - Q-value stability

3. **Compare to baseline**:
   - vs current 13K training (0.13 lines/ep)
   - vs visual-only 10K (0.21 lines/ep)
   - vs expected (2-3 lines/ep)

---

## Conclusion

### What We Learned

1. **Reward shaping is CRITICAL**: Even with perfect architecture, bad rewards = bad learning
2. **Logging saved us**: Comprehensive reward component logging revealed the 200:1 imbalance
3. **Curriculum alone insufficient**: Can't rely on curriculum to fix reward imbalance
4. **Hybrid architecture works for hole avoidance**: Agent learned holes (36→23.7) but not lines
5. **Q-values are essential**: Can't debug decision-making without them

### Current Training Status: ❌ FAILED

- **13,200 episodes completed**
- **0.13 lines/episode** (92% below target)
- **Reward system broken** (holes penalized 200x more than lines rewarded)
- **Not recommended to continue** without fixes

### Next Steps: ✅ FIX AND RETRAIN

1. Fix reward shaping (2-3 hours)
2. Add Q-value logging (1 hour)
3. Test fixes with 100 episodes (1 hour)
4. Retrain 20K episodes (25-30 hours)
5. **Expected outcome**: 2-4 lines/episode by episode 15-20K

### Estimated Timeline

- Fixes + testing: 4-5 hours
- Retraining: 25-30 hours
- Evaluation: 2-3 hours
- **Total: ~32-38 hours to success**

---

## Appendix: Comparison Tables

### Performance vs Expectations

| Metric | Expected (15K) | Actual (13K) | Gap |
|--------|----------------|--------------|-----|
| Lines/ep | 2.5 | 0.13 | **-2.37 (95%)** |
| Holes | <15 | 23.7 | **+8.7 (58%)** |
| Steps | 350 | 228.7 | **-121.3 (35%)** |

### Reward Component Analysis

| Component | Expected Range | Actual Range | Status |
|-----------|----------------|--------------|--------|
| Base | +500 to +2000 | +1900 to +2900 | ✅ OK |
| Line bonus | +500 to +5000 | +0 to +136 | ❌ 30x too small |
| Hole penalty | -50 to -200 | -25K to -41K | ❌ 200x too large |
| Survival | +10 to +50 | -3600 to -5700 | ❌ WRONG SIGN |
| Completable | +200 to +1000 | +1980 to +14400 | ⚠️ Too variable |

### Training Efficiency

| Architecture | Episodes to 1.0 lines/ep | Episodes to 2.0 lines/ep | Expected |
|--------------|--------------------------|--------------------------|----------|
| Visual-only (baseline) | ~15,000 | ~50,000 | Slow |
| Hybrid (current) | >13,000 (didn't reach) | >50,000 (projected) | SAME AS BASELINE ❌ |
| Hybrid (fixed rewards) | ~5,000 (projected) | ~12,000 (projected) | 4x faster ✅ |

---

**End of Analysis**

**Status**: Training failed due to reward imbalance
**Recommendation**: Fix rewards and retrain
**Confidence**: High (root cause identified via comprehensive logging)
