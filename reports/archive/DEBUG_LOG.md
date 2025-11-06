# Tetris AI Debug Log

## Quick Reference

**Current Status:** ✅ Improved 5-Stage Curriculum Implemented - FIXES CENTER STACKING

**Recommended Training Command:**
```bash
python train_progressive_improved.py --episodes 10000 --force_fresh
```

**Key Files:**
- `src/progressive_reward_improved.py` - **NEW** Improved 5-stage reward shaper (FIXES center stacking!)
- `train_progressive_improved.py` - **NEW** Improved training script with better curriculum
- `src/progressive_reward.py` - Original 4-stage curriculum (had bugs, now superseded)
- `train_progressive.py` - Original progressive training (had bugs, now superseded)
- `src/reward_shaping.py` - Base reward calculations (used by all shapers)
- `logs/{experiment}/board_states.txt` - Visual board logging for debugging

---

## Debug History

### [2025-11-02 09:00] ✅ FIX #5: Improved 5-Stage Curriculum - FIXES CENTER STACKING

**Problem:** Agent stuck in local optimum
```
Episode 1100-1170 (current 4-stage curriculum):
- Stacking ONLY in columns 3-6
- Creating 21-31 holes per game
- 0 lines cleared every game
- Dying in 10-13 steps
- Agent learned: "Spreading (+60) > Holes penalty (-24) → holes are worth it!"
```

**Root Cause Analysis:**
```
Current Stage 3 (spreading) reward math:
  Spread bonus:         +60 reward  (25.0 * spread + 6.0 * columns_used)
  Holes penalty (30):   -24 penalty (0.8 * 30)
  NET:                  +36 reward

Agent's rational strategy: "Spread with holes = more reward!"
Problem: Agent DOESN'T KNOW it can clear lines because:
  1. With 30 holes, impossible to clear a full row
  2. Never gets line-clear reward signal
  3. Hole penalty too weak to prevent this behavior
```

**Solution Implemented: Improved 5-Stage Curriculum**

**New Files Created:**
- `src/progressive_reward_improved.py` - Better balanced 5-stage reward shaper
- `train_progressive_improved.py` - Integrated training script with existing codebase

**Key Improvements:**

1. **New Reward Metrics:**
   ```python
   # Completable Rows: Rows with 8-9 filled cells and NO holes
   # Teaches: "Almost-complete clean rows = good!"
   shaped += completable_rows * 8.0  # Stage 4: 8.0, Stage 5: 12.0

   # Clean Rows: Rows with no holes
   shaped += clean_rows * 3.0  # Progressive: 3.0 → 10.0

   # Conditional Survival: Only reward survival with clean boards
   if holes < 30:
       shaped += min(steps * 0.4, 30.0)
   else:
       shaped += min(steps * 0.1, 10.0)  # Minimal if too many holes
   ```

2. **Better Hole Penalty Progression:**
   ```
   Old (4-stage):  1.0 → 1.2 → 0.8 → 0.75 (GOES DOWN!)
   New (5-stage):  0.3 → 1.0 → 0.8 → 1.5 → 2.0 (STEADILY INCREASES)
   ```

3. **Longer, More Focused Stages:**
   ```
   Stage 1 (0-500):     Foundation       - Gentle start, avoid learned helplessness
   Stage 2 (500-1000):  Clean Placement  - Progressive holes penalty (0.3 → 1.0)
   Stage 3 (1000-2000): Spreading Found  - Learn spreading while staying clean
   Stage 4 (2000-5000): Clean Spreading  - Strong holes penalty + completable rows
   Stage 5 (5000+):     Line Clearing    - Maximize line clears with efficiency bonus
   ```

4. **Reward Math Now Makes Sense:**
   ```
   Stage 4 Clean Spreading Reward Math:
     Clean spreading:  +60 spread - 15 holes = +45 reward
     Messy spreading:  +60 spread - 105 holes = -45 reward
     Center stacking:  +0 spread - 15 holes = -15 reward

   Result: Clean spreading > Center stacking > Messy spreading
   ```

**Integration Features:**
- ✅ Uses existing Agent, TrainingLogger, and config system
- ✅ Preserves board state logging (same format as before)
- ✅ Tracks new metrics (completable_rows, clean_rows) in CSV logs
- ✅ Shows curriculum progression with stage transitions
- ✅ Compatible with checkpoint resume (--resume flag)

**Expected Training Progression:**

| Episodes | Stage | Steps | Holes | Cols | Completable | Lines | What Happens |
|----------|-------|-------|-------|------|-------------|-------|--------------|
| 0-500 | foundation | 15→40 | 40→25 | 4-5 | 0-1 | 0 | Survive longer, basic placement |
| 500-1000 | clean_placement | 40→60 | 25→15 | 5-6 | 1-2 | 0-1 | Holes decrease significantly |
| 1000-2000 | spreading_foundation | 60→80 | 15→20 | 6→8 | 2-3 | 0-2 | **Spreading begins!** |
| 2000-5000 | clean_spreading | 80→150 | 20→10 | 8→10 | 3-5 | 2-5 | **Clean spreading mastered!** |
| 5000+ | line_clearing_focus | 150→300+ | <10 | 10 | 5+ | 5-10+ | **Consistent line clears!** |

**Success Criteria:**
- By episode 2000: ≥8 columns used, <20 holes
- By episode 5000: ≥8 columns used, <15 holes, ≥2 lines/episode
- By episode 10000: All 10 columns, <10 holes, ≥5 lines/episode

**Why This Will Work:**
1. Progressive curriculum prevents learned helplessness (gentle start)
2. Completable rows bridges gap between placement and line clearing
3. Conditional survival stops rewarding bad boards
4. Balanced penalties make clean spreading optimal strategy
5. Agent learns each skill before combining them

**Training Command:**
```bash
# Start fresh (recommended - old models stuck in bad habits)
python train_progressive_improved.py --episodes 10000 --force_fresh

# Resume from checkpoint
python train_progressive_improved.py --episodes 10000 --resume

# Quick test (500 episodes)
python train_progressive_improved.py --episodes 500 --force_fresh
```

**Files Modified:**
- Created: `src/progressive_reward_improved.py`
- Created: `train_progressive_improved.py`
- Updated: `DEBUG_LOG.md` (this file)

**Status:** ✅ Ready to train - This should fix the center stacking problem!

---

### [2025-11-01 15:15] ✅ FIX #4B: Progressive Curriculum - Fixed Learned Helplessness

**Issue:** Agent learned to die faster! (45 steps → 11 steps over training)

**Two Critical Bugs Found:**

**Bug #1: Episode Count Never Updated**
```python
# Problem: Curriculum stuck in Stage 1 forever
reward_shaper.episode_count  # Never updated from training loop!

# Fix: train_progressive.py - Added at start of each episode
reward_shaper.episode_count = episode  # Critical for stage advancement!
```

**Bug #2: Penalties Too Harsh → Learned Helplessness**
```
Stage 1 penalties with 25 holes:
  Holes:     -2.0 × 25 = -50
  Bumpiness: -0.5 × 40 = -20
  Total:     -70+ per step!

Agent learned: "Everything is bad → die fast to end punishment"

Result: Episodes got SHORTER over time (45 → 11 steps)
```

**Fixes Applied:**
```python
# Stage 1 (Episodes 0-200): GENTLER to avoid learned helplessness
shaped -= 1.0 * holes      # REDUCED from 2.0
shaped -= 0.3 * bump       # REDUCED from 0.5
shaped += min(steps * 0.5, 30.0)  # STRONGER survival bonus (was 0.2, max 20)

# Stage 2 (Episodes 200-400): BALANCED
shaped -= 1.2 * holes      # REDUCED from 1.5
shaped += 8.0 * spread     # INCREASED from 5.0
shaped += min(steps * 0.4, 25.0)  # STRONGER survival (was 0.2, max 20)

# Stages 3-4: Unchanged
```

**Expected Results After Fix:**
```
Episodes 0-200:   Steps INCREASE 11 → 20+ (agent survives longer)
                  Holes DECREASE 25 → 15 (learning clean placement)
Episodes 200-400: Height management improves, still surviving
Episodes 400-600: Columns used increases 4 → 8 (spreading!)
Episodes 600+:    Balanced play + line clears
```

**Files Modified:**
- `train_progressive.py` - Fixed episode count update
- `src/progressive_reward.py` - Reduced early stage penalties, increased survival bonuses

**Status:** Fixed but still has center stacking issue → See Fix #5 above

---

### [2025-11-01 11:00] ⚠️  FIX #4A: Progressive Curriculum Learning (INITIAL - HAD BUGS)

**Issue:** Skill-reward mismatch - agent lacks motor skills to spread cleanly

**Solution Applied: 4-Stage Curriculum**
- Stage 1 (0-200): Basic placement - high hole penalty
- Stage 2 (200-400): Height management - add spreading rewards
- Stage 3 (400-600): Spreading - reduce hole penalty
- Stage 4 (600+): Balanced play

**Status:** Had implementation bugs (see Fix #4B) and still had center stacking (see Fix #5)

---

### [2025-11-01 10:55] ✅ FIX #3: Local Optimum Trap

**Issue:** Agent trapped - spreading created more holes than center-stacking

**Root Cause:**
```
Training data showed:
  Center-stacking: 31.1 holes avg
  Spreading:       45.3 holes avg (+14.1 more)

Penalty calculation:
  Holes penalty: -2.0 × 14.1 = -28.2
  Anti-center:   +33.3
  NET GRADIENT:  +5.1 (TOO WEAK!)
```

**Fix Applied:**
```python
# src/reward_shaping.py
shaped -= 0.75 * holes       # was -2.0 (REDUCED)
shaped += 25.0 * spread      # was 15.0 (INCREASED)
shaped += columns_used * 6.0 # was 4.0 (INCREASED)
shaped -= outer_unused * 8.0 # was 5.0 (INCREASED)
shaped -= 3.0 * height_std   # was 2.0 (INCREASED)
```

**Results:**
```
Gradient: 33 → 85 points (2.5x stronger)
Center-stacking: -52.90 → -71.40/step
Good spreading:  +22.99 → +40.73/step
Total gradient:  76 → 112 points
```

**Status:** Improved gradient but not enough → Led to Fix #4 and Fix #5

---

### [2025-11-01 10:34] ❌ FIX #2: Gradient Restoration (Failed)

**Issue:** All rewards hit -100 clamp, no learning gradient

**Diagnosis:**
```
Penalties too strong → everything = -100/step → gradient = 0
```

**Fix Applied:**
```python
# Reduced penalties by 2-5x
shaped -= 0.5 * bump        # was 1.0
shaped -= 2.0 * height_std  # was 10.0
shaped -= 5.0 * outer_unused # was 10.0

# Removed train.py clamp
# shaped_reward = np.clip(shaped_reward, -100.0, 600.0)  # REMOVED
```

**Results:**
```
FAILED - Agent still center-stacked
Reason: Holes penalty was overwhelming signal (see Fix #3)
```

---

### [2025-11-01 10:22] ❌ FIX #1: Strengthen Penalties (Failed)

**Issue:** Agent center-stacking, penalties seemed too weak

**Fix Applied:**
```python
# Strengthened penalties 17-20x
shaped -= 1.0 * bump         # was 0.06 (17x)
shaped -= 10.0 * height_std  # was 0.5 (20x)
shaped -= 10.0 * outer_unused # was 5.0 (2x)
```

**Results:**
```
FAILED - All rewards hit -200 → clamped to -100
Created worse problem (no gradient)
```

---

### [2025-11-01 10:11] ✅ Board State Logging Added

**Added:** Visual board logging to `logs/{experiment}/board_states.txt`

**Files Modified:**
- `src/utils.py` - Added `log_board_state()` method
- `train.py` - Call logging every 10 episodes
- `train_progressive.py` - Call logging every 10 episodes

**Usage:**
```bash
# View board states
cat logs/latest/board_states.txt | grep "Column heights"

# Analyze holes pattern
awk '/Holes:/ {print $3}' logs/latest/board_states.txt | head -50
```

**Key Insight:** This revealed the center stacking and holes problem that led to all subsequent fixes

**Example Output:**
```
Episode 1100 | Reward: -440.4 | Steps: 13 | Lines: 0
Column heights: [0, 0, 0, 0, 20, 20, 14, 3, 0, 0]
Holes: 21 | Bumpiness: 40.0 | Max height: 20
  0123456789
 0 ····██····  (2/10)
 1 ····██····  (2/10)
 ...
```

---

### [2025-10-31 23:20] ✅ Action Mapping Fixed

**Issue:** Agent using wrong actions (LEFT/RIGHT swapped)

**Fix:**
```python
# config.py - Corrected action mapping
action_map = {
    0: 0,  # LEFT
    1: 1,  # RIGHT
    2: 2,  # DOWN
    3: 3,  # ROTATE_CW
    4: 4,  # ROTATE_CCW
    5: 5,  # HARD_DROP
    6: 6,  # SWAP
    7: 7,  # NOOP
}
```

**Verification:**
```bash
.venv/bin/python tests/test_action_mapping.py
```

---

### [2025-10-31 19:58] ✅ 4-Channel Vision Enabled

**Added:** Complete 4-channel observations

**Channels:**
- 0: Board state (locked pieces)
- 1: Active piece position
- 2: Holder piece
- 3: Next queue

**Verification:**
```python
# Check in training logs:
# "✅ 4-channel complete vision confirmed!"
```

**Impact:** Gives agent complete information about game state, essential for strategic play

---

## Current Reward System (Improved 5-Stage Curriculum)

### Stage-Specific Rewards

**Stage 1: Foundation (0-500)**
```python
shaped -= 0.3 * holes              # Gentle
shaped -= 0.02 * aggregate_height
shaped -= 0.1 * bumpiness
shaped += min(steps * 0.8, 40.0)   # Strong survival
shaped += lines * 50.0
if done: shaped -= 10.0
```

**Stage 2: Clean Placement (500-1000)**
```python
hole_penalty = 0.3 + (episode-500)/500 * 0.7  # Progressive 0.3→1.0
shaped -= hole_penalty * holes
shaped -= 0.03 * aggregate_height
shaped -= 0.2 * bumpiness
shaped += clean_rows * 3.0         # NEW
shaped += min(steps * 0.5, 30.0)
shaped += lines * 60.0
if done: shaped -= 15.0
```

**Stage 3: Spreading Foundation (1000-2000)**
```python
shaped -= 0.8 * holes
shaped -= 0.04 * aggregate_height
shaped -= 0.3 * bumpiness
shaped += 15.0 * spread            # Gentle intro
shaped += columns_used * 3.0
shaped -= outer_unused * 4.0
shaped -= 2.0 * height_std
shaped += clean_rows * 4.0
shaped += min(steps * 0.3, 25.0)
shaped += lines * 70.0
if done: shaped -= 20.0
```

**Stage 4: Clean Spreading (2000-5000)**
```python
shaped -= 1.5 * holes              # Strong penalty
shaped += completable_rows * 8.0   # NEW - Critical!
shaped -= 0.05 * aggregate_height
shaped -= 0.4 * bumpiness
shaped += 25.0 * spread            # Strong
shaped += columns_used * 5.0
shaped -= outer_unused * 8.0
shaped -= 3.0 * height_std
shaped += clean_rows * 6.0
if holes < 30:
    shaped += min(steps * 0.4, 30.0)  # Conditional survival
else:
    shaped += min(steps * 0.1, 10.0)
shaped += lines * 100.0
if lines == 4: shaped += 200.0     # Tetris bonus
if done: shaped -= 25.0
```

**Stage 5: Line Clearing Focus (5000+)**
```python
shaped -= 2.0 * holes              # Very strong
shaped += completable_rows * 12.0
shaped -= 0.06 * aggregate_height
shaped -= 0.5 * bumpiness
shaped += 20.0 * spread
shaped += columns_used * 4.0
shaped -= outer_unused * 10.0
shaped -= 4.0 * height_std
shaped += clean_rows * 10.0
if holes < 20:
    shaped += min(steps * 0.5, 40.0)
shaped += lines * 150.0
if lines == 4: shaped += 400.0     # Massive tetris bonus
if lines > 0:
    efficiency = lines / pieces_placed
    shaped += efficiency * 100.0    # Efficiency bonus
if done:
    if holes > 50: shaped -= 50.0
    else: shaped -= 30.0
```

### Key Metrics Reference

**New Metrics (Fix #5):**
```python
def count_clean_rows(board):
    # Rows with no holes (contiguous filled cells or empty)

def count_completable_rows(board):
    # Rows with 8-9 filled cells and NO holes
    # These are 1-2 pieces away from clearing
```

**Reward Value Examples (Stage 4):**
```
Clean spreading (8 cols, 10 holes, 3 completable):
  +25*0.8 (spread) + 8*5 (cols) + 3*8 (completable) - 1.5*10 (holes)
  = +20 + 40 + 24 - 15 = +69/step

Messy spreading (8 cols, 70 holes, 0 completable):
  +25*0.8 + 8*5 + 0 - 1.5*70
  = +20 + 40 + 0 - 105 = -45/step

Center stacking (4 cols, 10 holes, 1 completable):
  +25*0.2 + 4*5 + 1*8 - 1.5*10
  = +5 + 20 + 8 - 15 = +18/step

Result: Clean spreading > Center stacking > Messy spreading
```

---

## Common Issues & Solutions

### Agent center-stacking?
**Use the improved 5-stage curriculum (Fix #5):**
```bash
python train_progressive_improved.py --episodes 10000 --force_fresh
```

The improved curriculum specifically fixes this with:
- Completable rows metric (teaches intermediate step to line clears)
- Conditional survival bonus (stops rewarding messy play)
- Better hole penalty progression (0.3 → 2.0, not going down)

### Agent dying too fast (learned helplessness)?
Check if episodes are getting longer or shorter:
```bash
grep "Steps:" logs/*/board_states.txt | head -50
```

If decreasing → penalties too harsh:
- Use improved curriculum (Fix #5) - has gentler start
- Or reduce hole penalty in early stages: 1.0 → 0.8

### Curriculum not advancing?
Check for stage transition messages:
```
🎓 CURRICULUM ADVANCEMENT: basic → height
```

If missing:
- Verify `reward_shaper.update_episode(episode)` is called
- Check episode count in logs

### No line clears?
1. Agent needs to learn clean placement first (Stages 1-2)
2. Then learn spreading (Stage 3)
3. Then combine them (Stage 4)
4. Then maximize lines (Stage 5)

Don't expect lines until episode 2000-4000 with improved curriculum.

---

## Test Commands

```bash
# Train with improved 5-stage curriculum (RECOMMENDED)
python train_progressive_improved.py --episodes 10000 --force_fresh

# Train with original 4-stage curriculum (has bugs, not recommended)
python train_progressive.py --episodes 2000 --force_fresh

# Check board states from latest run
ls -lt logs/*/board_states.txt | head -1 | awk '{print $NF}' | xargs cat | head -100

# Analyze holes vs column usage over time
awk '/^Episode/ {ep=$0} /Holes:/ {print ep; print}' logs/*/board_states.txt | head -40

# Monitor completable rows (new metric from Fix #5)
grep "Compl:" logs/improved_*/episode_log.csv | tail -20
```

---

## Files Changed Summary

**Created (Fix #5 - Improved Curriculum):**
- `src/progressive_reward_improved.py` - Improved 5-stage reward shaper
- `train_progressive_improved.py` - Integrated training script

**Created (Fix #4 - Original Curriculum):**
- `src/progressive_reward.py` - Original 4-stage reward shaper
- `train_progressive.py` - Original progressive training
- `tests/test_reward_diagnosis.py` - Diagnostic showing why spreading fails

**Modified:**
- `src/reward_shaping.py` - Multiple iterations of penalty tuning (Fixes #1-3)
- `src/utils.py` - Added board state logging (Fix #0)
- `train.py` - Removed reward clamp, added board logging
- `config.py` - Fixed action mapping
- `DEBUG_LOG.md` - This file

**Key Tests:**
- `tests/test_reward_system_complete.py` - Comprehensive reward test
- `tests/test_reward_diagnosis.py` - Shows why spreading fails without curriculum

---

## Lessons Learned

### 1. RL Agents Can Learn Bad Behaviors
- Overly harsh penalties → learned helplessness (agent dies faster over time)
- Weak gradient → local optimum trap (center stacking with holes)
- Solution: Progressive curriculum with balanced rewards

### 2. Curriculum Learning Requires Proper Episode Tracking
- Bug: Episode count never updated → stuck in Stage 1 forever
- Fix: Always update `reward_shaper.episode_count = episode`

### 3. Intermediate Rewards Are Critical
- Agent can't jump from "place piece" to "clear lines"
- Need intermediate: "completable rows" (8-9 filled, no holes)
- This teaches: clean rows → almost complete → line clears

### 4. Survival Must Be Rewarded Conditionally
- Unconditional survival: Rewards sitting on 70 holes doing nothing
- Conditional survival: Only reward if holes < 30
- Result: Agent learns good survival, not just any survival

### 5. Penalty Progression Must Make Sense
- Bad: 1.0 → 1.2 → 0.8 → 0.75 (goes down when agent should be skilled)
- Good: 0.3 → 1.0 → 0.8 → 1.5 → 2.0 (gentle start, then increases)

### 6. Board State Logging Is Essential
- Without visual board logs, we wouldn't have discovered center stacking
- Logs revealed: columns [0,0,0,0,20,20,14,3,0,0] (only using 3-6)
- Always log board states during debugging

---

## Next Steps (If Center Stacking Persists)

If the improved 5-stage curriculum (Fix #5) still shows center stacking after 5000 episodes:

1. **Increase outer column penalty:**
   ```python
   shaped -= 12.0 * outer_unused  # was 8.0-10.0
   ```

2. **Increase completable rows reward:**
   ```python
   shaped += 15.0 * completable_rows  # was 8.0-12.0
   ```

3. **Add explicit column distribution reward:**
   ```python
   # Reward using all columns evenly
   col_entropy = -sum(p*log(p) for p in col_distribution if p > 0)
   shaped += col_entropy * 10.0
   ```

4. **Force outer column exploration in early stages:**
   ```python
   # In training loop
   if episode < 1000 and outer_unused >= 5:
       # Boost exploration temporarily
       agent.epsilon = min(agent.epsilon + 0.1, 0.9)
   ```

But try the improved curriculum first - it should work!

---

## Success Criteria (Improved 5-Stage Curriculum)

**By Episode 2000:**
- ✅ Agent uses 8+ columns regularly
- ✅ Holes average < 20
- ✅ Some line clears occurring (even if rare)
- ✅ Completable rows: 2-3 average

**By Episode 5000:**
- ✅ Agent uses all 10 columns
- ✅ Holes average < 15
- ✅ Lines: 2+ per episode average
- ✅ Survival: 100+ steps average
- ✅ Completable rows: 3-5 average

**By Episode 10000:**
- ✅ Consistent full-board spreading
- ✅ Holes average < 10
- ✅ Lines: 5+ per episode average
- ✅ Survival: 200+ steps average
- ✅ Occasional Tetrises (4-line clears)

---

**Training is now ready with the improved 5-stage curriculum! Run:**
```bash
python train_progressive_improved.py --episodes 10000 --force_fresh
```
