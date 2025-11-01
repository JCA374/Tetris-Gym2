# Tetris AI Debug Log

## Quick Reference

**Current Status:** ✅ Local optimum trap fixed - ready for fresh training

**Key Files:**
- `src/reward_shaping.py` - Reward calculations
- `train.py` - Training loop (no reward clamp)
- `tests/test_reward_system_complete.py` - Reward verification test

**Training Command:**
```bash
rm -rf models/* logs/* && .venv/bin/python train.py --episodes 2000 --reward_shaping positive --force_fresh
```

---

## Debug History

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

**Status:** Should overcome holes penalty and learn to spread

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

**Usage:**
```bash
# View board states
cat logs/latest/board_states.txt | grep "Column heights"
```

**Key Insight:** This revealed the holes problem in Fix #3

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

---

## Key Metrics Reference

### Reward Values (Current - Fix #3)
```
Perfect balance:    +68.19/step
Good spreading:     +40.73/step
Slight spreading:   -30.30/step
Center-stacking:    -71.40/step
Empty board:        -35.30/step

Gradient (good → center): 112 points
```

### Penalty Weights (Current)
```
Aggregate height: -0.05×
Holes:            -0.75×  ⚠️ Reduced from -2.0
Bumpiness:        -0.50×
Wells:            -0.10×
Spread bonus:     +25.0×  ⚠️ Increased from +15.0
Column usage:     +6.0×   ⚠️ Increased from +4.0
Outer unused:     -8.0×   ⚠️ Increased from -5.0
Height std:       -3.0×   ⚠️ Increased from -2.0
Survival:         +0.2× (max 20)
Death:            -5.0
Line clear:       +80× per line
Tetris bonus:     +120
```

---

## Common Issues & Solutions

### Agent center-stacking?
1. Check holes count: `grep "Holes:" logs/latest/board_states.txt`
2. Run reward test: `.venv/bin/python tests/test_reward_system_complete.py`
3. Verify gradient > 50 points
4. Check epsilon > 0.5 for exploration

### Rewards all the same value?
1. Check for clamping in train.py (should be NONE)
2. Verify penalties aren't too strong
3. Run reward test to see gradient

### Agent not exploring outer columns?
1. Check outer_unused penalty is strong enough (currently -8.0)
2. Verify spread bonus is working (currently +25.0)
3. Increase epsilon for more exploration

### No line clears?
1. Check episode length (agent dying too fast?)
2. Verify survival bonus (should encourage longer play)
3. Check if agent has 4-channel vision

---

## Test Commands

```bash
# Test reward system
.venv/bin/python tests/test_reward_system_complete.py

# Check board states from latest run
ls -lt logs/*/board_states.txt | head -1 | awk '{print $NF}' | xargs cat | head -100

# Analyze holes vs column usage
awk '/^Episode/ {ep=$0} /Holes:/ {print ep; print}' logs/latest/board_states.txt | head -40

# Train fresh
rm -rf models/* logs/* && .venv/bin/python train.py --episodes 2000 --reward_shaping positive --force_fresh
```

---

## Next Debug Steps (If Still Center-Stacking)

1. **Check actual training rewards:**
   ```bash
   grep "Episode.*Reward:" logs/latest/board_states.txt | head -20
   ```

2. **Verify gradient exists:**
   ```bash
   .venv/bin/python tests/test_reward_system_complete.py | grep "GRADIENT"
   ```

3. **Analyze holes pattern:**
   ```bash
   awk '/Holes:/ {print $3}' logs/latest/board_states.txt | head -50
   ```

4. **If spreading still creates too many holes:**
   - Reduce holes penalty further: `-0.75` → `-0.5`
   - Increase outer penalty: `-8.0` → `-10.0`
   - Increase survival bonus to encourage longer exploration

5. **If rewards look good but no learning:**
   - Check Q-network loss values
   - Verify epsilon decay (should be 0.9999 for long training)
   - Check if agent survives long enough (>20 steps/episode)

---

## Files Changed Summary

**Modified:**
- `src/reward_shaping.py` - Multiple iterations of penalty tuning
- `train.py` - Removed reward clamp, added board logging
- `src/utils.py` - Added board state logging
- `config.py` - Fixed action mapping

**Added:**
- `tests/test_reward_system_complete.py` - Comprehensive reward test
- `DEBUG_LOG.md` - This file

**Clean up candidates:**
- Old markdown files (see removal commands below)
