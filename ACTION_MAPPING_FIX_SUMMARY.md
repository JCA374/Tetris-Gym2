# Action Mapping Bug - Fixed!

## 🔴 Critical Bug Found and Fixed

### The Problem

Your action mapping in `config.py` was **completely wrong**! Every action was mapped to the wrong ID.

**Actual tetris-gymnasium v0.3.0 mapping:**
```python
0: move_left
1: move_right
2: move_down
3: rotate_clockwise
4: rotate_counterclockwise
5: hard_drop
6: swap
7: no_op
```

**Your WRONG mapping (before fix):**
```python
ACTION_NOOP=0       # Was actually LEFT!
ACTION_LEFT=1       # Was actually RIGHT!
ACTION_RIGHT=2      # Was actually DOWN!
ACTION_DOWN=3       # Was actually ROTATE_CW!
ACTION_ROTATE_CW=4  # Was actually ROTATE_CCW!
ACTION_ROTATE_CCW=5 # Was actually HARD_DROP!
ACTION_HARD_DROP=6  # Was actually SWAP!
ACTION_SWAP=7       # Was actually NOOP!
```

###  What This Meant

**Your agent was training with completely inverted controls!**

- When it tried to move LEFT → Actually moved RIGHT
- When it tried to move RIGHT → Actually moved DOWN
- When it tried to HARD_DROP → Actually SWAPPED pieces
- When it tried to do NOTHING → Actually moved LEFT

No wonder the agent couldn't learn properly!

---

## ✅ Fix Applied

**File: `config.py` lines 48-53**

```python
# FIXED: Correct action mapping for tetris-gymnasium v0.3.0
# ActionsMapping(move_left=0, move_right=1, move_down=2, rotate_clockwise=3,
#                rotate_counterclockwise=4, hard_drop=5, swap=6, no_op=7)
ACTION_LEFT=0; ACTION_RIGHT=1; ACTION_DOWN=2; ACTION_ROTATE_CW=3
ACTION_ROTATE_CCW=4; ACTION_HARD_DROP=5; ACTION_SWAP=6; ACTION_NOOP=7
ACTION_MEANINGS = {0:"LEFT",1:"RIGHT",2:"DOWN",3:"ROTATE_CW",4:"ROTATE_CCW",5:"HARD_DROP",6:"SWAP",7:"NOOP"}
```

---

## 🧪 Testing Results

### Movement Range Test ✅

**ALL COLUMNS (0-9) ARE REACHABLE!**

- **LEFT**: Can reach column 0 with ~8-10 LEFT actions
- **RIGHT**: Can reach column 9 with ~8-10 RIGHT actions
- Center-stacking was due to WRONG action mapping + insufficient movement

```
Test result:
  Column 0:  12 pieces (reachable!)
  Column 1:  12 pieces
  Column 2:  12 pieces
  Column 3:   9 pieces
  Column 4:  12 pieces
  Column 5:  12 pieces
  Column 6:  11 pieces
  Column 7:  11 pieces
  Column 8:  11 pieces
  Column 9:  11 pieces (reachable!)
```

### Line Clearing Status ⚠️

**Line clearing is POSSIBLE but VERY DIFFICULT with random/naive strategies**

- 1000 random episodes: **0 lines cleared**
- Max row fullness achieved: **9/10 cells**
- Random actions are insufficient - need proper Tetris strategy

**Why it's hard:**
1. **Auto-gravity**: Pieces fall down every step
2. **Piece shapes**: Different tetrominos (I, O, T, L, J, S, Z) require strategic placement
3. **Timing**: Need multiple LEFT/RIGHT moves before piece locks
4. **Strategy**: Requires rotation + precise positioning

**What this means for RL training:**
- Agent CAN learn to clear lines
- But reward shaping needs to heavily incentivize proper piece placement
- Early training will have very few/no line clears
- Line clear rewards should be VERY high to reinforce rare events

---

## 📊 Impact on Your Training

### Before Fix (Wrong Actions):
```
Episodes: 2000
Lines cleared: 9 total (0.0045/episode)
Center-stacking: Severe (columns 3-7 only)
Columns used: Limited range
Agent confused: Controls inverted!
```

### After Fix (Correct Actions):
**You need to retrain from scratch!**

All previous training data is invalid because the agent learned:
- LEFT means RIGHT
- RIGHT means DOWN
- etc.

### Expected New Training Results:

**Episodes 0-500 (Early Learning):**
- Lines/episode: 0-2
- Agent learns basic movement
- Discovers full column range (0-9)
- Very few line clears (it's hard!)

**Episodes 500-1500 (Skill Development):**
- Lines/episode: 2-10
- Agent learns piece rotation
- Better placement strategy
- More consistent clears

**Episodes 1500-3000+ (Mastery):**
- Lines/episode: 10-50+
- Strategic piece placement
- Uses all columns effectively
- Regular Tetris (4-line) clears

---

## 🚀 Action Plan

### Step 1: Delete Old Training Data ✅ CRITICAL

```bash
rm -rf models/*
rm -rf logs/*
```

**Why:** Old models learned with wrong action mapping and are unusable.

### Step 2: Update Reward Shaping

Since line clearing is rare early in training, increase line clear rewards:

**Edit `src/reward_shaping.py` lines 197-201:**

```python
# Line clear bonus (INCREASED for rarity)
lines = int(info.get("lines_cleared", 0))
if lines > 0:
    shaped += lines * 150.0  # Increased from 80
    if lines == 4:  # Tetris bonus
        shaped += 250.0      # Increased from 120
```

### Step 3: Start Fresh Training

```bash
.venv/bin/python train.py \
    --episodes 5000 \
    --reward_shaping positive \
    --force_fresh \
    --epsilon_decay 0.99995
```

**Why 5000 episodes:** Line clearing takes longer to learn now that we know it's difficult.

### Step 4: Monitor Metrics

Watch for these signs of learning:

**Good signs:**
- ✅ Pieces spreading across columns 0-9 (not just 3-7)
- ✅ Agent using rotation (not just LEFT/RIGHT/DROP)
- ✅ Lines cleared increasing over time (even if slow)
- ✅ Survival time increasing
- ✅ Episode rewards trending upward

**Bad signs:**
- ❌ Still only using columns 3-7 (suggests action mapping still wrong somehow)
- ❌ Zero line clears after 1000 episodes (may need more exploration)
- ❌ Immediate game overs (<10 steps per episode)

---

## 🎯 Updated Expectations

### Realistic Metrics

| Metric | Episodes 0-500 | Episodes 500-1500 | Episodes 1500-3000+ |
|--------|---------------|-------------------|---------------------|
| **Lines/Episode** | 0-2 | 2-10 | 10-50+ |
| **Avg Steps** | 50-150 | 150-300 | 300-1000+ |
| **Columns Used** | 2-8 | 1-9 | 0-9 |
| **Rotation Usage** | Rare | Occasional | Frequent |
| **Max Height** | 18-20 | 15-18 | 10-15 |

### What "Good" Looks Like

**Episode 2000 example:**
```
Episode 2000:
  Lines cleared: 12
  Steps survived: 450
  Column heights: [8, 12, 10, 14, 13, 11, 9, 10, 7, 5]
  Holes: 8
  Max height: 14
  Used columns: 0-9 (all!)
  Reward: +850
```

---

## 🔍 Root Cause Analysis

**Why was the action mapping wrong?**

Looking at the original `config.py`, it seems the mapping was based on an assumption or different Tetris environment. The actual tetris-gymnasium uses:

```python
class ActionsMapping(IntEnum):
    move_left = 0
    move_right = 1
    move_down = 2
    rotate_clockwise = 3
    rotate_counterclockwise = 4
    hard_drop = 5
    swap = 6
    no_op = 7
```

The fix aligns our constants with the actual enum values.

---

## ✅ Summary

| Issue | Status |
|-------|--------|
| **Action mapping** | ✅ **FIXED** |
| **Board extraction** | ✅ Fixed earlier (rows 0-19, not 2-21) |
| **4-channel wrapper** | ✅ Working |
| **Reward shaping** | ✅ Fixed (handles 4-channel) |
| **Column accessibility** | ✅ ALL columns (0-9) reachable |
| **Line clearing** | ⚠️ Possible but difficult (requires strategy) |
| **Training viability** | ✅ Ready to train! |

---

## 🎓 Key Learnings

1. **Always verify action mappings** against the actual environment
2. **Test with raw environment** (no wrappers) to isolate issues
3. **Line clearing in Tetris is genuinely hard** - not just a bug!
4. **Inverted controls explain center-stacking** - agent couldn't reach outer columns
5. **RL agents need MANY episodes** to learn complex strategies like Tetris

---

## 🙏 Credit

**Your intuition was RIGHT!** You suspected:
- Center-stacking was abnormal ✅
- Board size might be wrong ✅
- Lines weren't clearing ✅

The root cause was the action mapping bug, which we've now fixed completely.

**Next:** Retrain from scratch and watch your agent finally learn proper Tetris! 🚀
