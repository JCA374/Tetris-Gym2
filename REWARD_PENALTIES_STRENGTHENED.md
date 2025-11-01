# Reward Penalties Strengthened - Center-Stacking Fix

## 🔍 Diagnosis from Your Training Logs

Analyzed: `logs/fixed_training_20251101_101243/board_states.txt`

### What I Found:

**Episode 1:**
- Heights: `[0, 0, 0, 18, 20, 19, 19, 0, 0, 0]`
- Reward: -1352.8
- Steps: 29

**Episode 1971 (latest):**
- Heights: `[0, 0, 0, 13, 19, 18, 18, 0, 0, 0]`
- Reward: -227.0
- Steps: 13

**Result: NO IMPROVEMENT in column usage after 2000 episodes!**

Still only using columns 3-6. Outer columns (0-2, 7-9) completely unused.

---

## 🔴 Root Causes Identified

### Problem #1: Penalties TOO WEAK ❌

**Bumpiness:**
- Old penalty: -0.06 × 38 = **-2.28**
- Should be: -1.0 × 38 = **-38**
- **17x too weak!**

**Height std dev (center-stacking indicator):**
- Old penalty: -0.5 × 8.5 = **-4.23**
- Should be: -10.0 × 8.5 = **-85**
- **20x too weak!**

**Outer columns unused:**
- Old penalty: -5.0 × 6 = **-30**
- Should be: -10.0 × 6 = **-60**
- **2x too weak!**

### Problem #2: Agent Dies Too Fast 💀

- Surviving only **13-15 steps** per episode
- **Never experiences** what happens with balanced play
- Dies before it can learn spreading is better

### Problem #3: Death Penalty Too Harsh ⚠️

- Old: **-30** per death
- Discourages exploration
- Agent afraid to try new strategies

---

## ✅ Fixes Applied

### 1. Bumpiness Penalty: **17x STRONGER**
```python
# BEFORE:
shaped -= 0.06 * bump

# AFTER:
shaped -= 1.0 * bump  # 17x stronger!
```

For center-stacking (bumpiness=38):
- Before: -2.28
- After: **-38.0** ✅

---

### 2. Height Std Dev Penalty: **20x STRONGER**
```python
# BEFORE:
shaped -= 0.5 * height_std

# AFTER:
shaped -= 10.0 * height_std  # 20x stronger!
```

For center-stacking (std=8.5):
- Before: -4.25
- After: **-85.0** ✅

---

### 3. Outer Column Penalty: **2x STRONGER**
```python
# BEFORE:
shaped -= outer_unused * 5.0

# AFTER:
shaped -= outer_unused * 10.0  # 2x stronger!
```

For 6 unused outer columns:
- Before: -30
- After: **-60** ✅

---

### 4. Survival Bonus: **5x STRONGER**
```python
# BEFORE:
shaped += min(steps * 0.02, 3.0)

# AFTER:
shaped += min(steps * 0.1, 10.0)  # 5x stronger bonus, higher cap
```

Encourages agent to survive longer and discover balanced strategies.

---

### 5. Death Penalty: **3x WEAKER**
```python
# BEFORE:
shaped -= 30.0

# AFTER:
shaped -= 10.0  # Reduced to encourage exploration
```

Less harsh so agent isn't afraid to try new strategies.

---

### 6. Clamp Range: **WIDENED**
```python
# BEFORE:
return np.clip(shaped, -100.0, 600.0)

# AFTER:
return np.clip(shaped, -200.0, 600.0)  # Allow stronger penalties
```

---

## 📊 Reward Comparison (Before vs After)

### BEFORE (Weak Penalties):
```
Severe center-stacking [0,0,0,19,19,19,19,0,0,0]: -31.56
Balanced [5,6,8,10,12,11,9,7,5,4]:                 +22.23

Difference: +53.79 (moderate signal)
```

### AFTER (Strong Penalties):
```
Severe center-stacking [0,0,0,19,19,19,19,0,0,0]: -177.71 ❌
Balanced [5,6,8,10,12,11,9,7,5,4]:                  -8.67 ✅

Difference: +169.04 (MASSIVE signal!)
```

**3x stronger signal to avoid center-stacking!**

---

## 🚀 Training Instructions

### Step 1: Delete old models (CRITICAL!)

```bash
rm -rf models/*
rm -rf logs/*
```

**Why:** Old models learned with weak penalties. They're stuck in bad habits.

### Step 2: Start fresh training

```bash
.venv/bin/python train.py \
    --episodes 5000 \
    --reward_shaping positive \
    --force_fresh \
    --epsilon_start 1.0 \
    --epsilon_decay 0.9999
```

### Step 3: Monitor progress

Watch the board states log:
```bash
tail -f logs/<experiment_name>/board_states.txt
```

---

## 📈 Expected Results

### Episodes 0-100: Breaking Old Habits
```
Column heights: [0, 0, 0, 19, 19, 19, 19, 0, 0, 0]
Reward: -150 to -200  ← Much more negative now!
Agent learns: "Center-stacking = VERY BAD"
```

### Episodes 100-500: Discovery Phase
```
Column heights: [0, 1, 3, 15, 18, 17, 12, 6, 2, 0]
Reward: -80 to -120
Agent starts: Using columns 1-2 and 7-8 occasionally
```

### Episodes 500-1500: Learning Phase
```
Column heights: [2, 4, 7, 12, 15, 14, 10, 6, 3, 1]
Reward: -40 to -60
Agent learns: Spreading pieces across more columns
```

### Episodes 1500-5000: Mastery
```
Column heights: [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
Reward: -10 to +30
Agent masters: Balanced distribution, clearing lines
```

---

## 🎯 What to Watch For

### ✅ Good Signs (Learning!)

- **Column heights spreading out**: Not just 3-6 anymore
- **Rewards improving**: -200 → -100 → -50 → 0 → +50
- **Longer episodes**: 15 steps → 50 steps → 150 steps
- **Outer columns used**: Columns 0-2 and 7-9 showing non-zero heights
- **Lines being cleared**: Actually clearing lines!

### ❌ Bad Signs (Not Learning)

- **Still [0,0,0,X,X,X,X,0,0,0]** after 500 episodes
- **Rewards staying at -150 to -200**
- **Episodes still only 10-15 steps**
- **No improvement in bumpiness** (stays at 38-40)

If bad signs persist after 500 episodes:
1. Check epsilon is high (should be >0.6 early training)
2. Verify action mapping is correct (LEFT=0, RIGHT=1, etc.)
3. Check Q-network is learning (loss decreasing)

---

## 📝 Summary of Changes

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Bumpiness penalty** | -0.06× | -1.0× | **17x stronger** |
| **Height std penalty** | -0.5× | -10.0× | **20x stronger** |
| **Outer column penalty** | -5.0 | -10.0 | **2x stronger** |
| **Survival bonus** | +0.02× (max 3) | +0.1× (max 10) | **5x stronger** |
| **Death penalty** | -30 | -10 | **3x weaker** |
| **Center-stack penalty** | ~-32 | ~-178 | **5.5x stronger** |
| **Reward difference** | +54 | +169 | **3x better signal** |

---

## 💡 Why This Will Work

1. **Massive penalty** for center-stacking (-178 vs -32)
2. **Strong incentive** to survive longer (explore balanced play)
3. **Less harsh death** encourages trying new strategies
4. **Clear gradient** from bad (-178) to good (-9) play

The agent now has a **crystal-clear signal**:
- Center-stacking = **HUGE PENALTY**
- Balanced play = **MUCH BETTER**

---

## 🎓 Technical Explanation

**Why were the penalties too weak?**

The original penalties were calibrated for "gentle shaping" - nudging the agent slightly. But when the agent is stuck in a strong local optimum (center-stacking), gentle nudges don't work.

**Why strengthen survival bonus?**

The agent needs to survive long enough to **discover** that balanced play exists. With 13-step episodes, it never experiences:
- What happens when you use column 0
- What reward you get for balanced heights
- How line clearing feels

Stronger survival bonus → longer episodes → more exploration → discovery of better strategies.

**Why reduce death penalty?**

A harsh death penalty (-30) makes the agent **risk-averse**. It learns "don't try anything new, just survive." By reducing to -10, we say "it's OK to die while exploring."

---

## ✅ Ready to Train!

The reward shaping is now **properly calibrated** to:
- **Punish** center-stacking severely
- **Reward** balanced distribution
- **Encourage** longer survival
- **Allow** exploration

Start training and watch your agent finally learn to spread pieces! 🚀

---

**Status:** ✅ **READY - Reward penalties strengthened by up to 20x**
