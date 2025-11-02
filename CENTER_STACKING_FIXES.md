# Center-Stacking Fixes Applied

## Problem Diagnosis (Episode 10000)
- **Columns used: 4/10** (only columns 3,4,5,6)
- **Columns 0,1,2,7,8,9: COMPLETELY EMPTY** (0 height)
- **Center ratio: 100%** - All pieces in center
- Lines cleared: 0.00 per episode
- Average steps: 11 (dying immediately)
- Reward: -2979 (heavily negative)

## Fixes Applied

### 1. ✅ MASSIVELY Increased Center-Stacking Penalty
**Location:** `progressive_reward_improved.py` line 465-474

**OLD:**
```python
penalty = -100.0 * (center_ratio - 0.5)  # -50 for 100% center
```

**NEW:**
```python
# 70% = -100, 80% = -150, 90% = -200, 100% = -250 PER STEP!
penalty = -500.0 * (center_ratio - 0.5)
```

**Impact:**
- 100% center-stacking now costs **-250 per step** (was -50)
- Over 11 steps: **-2,750 penalty** (was -550)
- This should FORCE the agent to explore outer columns

---

### 2. ✅ MASSIVELY Increased Spread Bonuses

#### Stage 3 (Episodes 1000-2000)
| Reward Component | OLD | NEW | Increase |
|-----------------|-----|-----|----------|
| Spread bonus | +20 | **+40** | 2x |
| Per column used | +4 | **+8** | 2x |
| Outer unused penalty | -5 | **-15** | 3x |

#### Stage 4 (Episodes 2000-5000)
| Reward Component | OLD | NEW | Increase |
|-----------------|-----|-----|----------|
| Spread bonus | +25 | **+50** | 2x |
| Per column used | +6 | **+12** | 2x |
| Outer unused penalty | -8 | **-20** | 2.5x |

#### Stage 5 (Episodes 5000+)
| Reward Component | OLD | NEW | Increase |
|-----------------|-----|-----|----------|
| Spread bonus | +25 | **+60** | 2.4x |
| Per column used | +6 | **+15** | 2.5x |
| Outer unused penalty | -10 | **-30** | 3x |

---

### 3. ✅ Added Explicit Empty Outer Columns Tracking
**Location:** `progressive_reward_improved.py` line 366-370

New metric: `empty_outer_cols` - counts how many outer columns have height = 0

This makes it crystal clear when outer columns are completely unused.

---

## Expected Impact

### Current Reward Breakdown (100% center-stacking, 4 cols, 25 holes, 11 steps):
```
Center-stacking: -250 × 11 = -2,750
Holes:          -3.5 × 25 = -87
Spread bonus:         +0   (no spread)
Columns:         +15 × 4  = +60
Outer unused:    -30 × 6  = -180
Death penalty:           -200
--------------------------------
TOTAL:                 ~-3,157
```

### Target Reward Breakdown (even spread, 8 cols, 10 holes, 50 steps):
```
Center-stacking:       +0   (50% center ratio)
Holes:        -3.5 × 10 = -35
Spread bonus:         +60  (high spread)
Columns:       +15 × 8  = +120
Outer unused:   -30 × 2  = -60
Survival:              +40
Line clears:  +150 × 2  = +300 (quality-scaled)
--------------------------------
TOTAL:                 +425 ✅
```

---

## How to Test

### Start Fresh Training
```bash
python train_progressive_improved.py --episodes 10000 --force_fresh
```

### Check Progress at Episode 5000 (50% checkpoint)
```bash
cat logs/improved_*/DEBUG_SUMMARY.txt
```

**Look for:**
- ✅ Columns used: 6-8 (up from 4)
- ✅ Center ratio: <70% (down from 100%)
- ✅ Reward: increasing toward positive
- ✅ Steps: 20-50 (up from 11)

### Monitor Board States
```bash
tail -50 logs/improved_*/board_states.txt
```

**Should see pieces in columns 0,1,2,7,8,9 (not just 3-6)**

---

## Why This Will Work

1. **Massive Penalty Override:** -250/step for center-stacking is 5x stronger than before
2. **Carrot + Stick:** Huge rewards for spreading (+60 spread, +15/col) AND huge penalties for not spreading (-30/outer)
3. **Progressive Scaling:** Bonuses increase from Stage 3 → 4 → 5 as agent learns
4. **Quality Scaling:** Line clear rewards only maximize when board is clean AND spread

---

## If Still Center-Stacking After 5000 Episodes

Try these additional fixes:

1. **Increase center-stacking penalty to -750 or -1000**
2. **Start with Stage 3 rewards from Episode 0** (skip gentle Stage 1-2)
3. **Add explicit bonus for using edge columns** (0,1,8,9)
4. **Reduce hole penalty slightly** to allow exploration

