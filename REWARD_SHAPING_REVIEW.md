# Reward Shaping Review

## ✅ Status: FIXED AND READY

All reward shaping functions have been updated to use the corrected board extraction.

---

## 📊 **Current Configuration: `overnight_reward_shaping`**

### Design Philosophy:
- **Single unified shaping function** (no confusing variants)
- **Stable for overnight training** (predictable gradients)
- **Well-bounded outputs** ([-100, 600] range)

### Reward Components:

#### 1. **Base Reward Amplification** (100×)
```python
shaped = reward * 100.0
```
- Amplifies sparse environment rewards (line clears)
- Makes learning signal stronger

#### 2. **Penalties** (Discourage bad states)
```python
shaped -= 0.12 * aggregate_height    # Tall stacks bad
shaped -= 1.30 * holes               # Holes very bad
shaped -= 0.06 * bumpiness           # Uneven surface bad
shaped -= 0.10 * wells               # Deep wells bad
```

#### 3. **Bonuses** (Encourage good behavior)
```python
shaped += 4.0 * horizontal_spread    # Spread pieces across board
shaped += min(steps * 0.02, 3.0)     # Survival bonus (capped at 3.0)
```

#### 4. **Line Clear Rewards**
```python
shaped += lines_cleared * 80.0
if lines_cleared == 4:               # Tetris bonus
    shaped += 120.0                  # Total: 440 for Tetris
```

#### 5. **Death Penalty**
```python
if done:
    shaped -= 30.0                   # Dying is bad
```

---

## 🔧 **Fix Applied**

### Issue Found:
The `extract_board_from_obs()` function was using the OLD buggy extraction:

**Before:**
```python
if full.shape == (24, 18):
    board = full[2:22, 4:14]  # ❌ Included wall rows
```

**After:**
```python
if full.shape == (24, 18):
    board = full[0:20, 4:14]  # ✅ Correct extraction
```

### Impact:
- ✅ Reward metrics now calculated on correct board state
- ✅ No wall contamination in height/hole/bumpiness calculations
- ✅ More accurate reward signals for learning

---

## 📈 **Reward Ranges**

### Expected Values:

| Scenario | Approximate Reward |
|----------|-------------------|
| Single line clear (good state) | +50 to +150 |
| Double line clear | +150 to +250 |
| Triple line clear | +250 to +350 |
| Tetris (4 lines) | +400 to +600 |
| Normal gameplay (no lines) | -20 to +20 |
| Bad state (holes, tall) | -50 to -100 |
| Death | -100 (includes death penalty) |

### Clamping:
All rewards clamped to **[-100, 600]** to prevent:
- Gradient explosion
- Training instability
- Outlier rewards dominating learning

---

## 🎯 **Key Metrics Explanation**

### 1. **Aggregate Height** (0-200 typical)
- Sum of all column heights
- Lower is better (less stacked)
- Weight: 0.12 (light penalty)

### 2. **Holes** (0-200 typical)
- Empty cells with filled cells above them
- Very bad for Tetris strategy
- Weight: 1.30 (heavy penalty)

### 3. **Bumpiness** (0-100 typical)
- Sum of height differences between adjacent columns
- Smooth surface preferred
- Weight: 0.06 (light penalty)

### 4. **Wells** (0-100 typical)
- Columns significantly lower than neighbors
- Can trap pieces
- Weight: 0.10 (light penalty)

### 5. **Horizontal Spread** (0-1)
- Measures how evenly pieces distributed across columns
- Prevents center-stacking
- Weight: 4.0 (bonus for spreading out)

---

## 🔬 **Strengths of This Design**

1. ✅ **Monotonic line rewards**: More lines = more reward (predictable)
2. ✅ **Balanced penalties**: Not too harsh, allows exploration
3. ✅ **Spread bonus**: Actively fights center-stacking bug
4. ✅ **Survival incentive**: Encourages staying alive longer
5. ✅ **Bounded output**: Prevents training instability
6. ✅ **Fast computation**: Simple metrics, no complex calculations

---

## ⚠️ **Potential Weaknesses**

### 1. **Tetris Bonus May Be Too Small**
- Current: 440 total for Tetris (320 base + 120 bonus)
- Consider: Increase to 500-600 to really incentivize 4-line clears

### 2. **Death Penalty Relatively Light**
- Current: -30
- Agent may not fear death enough early in training
- Consider: -50 to -80 for stronger deterrent

### 3. **No Explicit T-Spin Rewards**
- Could add detection for advanced moves
- Not critical for basic training

### 4. **Horizontal Spread Weight**
- Weight 4.0 may be too low given the center-stacking issue
- Monitor if agent still center-stacks, increase to 6-8

---

## 🚀 **Recommendations for Training**

### Start With:
```bash
python train.py --episodes 2000 --reward_shaping positive
```

### If Center-Stacking Persists:
1. Increase spread bonus: Change line 181 from `4.0` to `6.0` or `8.0`
2. Add explicit penalty for center columns

### If Agent Dies Too Often:
1. Increase death penalty: Change line 196 from `30.0` to `50.0`
2. Increase survival bonus cap: Change line 185 from `3.0` to `5.0`

### If Not Clearing Lines:
1. Increase line clear bonuses (lines 190-192)
2. Decrease state penalties (they may be too harsh)

---

## 📋 **Monitoring During Training**

Watch for these in logs:

1. **Average reward per episode**: Should steadily increase
   - Early: -50 to 0
   - Mid: 0 to +100
   - Late: +100 to +300+

2. **Lines cleared per episode**: Should increase
   - Early: 0-5
   - Mid: 5-20
   - Late: 20-50+

3. **Episode length**: Should increase then stabilize
   - Early: 50-200 steps
   - Mid: 200-500 steps
   - Late: 500-2000+ steps

4. **Horizontal spread metric**: Should increase over time
   - Early: 0.3-0.5
   - Target: 0.6-0.8

---

## ✅ **Ready for Training**

- ✅ All extraction bugs fixed
- ✅ Reward shaping tested and validated
- ✅ Sensible default parameters
- ✅ Monitoring metrics identified
- ✅ Tuning guidance provided

**Status: APPROVED FOR TRAINING** 🚀
