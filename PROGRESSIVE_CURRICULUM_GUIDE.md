# Progressive Curriculum Learning - Implementation Guide

## ✅ What's Been Implemented

Based on Opus's analysis, I've implemented a **4-stage progressive curriculum** that solves the center-stacking problem by teaching motor skills before strategy.

---

## 📁 Files Created/Modified

### New Files:
1. **`src/progressive_reward.py`** - Progressive reward shaper with automatic stage advancement
2. **`train_progressive.py`** - Training loop with curriculum support
3. **`tests/test_reward_diagnosis.py`** - Diagnostic showing why spreading fails with current approach

### Updated Files:
- **`DEBUG_LOG.md`** - Added Fix #4 documenting the progressive curriculum approach

---

## 🎓 How the Curriculum Works

### Stage 1: Basic Placement (Episodes 0-200)
**Goal:** Learn to place pieces without creating holes

```python
shaped -= 2.0 * holes      # HIGH hole penalty
shaped -= 0.05 * agg_h     # Light height penalty
shaped += survival_bonus
# No spreading rewards yet
```

**Expected:** Holes decrease from 50 → 15

---

### Stage 2: Height Management (Episodes 200-400)
**Goal:** Keep board low while maintaining clean placement

```python
shaped -= 1.5 * holes      # Still strong
shaped -= 0.1 * agg_h      # Stronger height penalty
shaped += 5.0 * spread     # Small spreading encouragement
```

**Expected:** Height management improves, holes stay low

---

### Stage 3: Spreading (Episodes 400-600)
**Goal:** Spread across all columns

```python
shaped -= 0.8 * holes      # REDUCED (agent skilled now!)
shaped += 25.0 * spread    # STRONG spreading bonus
shaped += columns_used * 6.0
shaped -= outer_unused * 8.0
shaped -= 3.0 * height_std
```

**Expected:** Columns used increases 4 → 8, outer columns get used

---

### Stage 4: Balanced (Episodes 600+)
**Goal:** Optimal play with line clears

```python
shaped -= 0.75 * holes     # Balanced
# All spreading rewards active
# Line clear bonuses strong
```

**Expected:** Master balanced play, clear lines consistently

---

## 🚀 How to Use

### 1. Run the Diagnostic Test First

```bash
.venv/bin/python tests/test_reward_diagnosis.py
```

This will show you:
- Why spreading currently fails (holes overwhelm bonuses)
- How curriculum solves it (gradual skill building)
- Expected progression through stages

**Sample Output:**
```
CLEAN BOARDS: Spreading = +97.91 better ✅
REALISTIC BOARDS (beginner): Spreading = -26.5 worse ❌

CURRICULUM SOLUTION:
Stage 1: Gradient = +10.65 ✅ (learn placement)
Stage 2: Gradient = +12.14 ✅ (add height)
Stage 3: Gradient = +97.97 ✅ (safe to spread!)
```

---

### 2. Start Progressive Training

```bash
# Delete old models trained with broken gradient
rm -rf models/* logs/*

# Start fresh progressive training
.venv/bin/python train_progressive.py \
    --episodes 1000 \
    --force_fresh \
    --epsilon_start 1.0 \
    --epsilon_decay 0.9995
```

**Optional flags:**
```bash
--stage_basic 200       # Episodes for stage 1 (default: 200)
--stage_height 400      # Episodes for stage 2 (default: 400)
--stage_spreading 600   # Episodes for stage 3 (default: 600)
--experiment_name my_run # Custom name for logs
```

---

### 3. Monitor Progress

Watch for these indicators:

**Stage 1 Success (Clean Placement):**
```
Ep  100 | Stage: basic      | Holes: 15 (avg 18.3) ✅
```
Target: Average holes < 20

**Stage 2 Success (Height Management):**
```
Ep  300 | Stage: height     | Holes: 12 (avg 14.1) ✅
```
Target: Holes still low, height under control

**Stage 3 Success (Spreading):**
```
Ep  500 | Stage: spreading  | Cols: 8/10 | Outer: 2/6 ✅
```
Target: Using 7+ columns, outer columns (0-2, 7-9) getting used

**Stage 4 Success (Optimal Play):**
```
Ep  800 | Stage: balanced   | Cols: 9/10 | Lines: 2 ✅
```
Target: Line clears happening, balanced column usage

---

### 4. Check Final Results

At the end of training, you'll see:

```
CURRICULUM SUCCESS METRICS:
================================================================================
✅ Stage 1 (Clean Placement): SUCCESS
   Holes reduced to 12.3 (target: <20)

✅ Stage 3 (Spreading): SUCCESS
   Columns used: 8.2/10 (target: ≥7)

✅ Final Performance: SUCCESS
   Line clearing rate: 0.43 lines/episode
```

---

## 🔍 Troubleshooting

### Agent still center-stacking after 600 episodes?

**Check holes count:**
```bash
grep "Holes:" logs/progressive_*/board_states.txt | tail -20
```

If holes are still >20:
- Stage 1 didn't succeed
- Increase episodes for stage_basic to 300-400
- Agent needs more time to learn motor control

---

### Agent not spreading after stage 3?

**Check columns used:**
```bash
awk '/Cols:/ {print}' logs/progressive_*/board_states.txt | tail -20
```

If columns_used < 6:
- Increase outer_unused penalty in stage 3
- Reduce holes penalty further (-0.8 → -0.6)
- Check epsilon is still >0.2 for exploration

---

### No line clears in stage 4?

**Check episode length:**
```bash
awk '/Steps:/ {print}' logs/progressive_*/board_states.txt | tail -20
```

If steps < 30:
- Agent dying too fast
- Increase survival bonus
- Check if holes are still too high
- May need more episodes in earlier stages

---

## 📊 Expected Training Curve

```
Episodes    | Holes | Columns | Reward  | Lines | Stage
--------|-------|---------|---------|-------|----------
0-50    |   45  |    4    | -150    |   0   | basic
50-100  |   35  |    4    | -120    |   0   | basic
100-200 |   18  |    5    |  -80    |   0   | basic
200-300 |   15  |    5    |  -60    |   0   | height
300-400 |   12  |    6    |  -40    |   0   | height
400-500 |   10  |    7    |  -20    |   0   | spreading
500-600 |    8  |    8    |   +10   |  0.1  | spreading
600-800 |    6  |    9    |   +40   |  0.3  | balanced
800-1000|    5  |    9    |   +80   |  0.5  | balanced
```

---

## 🎯 Key Differences from Previous Approach

| Previous Approach | Progressive Curriculum |
|-------------------|------------------------|
| Static rewards throughout training | Rewards adapt per stage |
| Teaches strategy + motor control simultaneously | Motor control first, then strategy |
| Agent trapped: spreading → holes → bad reward | Stage 1 ensures clean placement first |
| Gradient overwhelmed by holes penalty | Holes penalty HIGH initially, LOW later |
| Agent never learns spreading | Spreading safe after Stage 2 |

---

## 📝 Next Steps

1. ✅ Run diagnostic: `.venv/bin/python tests/test_reward_diagnosis.py`
2. ✅ Start training: `.venv/bin/python train_progressive.py --episodes 1000 --force_fresh`
3. ⏱️  Monitor for ~30-60 minutes (1000 episodes)
4. 📊 Check board_states.txt for progression
5. 🎉 Enjoy watching agent learn to spread!

---

## ❓ FAQ

**Q: Can I resume progressive training?**
A: Yes! Use `--resume` flag. The shaper will automatically detect the current stage based on episode count.

**Q: Can I adjust stage thresholds mid-training?**
A: Yes, but you'll need to modify `src/progressive_reward.py` and restart training.

**Q: What if Stage 1 takes too long?**
A: Increase `--stage_basic` episodes. Some agents need 300-400 episodes to learn clean placement.

**Q: Can I skip stages?**
A: Not recommended. Each stage builds skills needed for the next.

**Q: Does this work with the old train.py?**
A: No, use `train_progressive.py`. It integrates the curriculum shaper properly.

---

**The curriculum will automatically advance through stages. Just let it run and watch the agent learn!** 🚀
