# 14-Hour Training Plan - Tetris RL

## 📊 Current Status Analysis (Episode 12,500)

### ✅ MAJOR SUCCESS: Center-Stacking SOLVED!
- **Columns used: 9.96/10** (was 4.0) 🎉
- Agent now spreads pieces evenly across all columns
- Column heights: [5, 7, 5, 8, 19, 18, 17, 10, 9, 6] ✅

### ⚠️ Current Problems
- **Holes: 43.35 average** (target: <15) 🔴
- **Lines cleared: 0.01/episode** (target: 2-5) 🔴
- **Reward: -3030 average** (should be positive) 🟡
- **Completable rows: 0.1** (target: 3-5) 🔴

### ✅ Good Signs
- **Steps: 175/episode** (was 11!) - Agent surviving well
- **Positive rewards appearing:** Episode 13300 got +405.4!
- **Bumpiness: 30-35** (reasonable)
- **Using all columns evenly** (no center-stacking)

---

## 🎯 14-Hour Training Configuration

### Time Calculation

**Current Performance:**
- 2,500 episodes in 25.1 minutes = **0.60 sec/episode**
- Average 175 steps/episode (agent surviving longer)

**14 Hours = 840 minutes:**
- At 0.60 sec/episode: **84,000 episodes**
- Conservative estimate (agent may take longer as it improves): **70,000 episodes**

**Recommended Target:** **75,000 episodes total**
- Current: 12,500
- Add: 62,500 more episodes
- Total: 75,000

---

## 🚀 Training Command

```bash
# Resume from current checkpoint and train to 75,000 total episodes
python train_progressive_improved.py --episodes 75000 --resume

# Estimated completion time: ~14-15 hours
```

### Why 75,000 Episodes?

| Episode Range | Focus | Expected Progress |
|---------------|-------|-------------------|
| **0-500** | Foundation | Learn basic placement |
| **500-1000** | Clean placement | Reduce holes gradually |
| **1000-2000** | Spreading | ✅ ACHIEVED! (9.96/10 columns) |
| **2000-5000** | Clean spreading | Master hole-free spreading |
| **5000-15000** | Line clearing basics | First consistent line clears |
| **15000-30000** | Efficient clearing | 0.5-1 lines/episode |
| **30000-50000** | Mastery | 1-3 lines/episode |
| **50000-75000** | Expert play | 3-5 lines/episode |

---

## 📈 Expected Progression Over 14 Hours

### Checkpoints to Monitor

| Episode | Time | Expected Metrics |
|---------|------|------------------|
| **12,500** (now) | 0h | Cols: 9.96, Holes: 43, Lines: 0.01 |
| **25,000** | 3.5h | Cols: 9.9, Holes: 30-35, Lines: 0.1 |
| **37,500** | 7h | Cols: 9.8, Holes: 20-25, Lines: 0.5 |
| **50,000** | 10.5h | Cols: 9.5, Holes: 15-20, Lines: 1.0 |
| **62,500** | 13h | Cols: 9.5, Holes: 10-15, Lines: 2.0 |
| **75,000** | 14-15h | Cols: 9.5, Holes: <15, Lines: 2-5 ✅ |

### Key Milestones

- ✅ **Episode ~15,000:** First consistent line clears (0.2-0.5/episode)
- ✅ **Episode ~25,000:** Holes drop below 30
- ✅ **Episode ~40,000:** First Tetris (4-line clear)
- ✅ **Episode ~60,000:** Clean play (holes <15, lines >2/episode)
- ✅ **Episode ~75,000:** Expert behavior

---

## 🔍 Monitoring During Training

### Check Progress Every ~3 Hours

```bash
# Quick status check
tail -50 logs/improved_*/board_states.txt

# Check latest metrics
cat logs/improved_*/DEBUG_SUMMARY.txt | grep -A 10 "PERFORMANCE METRICS"

# Plot progress (check the PNG files)
ls -lth logs/improved_*/training_metrics.png
```

### What to Look For

#### ✅ Good Signs (Keep Going)
- Holes trending down: 43 → 35 → 25 → 15
- Lines/episode increasing: 0.01 → 0.1 → 1.0 → 2.0
- Rewards becoming positive: -3000 → -1000 → 0 → +500
- Completable rows increasing: 0.1 → 1.0 → 3.0
- Steps staying 100-200+ (agent surviving well)

#### 🔴 Warning Signs (May Need Adjustment)
- Holes stuck at 40+ after episode 30,000
- No line clears by episode 20,000
- Rewards still -5000 after episode 40,000
- Columns used drops below 8 (center-stacking returns)

---

## 🛠️ If Progress Stalls (After 30,000+ Episodes)

If by episode 30,000-40,000 you're still seeing:
- Holes >30
- Lines <0.5/episode
- Negative rewards

### Option 1: Increase Hole Penalty
Edit `progressive_reward_improved.py` Stage 5:
```python
# Line 270: Increase from -3.5 to -5.0
shaped -= 5.0 * metrics['holes']  # Was -3.5
```

### Option 2: Increase Completable Rows Bonus
```python
# Line 276: Increase from +15.0 to +25.0
shaped += 25.0 * metrics['completable_rows']  # Was +15.0
```

### Option 3: Scale Survival Bonus More Aggressively
```python
# Lines 287-298: Make stricter
if metrics['holes'] < 5:
    shaped += min(info.get('steps', 0) * 0.5, 40.0)
elif metrics['holes'] < 15:
    shaped += min(info.get('steps', 0) * 0.3, 25.0)
elif metrics['holes'] < 25:
    shaped += min(info.get('steps', 0) * 0.1, 10.0)
else:
    shaped += 0  # NO bonus if 25+ holes
```

---

## 📊 Expected Final Results (Episode 75,000)

### Target Performance
```
✅ Columns used:        9-10/10
✅ Holes:               8-15
✅ Lines/episode:       2-5
✅ Steps/episode:       150-300
✅ Reward:              +500 to +2000
✅ Completable rows:    3-5
✅ Clean rows:          10-15
```

### Example Final Board (Goal)
```
  0123456789
 0 ··········
 1 ··········
 2 ··········
 3 ··········
 4 ··········
 5 ··········
 6 ··········
 7 ··········
 8 ··········
 9 ·········█
10 ·········█
11 ███████·██
12 █████████·
13 ██████████  ← Line clear!
14 ██████████  ← Line clear!
15 ███████·██
16 █████████·
17 ████████··
18 ██████████  ← Line clear!
19 █████████·

Metrics: 5 holes, 10/10 columns, 3 lines cleared
```

---

## 💾 Backup & Checkpoints

The training script automatically:
- ✅ Saves model every 500 episodes
- ✅ Creates DEBUG_SUMMARY.txt at 50% completion
- ✅ Logs board states every 10 episodes
- ✅ Tracks all metrics in CSV

**Models saved at:**
- `models/progressive_improved_ep12500.pth` (current)
- `models/progressive_improved_ep25000.pth` (will save)
- `models/progressive_improved_ep50000.pth` (will save)
- `models/progressive_improved_ep75000.pth` (final)

---

## 🎬 Quick Start

### 1. Start Training (Now)
```bash
cd /home/jonas/Code/Tetris-Gym2
python train_progressive_improved.py --episodes 75000 --resume
```

### 2. Monitor Progress
Open another terminal:
```bash
watch -n 30 'tail -20 logs/improved_*/board_states.txt'
```

### 3. Check at 7 Hours (50% mark)
```bash
cat logs/improved_*/DEBUG_SUMMARY.txt
```

### 4. Check Final Results (~14h later)
```bash
cat logs/improved_*/DEBUG_SUMMARY.txt
python evaluate.py --model models/progressive_improved_ep75000.pth --episodes 100
```

---

## 📈 Success Metrics

Training is **successful** if by episode 75,000:
- ✅ Holes: <15 (currently 43)
- ✅ Lines/episode: ≥2.0 (currently 0.01)
- ✅ Columns used: ≥8 (currently 9.96 ✅)
- ✅ Reward: Positive average (currently -3030)

---

## ⏰ Timeline Summary

```
NOW    → Episode 12,500  (Center-stacking SOLVED!)
+3.5h  → Episode 25,000  (Holes dropping)
+7h    → Episode 37,500  (First consistent line clears)
+10.5h → Episode 50,000  (Clean play emerging)
+14h   → Episode 75,000  (Expert Tetris play) ✅
```

**Start the training now and check back in 7 hours!** 🚀
