# Board State Logging - Feature Added

## ✅ What Was Added

A new logging feature that saves visual representations of the final board state at the end of each logged episode.

## 📁 Where to Find It

During training, board states are saved to:
```
logs/<experiment_name>/board_states.txt
```

For example:
```
logs/tetris_experiment_20250115_143022/board_states.txt
```

## 📊 What's Logged

For each episode (every `--log_freq` episodes, default: 10), the log includes:

### 1. Episode Metadata
- Episode number
- Reward achieved
- Steps survived
- Lines cleared

### 2. Board Metrics
- Column heights
- Number of holes
- Bumpiness
- Max height
- Max row fullness

### 3. Visual Board Representation
- 20 rows × 10 columns
- `█` = filled cell
- `·` = empty cell
- Row fullness count (X/10 cells)

## 📝 Example Output

```
Episode 500 | Reward: 32.8 | Steps: 180 | Lines: 8
Column heights: [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
Holes: 2 | Bumpiness: 12.5 | Max height: 12
  0123456789
 0 ··········  (0/10)
 1 ··········  (0/10)
...
15 █████████·  (9/10)
16 ██████████  (10/10)
17 ██████████  (10/10)
18 ██████████  (10/10)
19 ██████████  (10/10)
```

## 🔍 How to Use It

### During Training

The file is automatically created and updated as training progresses.

**View in real-time:**
```bash
# Follow the log file during training
tail -f logs/<experiment_name>/board_states.txt
```

**View specific episodes:**
```bash
# Search for a specific episode
grep "Episode 500" logs/<experiment_name>/board_states.txt -A 25
```

### Analyzing Results

**1. Check for center-stacking:**
Look for patterns like:
```
Column heights: [0, 0, 0, 19, 19, 19, 19, 0, 0, 0]  ❌ Bad!
Column heights: [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]   ✅ Good!
```

**2. Monitor learning progress:**
- Early episodes: Should show chaotic/unbalanced boards
- Middle episodes: Improving balance, fewer holes
- Late episodes: Good distribution across columns

**3. Diagnose problems:**
- High holes count → Agent creating gaps
- High bumpiness → Uneven stacking
- Max height 20 early → Dying too fast
- All columns 0-2 empty → Still center-stacking

## ⚙️ Configuration

**Control logging frequency:**
```bash
# Log every 10 episodes (default)
python train.py --log_freq 10

# Log every 50 episodes (less frequent, smaller file)
python train.py --log_freq 50

# Log every episode (very verbose!)
python train.py --log_freq 1
```

## 📈 What to Look For

### Good Signs (Learning!)

**Episode 100:**
```
Column heights: [0, 0, 0, 19, 19, 19, 19, 0, 0, 0]
Holes: 15
```

**Episode 1000:**
```
Column heights: [2, 5, 8, 12, 14, 13, 10, 7, 4, 2]
Holes: 3
```
↑ **Improvement!** Using more columns, fewer holes

### Bad Signs (Not Learning)

**Episode 100:**
```
Column heights: [0, 0, 0, 18, 19, 20, 19, 0, 0, 0]
Holes: 12
```

**Episode 1000:**
```
Column heights: [0, 0, 0, 18, 19, 20, 19, 0, 0, 0]
Holes: 14
```
↑ **No improvement!** Same pattern, similar holes

Possible causes:
- Reward shaping not working
- Epsilon too low (not exploring)
- Learning rate issues

## 🎯 Monitoring Anti-Center-Stacking

With the new reward shaping, you should see progression like:

**Episodes 0-200:**
```
Column heights: [0, 0, 0, 19, 19, 19, 19, 0, 0, 0]
Reward: -30 to -50
```

**Episodes 200-500:**
```
Column heights: [0, 1, 4, 15, 18, 17, 12, 6, 2, 0]
Reward: -10 to +10
```

**Episodes 500-1000:**
```
Column heights: [2, 4, 7, 12, 15, 14, 10, 6, 3, 1]
Reward: +10 to +30
```

**Episodes 1000+:**
```
Column heights: [5, 6, 8, 10, 12, 11, 9, 7, 5, 4]
Reward: +20 to +50
```

## 💡 Tips

1. **Compare episodes side-by-side:**
   ```bash
   # Extract episode 100 and 1000 for comparison
   grep "Episode 100 " board_states.txt -A 25 > ep100.txt
   grep "Episode 1000 " board_states.txt -A 25 > ep1000.txt
   diff ep100.txt ep1000.txt
   ```

2. **Count outer column usage:**
   ```bash
   # Check if columns 0-2 are being used
   grep "Column heights:" board_states.txt | grep -v "\[0, 0, 0,"
   ```

3. **Find best episode:**
   ```bash
   # Find episodes with highest rewards
   grep "Reward:" board_states.txt | sort -t: -k2 -n | tail -10
   ```

## 📊 Integration with Other Logs

This complements the existing CSV logs:
- `episode_log.csv` - Numerical metrics for plotting
- `board_states.txt` - Visual snapshots for debugging
- Console output - Real-time training progress

Together, they give you a complete picture of training!

---

## Summary

✅ **Added:** Visual board state logging
✅ **Location:** `logs/<experiment_name>/board_states.txt`
✅ **Frequency:** Every `--log_freq` episodes (default: 10)
✅ **Contents:** Metrics + visual 20×10 board representation
✅ **Purpose:** Monitor center-stacking, holes, and learning progress

Start training and check the file to see your agent's board states! 🚀
