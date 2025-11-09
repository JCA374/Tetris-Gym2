# Feature Vector DQN Logging & Analysis Guide

## Overview

The feature vector training system includes comprehensive logging and analysis tools to track all metrics needed for proper evaluation of training sessions.

## What's Logged

### Per-Episode Metrics (episode_log.csv)

Every episode logs the following metrics:

**Performance Metrics:**
- `episode`: Episode number
- `reward`: Total shaped reward for episode
- `steps`: Number of steps before game over
- `lines_cleared`: Number of lines cleared in episode
- `epsilon`: Current exploration rate
- `memory_size`: Size of replay buffer

**Feature Metrics (Final State):**
- `aggregate_height`: Sum of all column heights (normalized 0-1)
- `holes`: Number of holes in final board state (normalized 0-1)
- `bumpiness`: Sum of height differences between adjacent columns (normalized 0-1)
- `wells`: Depth of valleys between columns (normalized 0-1)
- `max_height`: Maximum column height (normalized 0-1)
- `min_height`: Minimum column height (normalized 0-1)
- `std_height`: Standard deviation of column heights (normalized 0-1)

**Metadata:**
- `timestamp`: ISO 8601 timestamp of episode completion

### Automatic Plot Generation

Training automatically generates plots:

1. **reward_progress.png**: Reward over time with moving average
2. **training_metrics.png**: Epsilon and steps per episode
3. **training_analysis.png**: 6-panel comprehensive analysis (via analyze_training.py)

### Checkpoint Saving

Checkpoints saved every 500 episodes (configurable with `--save_freq`):
- `checkpoint_ep<N>.pth`: Full checkpoint with model, optimizer, epsilon, best score
- `best_model.pth`: Best performing model (highest lines cleared)
- `final_model.pth`: Final model at end of training

## Log Directory Structure

```
logs/
└── feature_vector_fc_dqn_20251109_151316/         # Experiment directory
    ├── feature_vector_fc_dqn_20251109_151316/     # Nested (TrainingLogger structure)
    │   ├── episode_log.csv                        # All episode metrics
    │   ├── board_states.txt                       # Final board visualizations
    │   ├── reward_progress.png                    # Reward curve
    │   └── training_metrics.png                   # Epsilon/steps curves
    └── training_analysis.png                      # Comprehensive analysis
```

## Using the Analysis Script

### Basic Usage

```bash
# Analyze a training run
python analyze_training.py logs/feature_vector_fc_dqn_20251109_151316

# Custom moving average window
python analyze_training.py logs/feature_vector_fc_dqn_20251109_151316 --window 50
```

### Analysis Output

The script provides:

**1. Overall Statistics:**
- Total episodes trained
- Mean ± std reward
- Mean ± std steps per episode
- Mean ± std lines cleared
- Maximum lines achieved
- Epsilon range

**2. Recent Performance:**
- Statistics for last 100 episodes (shows recent learning)
- Useful for identifying if training has converged

**3. Feature Metrics:**
- Average final state quality metrics
- Helps identify if agent is learning good board management

**4. Learning Milestones:**
- When agent first achieved 1, 5, 10, 20, 50+ lines
- How many times each milestone was achieved
- Tracks progression through learning stages

**5. Training Curves:**
Six comprehensive plots:
- Reward over time (raw + moving average)
- Steps per episode (survival time)
- Lines cleared progression
- Epsilon decay curve
- Holes in final state (board quality)
- Aggregate height progression

## Training Commands

### Quick Test (100 episodes, ~1 minute)

```bash
python train_feature_vector.py --episodes 100 --log_freq 10
```

### Short Training (1,000 episodes, ~10 minutes)

```bash
python train_feature_vector.py --episodes 1000 --log_freq 50
```

### Full Training (5,000 episodes, ~5 hours)

```bash
python train_feature_vector.py \
    --episodes 5000 \
    --model_type fc_dqn \
    --log_freq 50 \
    --save_freq 500 \
    --experiment_name my_experiment
```

### Extended Training (10,000+ episodes)

```bash
python train_feature_vector.py \
    --episodes 10000 \
    --model_type fc_dueling_dqn \
    --lr 1e-4 \
    --batch_size 64 \
    --epsilon_decay 0.9995 \
    --experiment_name dueling_10k
```

## Monitoring During Training

### Real-Time Console Output

Training prints progress every N episodes (set with `--log_freq`):

```
Episode 1000/5000 | Steps: 245 | Lines: 12 | Reward: 1194.5 | Epsilon: 0.951 | Best: 18 lines | Speed: 17.5 ep/s
```

This shows:
- Current episode / total episodes
- Steps survived this episode
- Lines cleared this episode
- Total shaped reward
- Current exploration rate
- Best performance so far
- Training speed (episodes per second)

### Periodic Checkpoint Messages

Every 500 episodes (default):

```
   💾 Checkpoint saved: models/checkpoint_ep1000.pth
   📊 Logs and plots updated
```

### Graceful Interruption (Ctrl+C)

If you interrupt training with Ctrl+C:

```
⚠️  Training interrupted by user. Saving progress...
📊 Saving logs and plots...
✅ Logs saved successfully
```

All progress is automatically saved before exit.

## Evaluating Training Success

### Expected Progress for 5,000 Episodes

Based on research of successful implementations:

| Episodes | Expected Lines/Episode | Expected Steps | Key Milestone |
|----------|------------------------|----------------|---------------|
| 0-500    | 0-1                    | 50-150         | Learning survival |
| 500-1000 | 1-5                    | 150-250        | Basic line clearing |
| 1000-2000 | 5-20                  | 250-400        | Consistent clearing |
| 2000-5000 | 20-100                | 400-800        | Advanced strategy |
| 5000+    | 100-1000+             | 800-2000+      | Expert performance |

### Key Metrics to Watch

1. **Lines Cleared (Primary Metric)**
   - Should steadily increase over episodes
   - First line clear: Expected by episode 200-500
   - 10+ lines: Expected by episode 1000-2000

2. **Reward Progression**
   - Should show upward trend (less negative → positive)
   - Moving average should smooth out and increase

3. **Steps per Episode (Survival)**
   - Should increase as agent learns to survive longer
   - Longer survival → more opportunities for line clears

4. **Feature Metrics**
   - **Holes**: Should decrease (less holes = better board quality)
   - **Bumpiness**: Should decrease (smoother surface)
   - **Max height**: Should stabilize (not topping out immediately)

5. **Epsilon Decay**
   - Should decrease smoothly from 1.0 toward 0.05
   - As epsilon drops, agent relies more on learned policy

### Warning Signs

❌ **Reward stuck or decreasing:**
- Check if epsilon is decaying properly
- May need longer training
- Consider adjusting learning rate

❌ **Steps not increasing:**
- Agent isn't learning survival
- Check reward function
- May need different exploration strategy

❌ **Lines cleared always 0 after 1000+ episodes:**
- Training may have stagnated
- Try different hyperparameters
- Check if reward function properly incentivizes line clears

## Comparing Experiments

To compare different training runs:

```bash
# Analyze each experiment
python analyze_training.py logs/experiment_A
python analyze_training.py logs/experiment_B

# Compare the generated CSV files
import pandas as pd

df_a = pd.read_csv("logs/experiment_A/.../episode_log.csv")
df_b = pd.read_csv("logs/experiment_B/.../episode_log.csv")

# Compare mean performance in last 100 episodes
print("Experiment A:", df_a.tail(100)['lines_cleared'].mean())
print("Experiment B:", df_b.tail(100)['lines_cleared'].mean())
```

## Advanced: Custom Analysis

The CSV files can be loaded into pandas for custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load episode log
df = pd.read_csv("logs/<experiment>/episode_log.csv")

# Custom analysis
window = 50
df['lines_ma'] = df['lines_cleared'].rolling(window).mean()
df['reward_ma'] = df['reward'].rolling(window).mean()

# Custom plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(df['episode'], df['lines_ma'])
ax1.set_title('Lines Cleared (50-episode MA)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Lines')

ax2.plot(df['episode'], df['holes'])
ax2.set_title('Holes in Final State')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Holes (normalized)')

plt.tight_layout()
plt.savefig('custom_analysis.png')
```

## Logging Implementation Details

### Where Logging Happens

1. **train_feature_vector.py** (lines 228-248):
   - Calls `logger.log_episode()` after each episode
   - Extracts feature metrics from final state
   - Passes all metrics to logger

2. **src/utils.py** (`TrainingLogger` class):
   - Stores episode metrics in memory
   - Writes to CSV on `save_logs()` call
   - Generates plots on `plot_progress()` call

3. **Automatic Saving**:
   - Every 500 episodes (during training)
   - On training completion
   - On Ctrl+C interruption

### Adding Custom Metrics

To log additional metrics, modify `train_feature_vector.py`:

```python
# In the training loop, after episode ends:
logger.log_episode(
    episode=episode + 1,
    reward=total_reward,
    steps=steps,
    epsilon=agent.epsilon,
    lines_cleared=lines_cleared,
    # ... existing metrics ...
    custom_metric=my_custom_value  # Add your metric here
)
```

The metric will automatically appear in the CSV file.

## Summary

✅ **Complete logging of all metrics**: Performance, features, metadata
✅ **Automatic plot generation**: Reward curves, feature progressions
✅ **Periodic checkpoint saving**: Every 500 episodes + best/final models
✅ **Comprehensive analysis script**: Statistics, milestones, 6-panel plots
✅ **Graceful interruption handling**: Ctrl+C saves everything
✅ **Extensible**: Easy to add custom metrics and analysis

**You have everything needed to properly evaluate training sessions!**

For questions about specific metrics or analysis needs, check the code or ask for clarification.
