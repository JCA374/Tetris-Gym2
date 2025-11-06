# Agent Decision-Making: Information Analysis

**Date**: 2025-11-05
**Critical Analysis**: Does the agent have the information it needs to make informed decisions?

## 🔍 Two-Layer Information System

### Layer 1: What the Agent SEES (Observation)
**Pure visual input - Image-based (20, 10, 4)**

The agent receives a 4-channel "image" at each step:
- **Channel 0**: Board state (locked pieces) - binary (0 or 1)
- **Channel 1**: Active tetromino (current falling piece) - binary
- **Channel 2**: Holder (held piece for swap) - binary (top-left 4×4)
- **Channel 3**: Queue (next pieces preview) - binary (top-right)

**CRITICAL**: This is PURELY VISUAL. No numerical metrics!

### Layer 2: What the Agent LEARNS FROM (Reward Shaping)
**Rich metrics calculated every step**

The reward shaper calculates these metrics **every step** from the observation:
- Holes (count of trapped empty cells)
- Column heights (height of each column)
- Bumpiness (height variation)
- Aggregate height (total of all column heights)
- Wells (depth of valleys)
- Spread (horizontal distribution)
- Completable rows (8-9 filled, no holes)
- Clean rows (contiguous filled cells, no holes)
- Center stacking penalty
- Outer columns unused

**CRITICAL**: Agent gets reward signal based on these, but doesn't SEE them directly!

## ❌ The Critical Gap: Implicit vs Explicit Information

### What the Agent Must INFER from Visual Input

| Metric | Visible in Observation? | How Agent Sees It | Difficulty |
|--------|------------------------|-------------------|----------|
| **Holes** | ❌ Implicit only | Empty cells with filled cells above (spatial pattern) | HARD |
| **Column heights** | ❌ Implicit only | Visual height of each column | MEDIUM |
| **Bumpiness** | ❌ Implicit only | Height differences between adjacent columns | HARD |
| **Completable rows** | ❌ Implicit only | Rows that are almost full | VERY HARD |
| **Clean rows** | ❌ Implicit only | Rows without gaps | HARD |
| **Spread** | ❌ Implicit only | How evenly pieces are distributed | VERY HARD |
| **Board state** | ✅ Explicit | Channel 0 shows all locked pieces | EASY |
| **Current piece** | ✅ Explicit | Channel 1 shows falling piece | EASY |
| **Next pieces** | ✅ Explicit | Channel 3 shows queue | EASY |

### The Learning Challenge

**Example: Holes**

The agent experiences this:
1. **Visual**: Sees board with empty cell at (15, 5), filled cell at (14, 5)
2. **Reward**: Gets -1.25 per hole penalty (or more depending on stage)
3. **Must Learn**: "This spatial pattern (empty below filled) = bad reward"

The agent must:
- Learn to recognize holes from VISUAL patterns
- Associate those patterns with negative rewards
- Develop internal representation of "hole" concept
- Generalize across different board configurations

**This is HARD!** Much harder than seeing explicit "hole_count: 5" in observation.

## 🔄 Reward Feedback: Good News!

### Reward Shaping IS Applied Every Step

**Good**: The agent gets immediate feedback every step!

From `train_progressive_improved.py:685`:
```python
# Every step during episode:
shaped_reward = reward_shaper.calculate_reward(obs, action, raw_reward, done, info)
```

From `src/progressive_reward_improved.py:108-109`:
```python
board = extract_board_from_obs(obs)
metrics = self.calculate_metrics(board, info)  # Calculates holes, bumpiness, etc.
```

**This means**:
- ✅ Agent gets penalized for holes IMMEDIATELY (every step)
- ✅ Agent gets rewarded for clean rows IMMEDIATELY
- ✅ Agent gets rewarded for spreading IMMEDIATELY
- ✅ Feedback is instant, not delayed until game-over

**However**: Agent must LEARN the connection between:
- Visual patterns (what it sees)
- Reward signals (what it experiences)
- Actions (what it controls)

## 📊 Metrics Only Logged at End vs Used During Play

### ✅ FIXED (Nov 2025)

| Metric | Used in Reward (Per-Step) | Logged to CSV | Displayed to User |
|--------|---------------------------|---------------|-------------------|
| **Holes** | ✅ Every step | ✅ avg/min/final | ✅ Every log_freq |
| **Bumpiness** | ✅ Every step | ✅ avg/final | ✅ Every log_freq |
| **Completable rows** | ✅ Every step | ✅ avg/final | ✅ Every log_freq |
| **Clean rows** | ✅ Every step | ✅ avg/final | ✅ Every log_freq |
| **Max height** | ✅ Every step | ✅ avg/final | ✅ Every log_freq |
| **Column heights** | ✅ Every step | ✅ final | ✅ Every log_freq |
| **Spread** | ✅ Every step | ❌ Not logged | ❌ Not logged |

### The Fix

**Now tracking metrics throughout episodes!**

Metrics are sampled every 20 steps during play:
- **Average during play**: Shows typical board quality while playing
- **Final at game-over**: Shows end state (for reference)
- **Both logged separately**: Can see trajectory and progression

Example: Clean rows
- Agent gets +5 reward per clean row every step (in stage 4)
- We now see average clean rows during play (e.g., 12)
- AND final clean rows at game-over (e.g., 3)
- Shows agent maintains quality during play, degrades at end

## 🎯 Recommendations

### 1. ❌ LOW PRIORITY: Add Explicit Metrics to Observation

**Why low priority**: Agent is learning from visual patterns, which is what we want for generalization.

**Possible enhancement** (if learning is too slow):
```python
# Could add scalar channels with normalized metrics
# Channel 4: Hole density map (0-1 per cell indicating nearby holes)
# Channel 5: Height map (normalized column heights)
```

**Pros**: Faster learning, explicit information
**Cons**: Less generalizable, may become reliant on hand-crafted features

### 2. ✅ IMPLEMENTED: Track More Metrics During Play

**Status**: ✅ DONE (Nov 2025)!

**Now tracking during play** (sampled every 20 steps):
- ✅ Average holes (and min, final)
- ✅ Average bumpiness (and final)
- ✅ Average completable rows (and final)
- ✅ Average clean rows (and final)
- ✅ Average max height (and final)

**Benefits achieved**:
- ✅ See if agent maintains quality during play
- ✅ Understand trajectory, not just end state
- ✅ Better debugging when training stalls
- ✅ More realistic performance expectations

### 3. ✅ VERIFIED: Agent Gets Immediate Feedback

**Status**: ✅ Confirmed working!

The code shows:
- Reward shaper called every step ✅
- Metrics calculated from current observation ✅
- Shaped reward includes penalties/bonuses ✅
- Agent stores (state, action, reward, next_state) ✅

**The disconnect was only in LOGGING, not in LEARNING!**
Now fixed - we can see what the agent actually experiences.

## 🧠 How the Agent Actually Learns

### The CNN's Job

The Convolutional Neural Network must learn to:
1. **Recognize patterns**: "This shape in channel 0 = hole"
2. **Associate with value**: "Holes → lower Q-value"
3. **Predict consequences**: "This action → more holes → bad"
4. **Generalize**: "Holes are bad regardless of position"

### Why Visual-Only Is Actually Good

**Advantages**:
- Forces learning of spatial relationships
- Generalizes to different board configurations
- Doesn't overfit to specific metric ranges
- Learns "what" not just "how much"

**Example**: Agent learns holes are bad through spatial patterns, not just a number. This means it can recognize hole-creating moves before making them.

## 📈 What We Should Track Next

### Episode Progression Metrics

Instead of just logging at end, sample every 20 steps:

```python
# Already done for holes:
hole_samples = []
if episode_steps % 20 == 0:
    hole_samples.append(count_holes(board))

# Should add for:
bumpiness_samples = []
completable_samples = []
clean_rows_samples = []
max_height_samples = []
```

### Per-Step Reward Components

Track cumulative reward components:
```python
# Already tracked!
episode_component_totals = {
    'hole_penalty': -150.2,
    'completable_bonus': +25.0,
    'survival_bonus': +40.0,
    ...
}
```

This shows what's driving the reward signal.

## 🎓 Key Findings

### What's Working ✅

1. **Agent gets immediate feedback every step**
2. **Reward shaping calculates all metrics per-step**
3. **Visual observation provides spatial context**
4. **4-channel input gives board + piece + holder + queue**

### What Was Misleading ❌

1. **Logging only final states** (partially fixed for holes)
2. **Other metrics still only logged at game-over**
3. **Can't see metric trajectories during episodes**

### What Could Be Better 🔧

1. **Track more metrics during play** (like we did for holes)
2. **Log intermediate values** (every 20 steps)
3. **Show metric progression** in debugging output
4. **Consider adding explicit features** if learning too slow

## 🚀 Recommended Next Steps

### ✅ Completed (High Impact)
1. ✅ **DONE**: Track holes during play
2. ✅ **DONE**: Track bumpiness, completable_rows, clean_rows during play
3. ✅ **DONE**: Added during-play vs final metrics to logging
4. ✅ **DONE**: Updated console output to show play→final transitions

### Medium Priority (Optional Enhancements)
1. Analyze if CNN is learning to recognize holes (activation maps)
2. Test if explicit hole-count input speeds learning
3. Visualize metric trajectories over time (plots)
4. Add heatmaps showing where agent places pieces

### Low Priority (Research Extensions)
1. Consider auxiliary tasks (hole prediction from observation)
2. Add attention mechanisms to highlight critical regions
3. Experiment with additional input channels
4. Compare visual-only vs visual+metrics approaches

## 💡 Bottom Line

**Good News**: The agent IS getting the information it needs through:
- Visual observation (spatial patterns)
- Immediate reward feedback (every step)
- Rich metrics in reward signal

**The Issue**: We weren't SEEING the agent's actual performance during play, only at game-over.

**The Fix**: Track metrics during play (started with holes, should extend to others).

The agent must learn to infer complex concepts (holes, bumpiness) from visual patterns, which is hard but ultimately better for generalization. The reward signal is teaching it correctly; we just need better visibility into the learning process.
