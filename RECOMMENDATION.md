# Training Recommendation - November 6, 2025

## Current Situation

✅ **8-channel hybrid mode is CONFIRMED and working**
- Environment creates (20, 10, 8) observations
- Visual channels (0-3): Board, Active piece, Holder, Queue
- Feature channels (4-7): Holes, Heights, Bumpiness, Wells

❌ **Performance does NOT match expectations after 10K episodes**
- 0.21 lines/episode (target: 2-5+)
- First line clear: Episode 2,600 (expected: 200-500)
- Performance profile matches visual-only, not hybrid feature-based

## The Puzzle

We have a **paradox**:
1. ✅ 8-channel hybrid mode IS active
2. ✅ Feature channels ARE being computed correctly (tests passed)
3. ✅ Model CAN see current piece rotation (Channel 1: active_tetromino_mask)
4. ❌ But agent learns at visual-only speed, not feature-based speed

**Why?** The CNN may not be learning to **use** the feature channels effectively.

## Root Cause Analysis

### Problem: Generic CNN Architecture

The current model treats all 8 channels equally:
```python
# In src/model.py - Standard DQN
self.features = nn.Sequential(
    nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU()
)
```

This architecture:
- Mixes visual and feature channels immediately in first conv layer
- Doesn't distinguish between visual patterns (edges, shapes) and numeric features (hole counts, heights)
- May dilute the explicit feature signals with visual noise
- Requires the CNN to **learn** how to combine visual + feature data instead of being **told** how

### Why Visual-Only vs Feature-Based Are Different

**Visual-only (4 channels):**
- CNN must learn: "This L-shaped pattern at bottom = hole"
- CNN must learn: "Tall column of pixels = height problem"
- Takes 10,000s of episodes to learn these abstractions

**Feature-based (traditional):**
- Features are pre-computed: `holes = count_holes(board)`
- Features fed to fully-connected network, not CNN
- Network just learns: "If holes > 10, that's bad"
- Takes 100s-1,000s of episodes

**Our hybrid (8 channels with CNN):**
- Features are pre-computed: `holes_heatmap = compute_hole_heatmap(board)`
- But still fed through CNN that treats them like visual data
- CNN must learn: "Red pixels in feature channel = holes"
- May not be much faster than visual-only!

## Decision Point

You have **THREE options**:

### Option A: Continue Training (Monitor for 5K-10K more episodes) ⏱️

**When to choose this:**
- You have time/compute to spare
- You want to see if benefits emerge with more training
- You believe the CNN will eventually learn to use features

**Action:**
```bash
# Continue current resume training
# Monitor every 1,000 episodes for improvement

# Success criteria by episode 15K:
# - Lines/episode > 0.5 (2x current)
# - Holes decreasing trend
# - First multi-line clear (2-4 lines at once)

# Success criteria by episode 20K:
# - Lines/episode > 1.0
# - Consistent line clears
# - Average holes < 30
```

**Pros:**
- Simplest approach
- May work with enough time
- No code changes needed

**Cons:**
- Could waste 10+ hours of compute
- May never reach feature-based performance
- Opportunity cost of not trying better architecture

**My assessment:** 🟡 **20% chance of significant improvement**

---

### Option B: Architectural Fix - Dual-Branch Network 🏗️ **RECOMMENDED**

**When to choose this:**
- You want to maximize learning speed
- You want to properly leverage feature channels
- You're willing to restart training with better architecture

**The Fix:**

Create a dual-branch architecture that processes visual and feature channels separately:

```python
class HybridDQN(nn.Module):
    """Dual-branch DQN: Visual CNN + Feature CNN → Combined"""

    def __init__(self, input_channels=8, n_actions=8):
        super().__init__()

        # Branch 1: Visual CNN (channels 0-3)
        # Processes: Board, Active piece, Holder, Queue
        self.visual_branch = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 2: Feature CNN (channels 4-7)
        # Processes: Holes, Heights, Bumpiness, Wells
        # Simpler architecture - features already meaningful
        self.feature_branch = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate combined feature size
        # Visual: 64 * 10 * 5 = 3200
        # Feature: 32 * 10 * 5 = 1600
        # Total: 4800

        # Combined processing
        self.fc = nn.Sequential(
            nn.Linear(4800, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # Split input into visual and feature channels
        visual = x[:, :4, :, :]      # Channels 0-3
        features = x[:, 4:, :, :]    # Channels 4-7

        # Process separately
        visual_out = self.visual_branch(visual)
        feature_out = self.feature_branch(features)

        # Flatten and concatenate
        visual_flat = visual_out.view(visual_out.size(0), -1)
        feature_flat = feature_out.view(feature_out.size(0), -1)
        combined = torch.cat([visual_flat, feature_flat], dim=1)

        # Final Q-values
        return self.fc(combined)
```

**Implementation steps:**
1. Create `src/model_hybrid.py` with HybridDQN class
2. Modify `src/agent.py` to support `model_type='hybrid_dqn'`
3. Update `train_progressive_improved.py` to add `--model_type hybrid_dqn` option
4. Train from scratch with new architecture

**Expected results:**
- Episodes 0-500: Similar to current (foundation learning)
- Episodes 500-2000: **MUCH faster hole reduction** (feature branch kicks in)
- Episodes 1000-2000: First consistent line clears
- Episodes 2000-5000: 2-5 lines/episode
- Episodes 5000-10000: 5-10 lines/episode

**Pros:**
- Proper utilization of feature channels
- Likely 5-10x faster learning
- Matches research expectations for hybrid approach

**Cons:**
- Requires code changes
- Need to restart training from episode 0
- Slightly more complex model

**My assessment:** 🟢 **70% chance of significant improvement**

---

### Option C: Hybrid Approach - Continue + Prepare Architecture Fix 🔀

**When to choose this:**
- You're unsure which approach to take
- You want to hedge your bets
- You have time to try both sequentially

**Action:**
```bash
# 1. Let current resume training continue to 15,000 episodes
#    (Check progress at 12K, 13K, 14K, 15K)

# 2. Meanwhile, implement dual-branch architecture
#    (Parallel work, can be done while training runs)

# 3. At episode 15,000, evaluate:
#    If lines/ep > 0.8: Continue to 20K
#    If lines/ep < 0.8: Stop and switch to dual-branch architecture

# 4. If switching to dual-branch, use learnings from 15K run:
#    - Which curriculum stages worked well
#    - Which reward weights were effective
#    - What epsilon decay schedule was good
```

**Pros:**
- Don't waste current training run
- Prepare backup plan
- Make data-driven decision

**Cons:**
- Takes longer total time
- May end up discarding 15K episode run anyway

**My assessment:** 🟡 **40% chance of success without architecture change**

---

## My Recommendation: **Option B** (Dual-Branch Architecture) 🏗️

### Why?

1. **Root cause identified**: Generic CNN doesn't distinguish visual vs feature data
2. **Research supports it**: Feature-based approaches need feature-aware architectures
3. **Time efficiency**: 5-10K episodes with good architecture > 30K episodes with poor architecture
4. **Learning opportunity**: You'll understand why architecture matters for hybrid approaches

### Implementation Timeline

**Time investment:** 2-3 hours
- 1 hour: Implement `HybridDQN` class
- 30 mins: Modify agent to support new model type
- 30 mins: Test with 10-episode run
- 30 mins: Start full training run

**Expected training time (10,000 episodes):**
- Same as before: ~7 hours
- But with MUCH better results

### Confidence Level

Based on:
- ✅ 8-channel mode working correctly
- ✅ Feature computation verified
- ✅ Research showing feature-based approaches work
- ❌ Generic CNN not optimized for hybrid data

**I'm 70% confident** dual-branch architecture will achieve:
- First line clears by episode 500-1000
- 1-2 lines/episode by episode 5000
- 3-5 lines/episode by episode 10000

This matches the research expectations for hybrid feature-based approaches.

---

## Quick Decision Matrix

| Scenario | Choose Option A | Choose Option B | Choose Option C |
|----------|----------------|-----------------|-----------------|
| Limited time, want quick results | ❌ | ✅ | ❌ |
| Unlimited time/compute | ✅ | ❌ | ✅ |
| Want to learn/experiment | ❌ | ✅ | ✅ |
| Risk-averse (try everything) | ❌ | ❌ | ✅ |
| Confident in current approach | ✅ | ❌ | ❌ |
| Skeptical of current approach | ❌ | ✅ | ✅ |

---

## If You Resume Training (Option A), Watch For:

### Good Signs (Continue)
- ✅ Lines/episode increasing trend (even if slow)
- ✅ Average holes decreasing below 40
- ✅ First 2-line or 3-line clear
- ✅ Reward variance decreasing (more consistent play)
- ✅ Epsilon below 0.01 (exploitation mode)

### Bad Signs (Stop and Switch to Option B)
- ❌ Lines/episode flat or decreasing
- ❌ Holes staying above 45
- ❌ Only single-line clears after 15K episodes
- ❌ Reward still highly variable
- ❌ Agent making obviously bad moves (placing pieces with holes)

---

## Next Steps Based on Your Choice

### If Option A (Continue Current Training):
```bash
# Monitor training every 1000 episodes
tail -f logs/improved_*/DEBUG_SUMMARY.txt

# At episode 12K, 13K, 14K, 15K, check:
# - Lines/episode trend
# - Holes trend
# - Reward trend
# - Board quality (visualize final boards)

# Kill training if no improvement by 15K:
pkill -f train_progressive
```

### If Option B (Implement Dual-Branch):
```bash
# 1. Create new model architecture
# I can help implement this!

# 2. Test with short run
python train_progressive_improved.py \
    --episodes 100 \
    --force_fresh \
    --model_type hybrid_dqn \
    --experiment_name "hybrid_test"

# 3. Verify it learns faster
# Compare logs/hybrid_test/ vs logs/improved_20251104_224000/
# Should see first line clears MUCH earlier

# 4. Full training run
python train_progressive_improved.py \
    --episodes 10000 \
    --force_fresh \
    --model_type hybrid_dqn \
    --experiment_name "hybrid_10k"
```

### If Option C (Hybrid Approach):
```bash
# 1. Let resume training continue
# (No action needed if already running)

# 2. Start implementing dual-branch architecture in parallel
# (I can help with this)

# 3. Decision point at episode 15,000:
#    Evaluate results and choose A or B for next phase
```

---

## My Call: **Implement Dual-Branch Architecture (Option B)**

**Bottom line:** The generic CNN is like giving someone a calculator but making them use it as a hammer. The dual-branch architecture uses the calculator for math and the hammer for nails.

Your feature channels are great - they're just not being used effectively. Fix the architecture, and you'll likely see the 10-50x speedup that research promises.

**Want me to implement it?** I can create the HybridDQN class and integrate it into your training pipeline. Estimated time: 2-3 hours including testing.

---

*Recommendation generated: 2025-11-06*
*Based on: 10K episode training analysis + 8-channel verification*
