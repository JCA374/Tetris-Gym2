# Tetris DQN Project History

**Project**: Deep Q-Network for Tetris using Gymnasium
**Timeline**: November 2025
**Current Status**: Hybrid Dual-Branch DQN training in progress

---

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Critical Fixes Applied](#critical-fixes-applied)
3. [Hole Measurement Discovery](#hole-measurement-discovery)
4. [Research Phase](#research-phase)
5. [Feature Channel Implementation](#feature-channel-implementation)
6. [First Training Run (10K Episodes)](#first-training-run-10k-episodes)
7. [Root Cause Analysis](#root-cause-analysis)
8. [Hybrid Architecture Implementation](#hybrid-architecture-implementation)
9. [Current Training](#current-training)
10. [Key Learnings](#key-learnings)

---

## Initial Setup

**Goal**: Train DQN agent to play Tetris

**Architecture**:
- Environment: Tetris Gymnasium v0.3.0
- Observation: (20, 10, 4) - 4-channel vision
  - Channel 0: Board state
  - Channel 1: Active tetromino
  - Channel 2: Holder
  - Channel 3: Queue
- Model: Standard CNN-based DQN
- Reward: Progressive curriculum (5 stages)

**Initial Results**: Agent learned survival but struggled with line clears

---

## Critical Fixes Applied

### Issue 1: Dropout Rate Too High
**Problem**: Dropout=0.3 (typical for classification) too high for RL
**Fix**: Reduced to 0.1 in all model layers
**Impact**: More stable learning
**File**: `src/model.py`

### Issue 2: Train/Eval Mode Not Set
**Problem**:
- `model.train()` never called → dropout active during learning
- `model.eval()` never called → dropout active during inference
- 30% of neurons randomly off even during gameplay!

**Fix**:
- Added `model.train()` in `agent.learn()` (line 309)
- Added `model.eval()` in `agent.select_action()` (line 224)

**Impact**: MAJOR - proper dropout behavior
**File**: `src/agent.py`

**Reference**: `reports/CRITICAL_FIXES_APPLIED.md`

---

## Hole Measurement Discovery

**Date**: Early November 2025

**Problem Found**:
User asked: "Are we measuring holes wrong? Goal of <15 holes - is that possible? At the end it will always be high."

**Investigation**:
- Holes were ONLY measured at game-over (worst possible moment)
- Example: "48 holes at step 204" = board filled up, not representative
- Agent was actually playing better than metrics showed!

**Fix**:
- Track holes during play (sample every 20 steps)
- Added metrics:
  - `holes_avg`: Average during play (PRIMARY METRIC)
  - `holes_min`: Best board state achieved
  - `holes_final`: At game-over (for reference only)
  - `holes_at_step_X`: Consistent checkpoints

**New Goals**:
- Avg holes during play: <15 (achievable)
- Final holes at game-over: <30-50 (depends on line clears)

**Extended Fix**:
Applied same sampling to ALL metrics:
- Bumpiness
- Completable rows
- Clean rows
- Max height

**Impact**: Metrics now accurately reflect agent performance during play

**Reference**: `reports/HOLE_MEASUREMENT_FIX.md`

---

## Research Phase

**Trigger**: Agent achieving 0.21 lines/episode after 10K episodes seemed too slow

**Research Questions**:
1. How do other DQN Tetris implementations work?
2. What feedback do they provide to the model?
3. Are we missing something critical?

**Method**: Web research on DQN Tetris state representations

**Key Findings**:

### Visual-Only Approaches
- Use raw pixels or board representation only
- Agent must learn spatial relationships from scratch
- Performance: 0-50 lines/episode even after 75K episodes
- Very slow learning curve

### Feature-Based Approaches (90% of successful implementations)
- Provide explicit features: holes, heights, bumpiness, wells, etc.
- Features fed directly to fully-connected network (not CNN)
- Performance: 200-500 lines/episode in just 1K-5K episodes
- **10-50x faster learning**

**Conclusion**: We're visual-only, need to add explicit features

**Reference**: `reports/DQN_RESEARCH_ANALYSIS.md`

---

## Feature Channel Implementation

**Decision**: Hybrid approach - visual channels + explicit feature heatmaps

**Plan**:
Extend observation from (20, 10, 4) to (20, 10, 8)
- Channels 0-3: Visual (board, active, holder, queue)
- Channels 4-7: Feature heatmaps (holes, heights, bumpiness, wells)

**Phase 1 Implementation**:

### Step 1: Feature Heatmap Functions
**File**: `src/feature_heatmaps.py`

Functions created:
- `compute_hole_heatmap()` - Spatial map of holes
- `compute_height_map()` - Normalized column heights
- `compute_bumpiness_map()` - Height variation
- `compute_well_map()` - Valleys between columns

**Tests**: `tests/test_feature_heatmaps.py` - 14 unit tests, all passed

### Step 2: Environment Integration
**File**: `config.py` - Modified `CompleteVisionWrapper`

Added:
- `use_feature_channels` parameter (default: True)
- 8-channel observation space (20, 10, 8)
- Feature computation in `observation()` method
- Changed dtype from uint8 to float32 for normalization

### Step 3: Training Script Updates
**File**: `train_progressive_improved.py`

Added flags:
- `--use_feature_channels` (enable 8ch mode)
- `--no_feature_channels` (4ch visual-only mode)
- Console output showing channel count confirmation

### Step 4: Visualization
**File**: `visualize_features.py`

Tool to visualize all 8 channels and analyze distributions

### Step 5: Integration Tests
**File**: `tests/test_feature_channels_training.py`

5 tests:
- 4-channel mode
- 8-channel mode
- Agent compatibility
- Training loop
- Feature data verification

All tests passed! ✅

**Expected Results**: 10-50x faster learning based on research

**Reference**: `reports/IMPLEMENTATION_PLAN.md`

---

## First Training Run (10K Episodes)

**Date**: November 4-5, 2025
**Run ID**: `improved_20251104_224000`
**Episodes**: 10,000
**Duration**: 6.91 hours
**Model**: Standard DQN (generic CNN)

**Configuration**:
- 8-channel mode: CONFIRMED ✅
- Feature channels: Computed correctly ✅
- Model: Standard DQN with generic CNN
- Epsilon decay: 0.9995
- Learning rate: 5e-4

**Results**:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Lines/episode | 0.21 | ≥2.0 | ❌ FAILED |
| First line clear | Episode 2,600 | <1,000 | ❌ FAILED |
| Holes (avg) | 48.2 | <15 | ❌ FAILED |
| Columns used | 10/10 | ≥8 | ✅ SUCCESS |
| Steps/episode | 311.7 | 300-500 | ✅ OK |

**Key Milestones**:
- Episode 100: 0 lines
- Episode 500: 0 lines
- Episode 2,600: **First line clear**
- Episode 10,000: 0.21 lines/episode

**Observation**: Performance matched visual-only profile, NOT feature-based!

**Reference**: `reports/TRAINING_ANALYSIS_10K.md`

---

## Root Cause Analysis

**The Paradox**:
- ✅ 8-channel mode confirmed working
- ✅ Feature channels computed correctly
- ✅ Model can see current piece rotation
- ❌ But learning speed = visual-only (not 10-50x faster)

**Investigation**: Why no speedup from explicit features?

**Discovery**: Generic CNN Architecture Problem

```python
# What we had:
class DQN:
    def __init__(self, input_channels=8):
        self.conv1 = nn.Conv2d(8, 32, ...)  # ALL 8 channels mixed!
        self.conv2 = nn.Conv2d(32, 64, ...)
        self.conv3 = nn.Conv2d(64, 64, ...)
```

**The Problem**:
- Generic CNN treats ALL channels the same
- Visual data (edges, shapes) mixed with feature data (numeric values)
- First conv layer mixes channels 0-7 immediately
- Feature signals (holes=15, height=12) diluted with visual noise
- CNN must learn: "These pixel values in channel 4 = holes"
- Nearly as hard as learning from visual-only!

**Why Feature-Based Approaches Work**:
Traditional feature-based approaches use:
```python
features = [holes, heights, bumpiness, wells, ...]  # Explicit values
network = FullyConnected(features)  # Direct processing
```

Not:
```python
features_as_heatmaps = spatial_distribution(features)  # Heatmaps
network = CNN(features_as_heatmaps)  # Treats as images
```

**Conclusion**:
We have explicit features but the CNN architecture doesn't leverage them properly. It treats pre-computed numeric features like they're visual patterns that need edge detection.

**Three Options Considered**:

**Option A**: Continue training (20% success)
- Maybe CNN learns eventually (10K-20K more episodes)
- Pros: No work needed
- Cons: May waste 10+ hours

**Option B**: Dual-branch architecture (70% success) ⭐ **CHOSEN**
- Separate visual and feature processing
- Visual CNN for spatial patterns
- Feature CNN for numeric distributions
- Late fusion before FC layers
- Pros: Proper utilization of features
- Cons: Need to restart training

**Option C**: Hybrid approach (40% success)
- Continue to 15K, implement B in parallel
- Decision point at 15K
- Pros: Hedge bets
- Cons: Takes longer

**Reference**: `reports/RECOMMENDATION.md`

---

## Hybrid Architecture Implementation

**Date**: November 6, 2025
**Decision**: Implement Option B - Dual-Branch Architecture

### Architecture Design

```
Input (20×10×8)
    ↓
    ├─→ Visual CNN (channels 0-3) ──→ 3,200 features
    │   Conv2d(4→32→64→64)
    │   Optimized for spatial patterns
    │   (Board, Active piece, Holder, Queue)
    │
    └─→ Feature CNN (channels 4-7) ──→ 1,600 features
        Conv2d(4→16→32)
        Simpler architecture
        (Holes, Heights, Bumpiness, Wells)
            ↓
        Concatenate (4,800 features)
            ↓
        FC: 4800→512→256→8 Q-values
```

**Key Innovations**:
1. **Separate processing** - Visual and feature branches process independently
2. **Feature branch simpler** - Features already meaningful, don't need complex CNN
3. **Late fusion** - Concatenate only after feature extraction
4. **Architecture-aware** - Each branch optimized for its data type

### Files Created

**1. src/model_hybrid.py** (367 lines)
- `HybridDQN` - Dual-branch standard DQN
- `HybridDuelingDQN` - Dual-branch + dueling streams

**2. test_hybrid_model.py** (259 lines)
- Comprehensive test suite
- Model creation and forward pass
- 10-episode training loop
- Gradient flow verification
- Channel separation test

**3. HYBRID_DQN_GUIDE.md** (400+ lines)
- Complete usage guide
- Expected results
- Troubleshooting
- Comparison tables

### Files Modified

**1. src/model.py**
- Updated `create_model()` to support hybrid types
- Added: 'hybrid_dqn', 'hybrid_dueling_dqn'

**2. train_progressive_improved.py**
- Added `--model_type` choices
- Validation: hybrid requires 8-channel mode
- Enhanced console output

### Testing

```bash
python test_hybrid_model.py
```

**Results**:
```
✅ Environment: 8 channels confirmed
✅ HybridDQN: Initialized successfully
✅ Forward pass: Working
✅ Training loop: 10 episodes completed
✅ Gradient flow: Both branches learning
✅ Channel separation: Visual and feature processed separately

✅ ALL TESTS PASSED - HYBRID ARCHITECTURE READY FOR TRAINING!
```

**Expected Results** (based on research):

| Episode Range | Visual-Only | Hybrid Expected | Speedup |
|---------------|-------------|-----------------|---------|
| 0-1,000 | 0 lines/ep | 0.1-0.5 lines/ep | First lines! |
| 1,000-2,000 | 0 lines/ep | 0.5-1.5 lines/ep | 5-10x faster |
| 2,000-5,000 | 0 lines/ep | 1.5-3.0 lines/ep | 10-20x faster |
| 5,000-10,000 | 0.2 lines/ep | 3.0-5.0 lines/ep | **15-25x faster** |

**Confidence**: 70% this achieves 3+ lines/episode by episode 10,000

**Reference**: `HYBRID_DQN_GUIDE.md`

---

## Current Training

**Started**: November 6, 2025
**Model**: `hybrid_dqn` (Dual-Branch Architecture)
**Episodes**: 10,000
**Expected Duration**: ~7 hours

**Configuration**:
```bash
python train_progressive_improved.py \
    --episodes 10000 \
    --force_fresh \
    --model_type hybrid_dqn \
    --experiment_name hybrid_10k
```

**Monitoring**:
- Watch for first line clear: Episode 500-1,000 (vs 2,600 baseline)
- Lines/episode trend: Should increase steadily
- Holes: Should decrease below 30
- Multi-line clears: Should appear around episode 5,000

**Success Criteria**:
- ✅ Lines/episode > 2.0 at 10K (vs 0.21 baseline)
- ✅ First line clear before episode 1,500
- ✅ Holes < 25 (vs 48 baseline)
- ✅ Multi-line clears (2-4 lines at once)

**Next Steps After Training**:
1. Analyze DEBUG_SUMMARY.txt
2. Compare to visual-only baseline
3. If successful (3+ lines/ep): Extend to 20K episodes
4. If very successful: Try hybrid_dueling_dqn
5. Document results and final recommendations

---

## Key Learnings

### 1. Metrics Must Reflect Reality
**Learning**: Measuring holes only at game-over is misleading
**Fix**: Track metrics during play (every 20 steps)
**Impact**: Accurate performance assessment

### 2. Visual-Only Learning is Slow
**Learning**: Pure visual approaches take 50K-100K episodes
**Evidence**: 0.21 lines/ep after 10K with visual-only
**Solution**: Add explicit features

### 3. Features Alone Are Not Enough
**Learning**: Adding feature channels (4→8) doesn't guarantee speedup
**Problem**: Generic CNN doesn't know how to use them
**Evidence**: 8-channel training performed like 4-channel visual-only

### 4. Architecture Matters for Hybrid Approaches
**Learning**: Feature-based approaches need feature-aware architectures
**Problem**: Treating numeric features like visual patterns defeats the purpose
**Solution**: Dual-branch processing with late fusion

### 5. Research Validates Implementation Choices
**Learning**: Successful Tetris DQN implementations use explicit features
**Evidence**: 90% of high-performing approaches are feature-based
**Impact**: Confidence in hybrid architecture approach

### 6. Testing Before Training Saves Time
**Learning**: Comprehensive tests catch issues early
**Example**: Test script found API mismatch (act vs select_action)
**Impact**: Avoided wasting 7 hours on broken training run

### 7. Progressive Curriculum Works
**Learning**: 5-stage reward shaping helps agent learn incrementally
**Stages**: Foundation → Clean placement → Spreading → Clean spreading → Line clearing
**Evidence**: Agent successfully learned column spreading (10/10 columns used)

### 8. Dropout and Train/Eval Mode Are Critical for RL
**Learning**: Classification defaults don't work for RL
**Fixes Applied**:
- Dropout: 0.3 → 0.1
- Must call model.train() and model.eval() appropriately
**Impact**: Stable learning, no random neuron dropout during inference

---

## Experiments Summary

| Experiment | Method | Result | Lesson Learned |
|------------|--------|--------|----------------|
| **Initial Training** | 4-channel visual-only | Slow learning | Need more information |
| **Hole Measurement Fix** | Track during play vs end | Better metrics | Measure what matters |
| **8-Channel Visual+Features** | Generic CNN with 8ch | Visual-only speed | Architecture matters |
| **Hybrid Dual-Branch** | Separate visual/feature CNNs | **In progress** | Proper feature utilization |

---

## Timeline

**Week 1 (Early Nov 2025)**:
- Initial setup and training
- Critical fixes (dropout, train/eval mode)
- Hole measurement discovery and fix

**Week 2 (Mid Nov 2025)**:
- Research phase (DQN approaches)
- Feature channel implementation
- 8-channel environment completed
- Integration tests passed

**November 4-5, 2025**:
- First 10K training run (8-channel generic CNN)
- Results: Visual-only performance despite features

**November 6, 2025**:
- Root cause analysis
- Dual-branch architecture implementation
- Testing completed
- **Hybrid DQN training started**

---

## Files Organization

### Active Documentation (Root)
- README.md - Project overview
- CLAUDE.md - Claude Code instructions
- HYBRID_DQN_GUIDE.md - Current implementation
- DQN_RESEARCH_ANALYSIS.md - Research findings
- IMPLEMENTATION_PLAN.md - Feature channel plan
- PROJECT_HISTORY.md - This file

### Historical Documentation (reports/)
- All analysis and fix documentation
- Training result analyses
- Implementation summaries
- Debug logs

### Code Structure
- `src/` - Core modules
  - `model.py` - Standard DQN models
  - `model_hybrid.py` - Hybrid dual-branch models
  - `agent.py` - DQN agent with adaptive epsilon
  - `feature_heatmaps.py` - Feature computation
  - `progressive_reward_improved.py` - 5-stage curriculum
  - `reward_shaping.py` - Core reward functions
  - `utils.py` - Logging and utilities

- `tests/` - Test suites
  - `test_feature_heatmaps.py` - Feature computation tests
  - `test_feature_channels_training.py` - Integration tests
  - `test_hybrid_model.py` - Hybrid architecture tests

- `config.py` - Environment wrapper (8-channel mode)
- `train_progressive_improved.py` - Main training script
- `visualize_features.py` - Visualization tool
- `evaluate.py` - Model evaluation

---

## What's Next

**Immediate** (Nov 6, 2025):
- Monitor hybrid DQN training progress
- Check for first line clear around episode 500-1,000
- Verify learning curve shows acceleration

**Short-term** (After 10K training):
- Analyze results vs baseline
- If successful: Extend to 20K episodes
- If very successful: Try hybrid_dueling_dqn
- Document final results

**Medium-term**:
- Potential optimizations:
  - Hyperparameter tuning (learning rate, epsilon schedule)
  - Curriculum stage adjustments
  - Reward function refinement
- Consider multi-game training
- Evaluate transfer learning potential

**Long-term**:
- Compare against human play
- Analyze learned strategies
- Potential publication or demo

---

## Success Metrics

**Technical Success**:
- ✅ Proper metric tracking (holes during play)
- ✅ 8-channel environment working
- ✅ Feature heatmaps computed correctly
- ✅ Dual-branch architecture implemented
- 🔄 Training in progress

**Performance Success** (to be measured):
- Target: 3+ lines/episode at 10K episodes
- Target: First line clear before episode 1,500
- Target: Holes < 25 during play
- Target: Multi-line clears appearing

**Learning Success**:
- Understanding of why visual-only is slow
- Knowledge of hybrid architecture benefits
- Proper metric design and tracking
- Importance of architecture for data types

---

*Last Updated*: November 6, 2025 (Training started)
*Next Update*: After hybrid training completes (~7 hours)
