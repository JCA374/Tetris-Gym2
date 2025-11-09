# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) implementation for training AI agents to play Tetris using the Tetris Gymnasium environment. The project uses a **Feature Vector DQN approach** with direct scalar features (17 values) fed to a simple fully-connected network.

**Current Status**: Feature vector implementation complete and active. This approach represents a pivot from the previous hybrid CNN method after competitive analysis showed feature vectors outperform visual approaches by 100-1000x.

**Key Insight**: Most successful Tetris DQN implementations (90%+) use direct feature scalars, not image-based CNNs. The current approach extracts 17 features (holes, heights, bumpiness, wells) and feeds them directly to a fully-connected network, achieving much better sample efficiency.

**Previous Approach**: Hybrid Dual-Branch CNN with 8-channel observations has been archived. It was theoretically interesting but practically inefficient (0.7 lines/episode at 15K episodes vs expected 100-1000+ lines at 5K with feature vectors).

## Common Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Training (Feature Vector DQN - CURRENT)
```bash
# Quick test (100 episodes, ~1 minute)
python train_feature_vector.py --episodes 100 --log_freq 10

# Recommended training (5,000 episodes, ~3-5 hours)
python train_feature_vector.py \
    --episodes 5000 \
    --model_type fc_dqn \
    --experiment_name feature_5k

# Full training (10,000 episodes, ~6-8 hours)
python train_feature_vector.py \
    --episodes 10000 \
    --model_type fc_dqn \
    --experiment_name feature_10k

# Try Dueling DQN (may be 10-20% better)
python train_feature_vector.py \
    --episodes 5000 \
    --model_type fc_dueling_dqn \
    --experiment_name feature_dueling_5k

# Force fresh start (ignore existing checkpoints)
python train_feature_vector.py --episodes 5000 --force_fresh

# Custom hyperparameters
python train_feature_vector.py \
    --episodes 5000 \
    --lr 0.0005 \
    --batch_size 128 \
    --epsilon_decay 0.999
```

### Analysis & Evaluation
```bash
# Analyze completed training run
python analyze_training.py logs/feature_vector_fc_dqn_<timestamp>

# Evaluate trained model (if evaluate.py supports feature vector)
python evaluate.py --model_path models/best_model.pth --render
```

### Testing & Diagnostics
```bash
# Test feature extraction
python src/feature_vector.py

# Test model creation
python src/model_fc.py

# Test environment wrapper
python src/env_feature_vector.py
```

## Architecture Overview

### Current Approach: Feature Vector DQN ⭐ ACTIVE

**1. Feature Extraction (17 Scalar Values)**
- Located in `src/feature_vector.py`
- Extracts from Tetris board state:
  ```python
  Feature Vector (17 values):
  - aggregate_height    # Sum of all column heights
  - holes              # Count of holes (empty cells with filled above)
  - bumpiness          # Sum of height differences between adjacent columns
  - wells              # Sum of well depths (valleys between columns)
  - column_heights[10] # Individual heights for each column
  - max_height         # Maximum column height
  - min_height         # Minimum column height
  - std_height         # Standard deviation of heights
  ```
- All features normalized to [0, 1] range for stable learning
- Board extraction: `board[0:20, 4:14]` from Tetris Gymnasium's `(24, 18)` raw board

**2. Environment Wrapper**
- `FeatureVectorWrapper` in `src/env_feature_vector.py`
- Converts dict observations (board, active, holder, queue) → 17-dim feature vector
- Factory function: `make_feature_vector_env()`
- Observation space: `Box(0.0, 1.0, (17,), float32)`

**3. Feature Vector DQN Model**
- Located in `src/model_fc.py`
- **Architecture**: Simple fully-connected network (NO CNNs)
  ```
  Input: 17 features (normalized 0-1)
      ↓
  FC: 17 → 256 (ReLU, Dropout 0.1)
      ↓
  FC: 256 → 128 (ReLU, Dropout 0.1)
      ↓
  FC: 128 → 64 (ReLU, Dropout 0.1)
      ↓
  Output: 64 → 8 Q-values

  Total parameters: ~46,000 (vs 1.2M for hybrid CNN)
  ```
- Two variants: `FeatureVectorDQN` (standard), `FeatureVectorDuelingDQN` (value/advantage streams)
- Factory: `create_feature_vector_model(model_type='fc_dqn')`

**4. Why Feature Vectors Work Better**
- **Direct representation**: Agent immediately "sees" holes, heights (no need to learn detection)
- **Smaller state space**: 17 continuous dimensions vs 1,600 visual pixels
- **Proven approach**: 90% of successful Tetris DQN implementations use feature vectors
- **Sample efficiency**: 100-1000x better than visual approaches (research-validated)

**5. Simple Reward Function**
- Located in `train_feature_vector.py` (`simple_reward()` function)
- Positive survival reward + big line clear bonuses - penalties for holes/height
  ```python
  reward = 1.0                          # Positive for surviving
         + lines_cleared * 100          # Huge bonus for lines
         - holes * 2.0                  # Penalize holes (if available)
         - aggregate_height * 0.1       # Slight penalty for height
  ```
- **Critical fix**: Changed from negative per-step penalty (which taught agent to die fast)

**6. Agent Implementation**
- Located in `src/agent.py`
- Supports feature vector models via `model_type='fc_dqn'` or `fc_dueling_dqn`
- Adaptive epsilon decay with 4 phases (discovery, exploitation, refinement, mastery)
- Experience replay buffer: 100,000 transitions, 1,000 warmup
- Target network updated every 1,000 steps
- **Critical**: Uses `model.eval()` for inference, `model.train()` for learning (dropout fix)

## Important Code Patterns

### Feature Extraction
When working with board states:
- Extract playable area from raw board: `board[0:20, 4:14]` → `(20, 10)` array
- Binarize boards for feature computation: `(board > 0).astype(np.uint8)`
- Use helper functions in `src/feature_vector.py`:
  - `get_column_heights(board)` - returns 10-element array
  - `count_holes(board)` - counts empty cells with filled cells above
  - `calculate_bumpiness(column_heights)` - sum of adjacent height differences
  - `calculate_wells(column_heights)` - sum of valley depths
- Always normalize features to [0, 1] with `normalize_features(features)`

### Adding New Features
When extending the feature vector:
1. Add extraction logic in `src/feature_vector.py`
2. Update `extract_feature_vector()` to include new feature
3. Update normalization constants in `normalize_features()`
4. Update model input size if total feature count changes
5. Test extraction with `python src/feature_vector.py`

### Reward Function Design
When modifying `simple_reward()` in `train_feature_vector.py`:
- **Always use positive survival reward** (not negative per-step penalty)
- Large bonuses for line clears (e.g., `lines * 100`)
- Moderate penalties for bad board states (holes, height)
- Keep reward components balanced (avoid one term dominating)
- **Critical**: Negative per-step rewards teach agent to die quickly!

### Training Loop Modifications
When editing `train_feature_vector.py`:
- Call `agent.select_action(state, training=True)` for action selection
- Store transitions: `agent.remember(state, action, reward, next_state, done, info=info, original_reward=env_reward)`
- Trigger learning: `agent.learn()`
- End episode: `agent.end_episode(total_reward, steps, lines_cleared, original_reward=env_reward)`
- Log metrics: `logger.log_episode(episode, reward, steps, epsilon, lines_cleared, ...)`
- Log board states: `logger.log_board_state(episode, board, reward, steps, lines_cleared, heights=..., features_normalized=...)`

## Directory Structure

```
Tetris-Gym2/
├── README.md                           # Project overview
├── CLAUDE.md                           # This file - Claude Code guidance
├── INDEX.md                            # Documentation navigation guide
├── FEATURE_VECTOR_GUIDE.md            # Complete guide for feature vector approach
├── COMPETITIVE_ANALYSIS.md            # Why feature vectors beat CNNs
├── LOGGING_GUIDE.md                   # Logging and analysis documentation
├── CLEANUP_PLAN.md                    # Record of archived hybrid CNN files
├── train_feature_vector.py            # Training script (CURRENT)
├── analyze_training.py                # Post-training analysis tool
├── requirements.txt                    # Python dependencies
│
├── src/                               # Core library code
│   ├── agent.py                       # DQN agent (supports feature vector models)
│   ├── feature_vector.py              # Feature extraction (17 scalars)
│   ├── model_fc.py                    # Feature vector DQN models (CURRENT)
│   ├── env_feature_vector.py          # Environment wrapper for feature vectors
│   ├── utils.py                       # Logging, plotting utilities
│   ├── model.py                       # Legacy: Standard/Dueling DQN (CNNs)
│   ├── progressive_reward_improved.py # Legacy: 5-stage curriculum (for hybrid)
│   └── reward_shaping.py              # Legacy: Complex reward functions
│
├── archive_files/                     # Archived hybrid CNN implementation
│   ├── hybrid_cnn/                    # Hybrid CNN models and wrappers
│   ├── training_scripts/              # Old training scripts
│   ├── tests/                         # Diagnostic tests for hybrid approach
│   └── docs/                          # Hybrid CNN documentation
│
├── logs/                              # Training logs (auto-created)
│   └── feature_vector_fc_dqn_<timestamp>/
│       ├── episode_log.csv            # Per-episode metrics
│       ├── board_states.txt           # Final board visualizations
│       ├── reward_progress.png        # Reward curves
│       ├── training_metrics.png       # Steps/epsilon curves
│       └── training_analysis.png      # 6-panel analysis
│
└── models/                            # Saved checkpoints (auto-created)
    ├── best_model.pth                 # Best performance
    ├── final_model.pth                # End of training
    └── checkpoint_ep<N>.pth           # Periodic checkpoints
```

## Training Configuration

Default hyperparameters in `train_feature_vector.py`:
- Learning rate: 0.0001
- Gamma (discount): 0.99
- Batch size: 64
- Memory size: 100,000 transitions
- Min memory before learning: 1,000 transitions
- Epsilon start: 1.0, end: 0.05, decay: 0.9995
- Target network update: every 1,000 steps
- Log frequency: every 10 episodes
- Save frequency: every 500 episodes

## Action Space

Tetris Gymnasium v0.3.0 action mapping:
```python
0: LEFT, 1: RIGHT, 2: DOWN, 3: ROTATE_CW,
4: ROTATE_CCW, 5: HARD_DROP, 6: SWAP, 7: NOOP
```

## Important Files to Check Before Modifying

**Essential Guides:**
- `FEATURE_VECTOR_GUIDE.md` - **START HERE**: Complete implementation and usage guide
- `COMPETITIVE_ANALYSIS.md` - Why feature vectors beat hybrid CNNs (100-1000x better)
- `LOGGING_GUIDE.md` - How to monitor and evaluate training

**Core Implementation:**
- `src/feature_vector.py` - Feature extraction logic (17 scalars)
- `src/model_fc.py` - Model architectures (FC networks)
- `train_feature_vector.py` - Training script with reward function
- `analyze_training.py` - Post-training analysis tool

**Historical Context:**
- `CLEANUP_PLAN.md` - What was archived and why
- `archive_files/` - Previous hybrid CNN implementation (archived Nov 2025)

## Known Issues & Solutions

**Issue: Broken reward function caused agent to learn dying fast (FIXED - Nov 2025)**
- **Problem**: Original reward used negative per-step penalty (`-0.1 per step`)
- **Result**: With no line clears, agent learned optimal strategy was dying immediately to minimize penalty
- **Symptoms**: Steps decreased from 70 → 23 over 5,000 episodes (getting worse!)
- **Solution**: Changed to positive survival reward (`+1.0 per step`) with line clear bonuses
- **Location**: `train_feature_vector.py` `simple_reward()` function
- **Critical lesson**: Never use negative per-step rewards in survival tasks!

**Issue: Duplicate directory nesting in logs (FIXED - Nov 2025)**
- **Problem**: Logs created nested structure: `logs/name/name/files`
- **Cause**: TrainingLogger and training script both adding experiment_name to path
- **Solution**: Pass only `Path("logs")` to TrainingLogger, let it handle nesting once
- **Status**: Fixed in `train_feature_vector.py` line 185

**Issue: Cluttered console output (FIXED - Nov 2025)**
- **Problem**: Agent printed epsilon/phase info every episode
- **Solution**: Reduced agent logging to 1000-episode milestones only
- **Result**: Clean console output showing training progress clearly

**Issue: Agent learning slowly or not at all**
- Check reward function - should see positive rewards accumulating
- Verify epsilon is decaying (should be <0.5 by episode 1000)
- Ensure memory buffer is filling (needs 1,000 transitions before learning starts)
- Check if lines cleared is increasing over episodes
- Try longer training (5,000+ episodes for feature vector approach)

**Issue: High hole counts in logs**
- Feature vector logs show normalized holes (0-1 range), not raw counts
- Final state holes are naturally higher than during-play average
- Focus on trend over episodes, not absolute values

## Performance Expectations

### Feature Vector DQN (5,000 episodes, ~3-5 hours)

Based on research of successful feature vector implementations:

| Episode Range | Lines/Episode | Steps/Episode | Status |
|---------------|--------------|---------------|---------|
| 0-500 | 0-1 | 50-150 | Learning survival |
| 500-1,000 | 1-5 | 150-250 | Basic line clearing |
| 1,000-2,000 | 5-20 | 250-400 | Consistent clearing |
| 2,000-5,000 | 20-100 | 400-800 | Advanced strategy |
| 5,000+ | 100-1,000+ | 800-2,000+ | Expert performance |

**Success Criteria at 5,000 Episodes:**
- ✅ Lines/episode > 50 (vs 0.7 for hybrid CNN)
- ✅ First 100+ line episode before episode 3,000
- ✅ Consistent line clearing (not random luck)
- ✅ Reward trend clearly upward
- ✅ Longer survival (500+ steps regularly)

**Key Metrics to Watch:**
- **Lines cleared**: Primary metric, should increase over time
- **Reward**: Should trend upward (becomes positive as agent improves)
- **Steps**: Should increase (longer survival = more line opportunities)
- **Epsilon**: Should decay smoothly from 1.0 toward 0.05
- **Holes (normalized)**: Should decrease over time in final states

## Debugging Tips

1. **Agent not learning**:
   - Check reward function - should see positive rewards for survival
   - Verify epsilon decaying (check console output every 10 episodes)
   - Ensure replay buffer filling (needs 1,000 transitions to start learning)
   - Look at training curves with `analyze_training.py`

2. **Zero line clears after many episodes**:
   - Normal for first 200-500 episodes while agent learns survival
   - If still zero after 1,000 episodes, check reward function incentivizes line clears
   - Verify agent is learning at all (check if reward is improving)

3. **Training very slow**:
   - Feature vector should be fast (~15-20 episodes/second on CPU)
   - If much slower, check for infinite loops or blocking I/O
   - Reduce logging frequency if needed: `--log_freq 50`

4. **Observation shape errors**:
   - Verify FeatureVectorWrapper is active (should see "Feature vector mode ACTIVE" on startup)
   - Check observation space is `Box(0.0, 1.0, (17,), float32)`
   - Test wrapper: `python src/env_feature_vector.py`

5. **Import errors**:
   - Activate virtual environment: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`
   - Test individual modules: `python src/feature_vector.py`, `python src/model_fc.py`

6. **Model not loading**:
   - Verify checkpoint path exists in `models/` directory
   - Check if model architecture matches (input_size=17, output_size=8)
   - Feature vector models incompatible with CNN models

7. **Reward function issues**:
   - **Critical**: Never use negative per-step rewards (teaches dying fast!)
   - Reward should be positive for survival + bonuses for good actions
   - Check `simple_reward()` in `train_feature_vector.py`

## Testing Guidelines

**Quick Tests:**
```bash
# Test feature extraction
python src/feature_vector.py

# Test model creation
python src/model_fc.py

# Test environment wrapper
python src/env_feature_vector.py

# Quick training test (100 episodes)
python train_feature_vector.py --episodes 100 --log_freq 10
```

**Validating Training:**
- Run short training (100-500 episodes) first to verify setup
- Check logs directory created with episode_log.csv
- Verify reward is increasing over episodes
- Confirm epsilon is decaying properly
- Look for first line clear within first 200-500 episodes
