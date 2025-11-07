# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) implementation for training AI agents to play Tetris using the Tetris Gymnasium environment. The project uses a **Hybrid Dual-Branch CNN architecture** with 8-channel observations (visual + explicit features) and progressive reward shaping to teach agents advanced Tetris strategies.

**Current Status**: Hybrid DQN training in progress (expected 15-25x faster learning than visual-only)

**Key Innovation**: Dual-branch architecture that separately processes visual channels (board, active piece, holder, queue) and feature channels (holes, heights, bumpiness, wells) before fusion.

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

### Training
```bash
# RECOMMENDED: Hybrid Dual-Branch DQN (10-50x faster learning)
python train_progressive_improved.py \
    --episodes 10000 \
    --force_fresh \
    --model_type hybrid_dqn \
    --experiment_name hybrid_10k

# Advanced: Hybrid Dueling DQN (10-20% better than standard hybrid)
python train_progressive_improved.py \
    --episodes 10000 \
    --model_type hybrid_dueling_dqn \
    --force_fresh

# Test hybrid architecture before full training
python test_hybrid_model.py

# Resume training from checkpoint
python train_progressive_improved.py --episodes 20000 --resume --model_type hybrid_dqn

# Baseline: Standard DQN (for comparison)
python train_progressive_improved.py --episodes 10000 --force_fresh --model_type dqn
```

### Evaluation
```bash
# Evaluate trained model with rendering
python evaluate.py --model_path models/best_model.pth --render

# Evaluate without rendering (faster)
python evaluate.py --model_path models/best_model.pth

# Detailed evaluation with video
python evaluate.py --episodes 20 --render --save_video --detailed
```

### Testing & Diagnostics
```bash
# Test environment setup
python config.py

# Run specific diagnostic tests
python tests/test_actions_simple.py
python tests/test_reward_helpers.py
python tests/test_4channel_wrapper.py
python tests/diagnose_training.py
python tests/diagnose_model.py

# Verify imports
python tests/verify_imports.py
```

### Monitoring Training
```bash
# Monitor ongoing training
python monitor_training.py
```

## Architecture Overview

### Key Architectural Decisions

**1. 8-Channel Hybrid Vision System**
- The environment wrapper (`CompleteVisionWrapper` in `config.py`) converts dict observations to 8-channel arrays
- Output shape: `(20, 10, 8)` representing height × width × channels
- **Visual channels (0-3)**:
  - Channel 0: Board state (locked pieces)
  - Channel 1: Active tetromino (falling piece with rotation)
  - Channel 2: Holder (held piece for swap)
  - Channel 3: Queue (preview of next pieces)
- **Feature channels (4-7)** - Explicit spatial heatmaps:
  - Channel 4: Holes heatmap (where holes exist)
  - Channel 5: Height map (normalized column heights)
  - Channel 6: Bumpiness map (height variation)
  - Channel 7: Wells map (valleys between columns)

**2. Board Extraction (CRITICAL)**
- Tetris Gymnasium raw board is `(24, 18)`:
  - Rows 0-19: Spawn + playable area (extract these)
  - Rows 20-23: Bottom wall (NOT playable, skip these)
  - Cols 4-13: Playable width
  - Cols 0-3, 14-17: Side walls
- Extract playable area: `board[0:20, 4:14]` to get `(20, 10)` array
- This extraction logic is in `CompleteVisionWrapper.observation()` and `extract_board_from_obs()` in `src/reward_shaping.py`

**3. Hybrid Dual-Branch DQN Model** ⭐ RECOMMENDED
- Located in `src/model_hybrid.py`
- **Architecture**: Separate processing for visual and feature data
  ```
  Input (20×10×8)
      ↓
      ├─→ Visual CNN (ch 0-3) → 3,200 features
      │   Conv2d(4→32→64→64)
      │   Optimized for spatial patterns
      │
      └─→ Feature CNN (ch 4-7) → 1,600 features
          Conv2d(4→16→32)
          Simpler - features already meaningful
              ↓
          Concatenate (4,800 features)
              ↓
          FC: 4800→512→256→8 Q-values
  ```
- **Why dual-branch?** Generic CNNs mix visual and feature channels immediately, diluting explicit feature signals. Dual-branch processes each optimally.
- Two variants: `HybridDQN`, `HybridDuelingDQN`
- **CRITICAL**: Dropout rate is 0.1 (not 0.3) for RL applications

**4. Standard DQN Models** (Baseline)
- Located in `src/model.py`
- Generic CNN that treats all channels the same
- Use for comparison or 4-channel visual-only mode
- Variants: `DQN`, `DuelingDQN`

**5. Critical Fixes Applied (See reports/archive/CRITICAL_FIXES_APPLIED.md)**
- **Dropout Fix**: Reduced from 0.3 to 0.1 in all model layers
- **Train/Eval Mode Fix**: Added `model.train()` in `agent.learn()` and `model.eval()` in `agent.act()`
  - Without these, dropout was ALWAYS active (even during inference)
  - This bug caused 30% random neurons to be off during play
  - Fix in `src/agent.py` lines 224 (eval mode) and 309 (train mode)

**6. Progressive Reward Shaping**
- Implemented in `src/progressive_reward_improved.py`
- 5-stage curriculum:
  - Stage 1 (0-500 episodes): Foundation - basic placement
  - Stage 2 (500-1000): Clean placement - reduce holes
  - Stage 3 (1000-2000): Spreading - use all columns
  - Stage 4 (2000-5000): Clean spreading - hole-free spreading
  - Stage 5 (5000+): Line clearing - efficient clearing
- Dynamically adjusts reward weights based on episode count
- Metrics tracked: holes, bumpiness, column heights, completable rows, clean rows, line clears

**7. Agent with Adaptive Epsilon**
- Located in `src/agent.py`
- Supports three epsilon decay methods:
  - Exponential decay (default)
  - Linear decay
  - Adaptive schedule (optimized for Tetris learning phases)
- Experience replay with prioritization
- Target network updated every 1000 steps
- Memory size: 200,000 transitions

## Important Code Patterns

### State Preprocessing
When adding features that process observations:
- Always handle both dict observations (raw environment) and array observations (wrapped)
- Extract channel 0 for board state from 3D arrays: `board = obs[:, :, 0]`
- For raw boards, extract playable area: `board[0:20, 4:14]`
- Binarize boards for metrics: `(board > 0).astype(np.uint8)`

### Adding New Reward Metrics
When modifying reward shaping in `src/reward_shaping.py` or `src/progressive_reward_improved.py`:
- Use `extract_board_from_obs()` to get consistent 20×10 binary board
- Helper functions available: `get_column_heights()`, `count_holes()`, `calculate_bumpiness()`, `count_completable_rows()`, `count_clean_rows()`
- Keep reward components in predictable ranges (normalize large penalties)
- Test with `tests/test_reward_helpers.py`

### Model Changes
When modifying neural networks in `src/model.py`:
- Keep dropout at 0.1 for RL (not higher)
- Ensure `model.train()` is called before training steps
- Ensure `model.eval()` is called before inference/action selection
- Maintain input format: `(batch, channels, height, width)` for CNNs
- Test with `tests/diagnose_model.py`

### Training Loop Modifications
When editing training scripts (`train.py`, `train_progressive_improved.py`):
- Use `agent.act(state)` for action selection (handles exploration/exploitation)
- Call `agent.remember(state, action, reward, next_state, done)` to store transitions
- Call `agent.learn()` to trigger training updates
- Use `logger.log_episode()` for consistent logging
- Save checkpoints at regular intervals (e.g., every 500 episodes)

## Directory Structure

```
tetris-rl/
├── README.md                           # Project overview and quick start
├── CLAUDE.md                           # This file - Claude Code guidance
├── HYBRID_DQN_GUIDE.md                # Current implementation guide
├── DQN_RESEARCH_ANALYSIS.md           # Research findings on DQN approaches
├── IMPLEMENTATION_PLAN.md             # Feature channel implementation plan
├── PROJECT_HISTORY.md                 # Complete project history and learnings
├── config.py                           # Environment config (8-channel wrapper)
├── train_progressive_improved.py       # Main training script with hybrid support
├── test_hybrid_model.py               # Test hybrid architecture
├── evaluate.py                         # Model evaluation
├── visualize_features.py              # Visualize 8 channels
├── requirements.txt                    # Python dependencies
├── src/
│   ├── agent.py                       # DQN agent with adaptive epsilon
│   ├── model.py                       # Standard DQN and Dueling DQN
│   ├── model_hybrid.py                # Hybrid Dual-Branch DQN (RECOMMENDED)
│   ├── feature_heatmaps.py            # Compute feature channel heatmaps
│   ├── reward_shaping.py              # Core reward shaping functions
│   ├── progressive_reward_improved.py # Progressive 5-stage curriculum
│   └── utils.py                       # Logging, plotting utilities
├── tests/
│   ├── test_feature_heatmaps.py       # Feature computation tests
│   ├── test_feature_channels_training.py # Integration tests
│   └── test_*.py                      # Other unit tests
├── reports/
│   ├── archive/                       # Historical documentation
│   │   ├── CRITICAL_FIXES_APPLIED.md  # Dropout and train/eval mode fixes
│   │   ├── HOLE_MEASUREMENT_FIX.md    # Metric tracking improvements
│   │   ├── TRAINING_ANALYSIS_10K.md   # Visual-only baseline analysis
│   │   ├── RECOMMENDATION.md          # Why dual-branch architecture
│   │   └── ...                        # Other historical docs
│   └── training_dqn_reward_review.md  # Old training analysis
├── archive_scripts/                    # Old debug scripts
├── models/                             # Saved model checkpoints
└── logs/                              # Training logs and plots
```

## Training Configuration

Default hyperparameters in `config.py`:
- Learning rate: 0.0001
- Gamma (discount): 0.99
- Batch size: 64
- Memory size: 100,000 (Agent uses 200,000)
- Epsilon start: 1.0, end: 0.05
- Target network update: every 1000 steps
- Board size: 20 (height) × 10 (width)

## Action Space

Tetris Gymnasium v0.3.0 action mapping (from `config.py`):
```python
0: LEFT, 1: RIGHT, 2: DOWN, 3: ROTATE_CW,
4: ROTATE_CCW, 5: HARD_DROP, 6: SWAP, 7: NOOP
```

## Important Files to Check Before Modifying

- `AGENTS.md` - Repository guidelines and conventions
- `CRITICAL_FIXES_APPLIED.md` - Critical bug fixes (dropout, train/eval)
- `HOLE_MEASUREMENT_FIX.md` - **CRITICAL**: How holes are measured (during play vs at game-over)
- `14H_TRAINING_PLAN.md` - Long-term training strategy
- `CENTER_STACKING_FIXES.md` - Solutions to column spreading problem

## Known Issues & Solutions

**Issue: Hole metrics misleading (CRITICAL FIX - Nov 2025)**
- **Problem**: Holes were only measured at game-over (worst possible moment)
- **Example**: "48 holes" at step 204 after board filled up ≠ board quality during play
- **Solution**: Now track average holes during play, minimum holes, and checkpoints
- **New metrics**:
  - `holes`: Average during play (sampled every 20 steps) - PRIMARY METRIC
  - `holes_min`: Best board state achieved in episode
  - `holes_final`: At game-over (for reference only)
  - `holes_at_step_X`: Consistent checkpoints (50, 100, 150)
- **Realistic goals**:
  - Average holes during play: <15 (achievable)
  - Final holes at game-over: <30-50 (depends on line clears)
- **See**: `HOLE_MEASUREMENT_FIX.md` for complete details

**Issue: Center stacking (agent only uses middle columns)**
- Solution: Progressive reward shaping with column spread penalty
- Implementation: See `src/progressive_reward_improved.py`

**Issue: Agent not learning line clears**
- Solution: Extended training (75,000+ episodes) with curriculum
- Stage 5 focuses on line clearing after mastering hole avoidance

**Issue: Dropout making inference inconsistent**
- Solution: Use `model.eval()` before action selection
- Already fixed in `src/agent.py` line 224

**Issue: Agent learning slowly**
- Solutions:
  - Verify dropout is 0.1 (not 0.3)
  - Ensure train/eval modes are properly set
  - Use progressive training for longer runs
  - Check epsilon decay schedule

## Performance Expectations

For 75,000 episode training (14-15 hours):

**Note**: Hole metrics changed Nov 2025 - now tracking average during play instead of at game-over.

| Episode Range | Avg Holes | Min Holes | Final Holes | Steps/Ep | Lines/Ep | Stage |
|---------------|-----------|-----------|-------------|----------|----------|-------|
| 0-500 | 30-40 | 20-30 | 50-70 | 20-40 | 0 | Foundation |
| 500-2000 | 25-35 | 15-25 | 45-60 | 100-150 | 0-0.1 | Clean placement |
| 2000-12500 | 20-30 | 10-15 | 40-55 | 150-200 | 0.1-0.5 | Spreading |
| 12500-30000 | 15-25 | 8-12 | 35-50 | 200-250 | 0.5-1.5 | Clean spreading |
| 30000-50000 | 12-20 | 5-10 | 25-40 | 250-350 | 1.5-3 | Line clearing |
| 50000-75000 | <15 | <8 | <30 | 300-500 | 3-5 | Expert play |

**Key Points**:
- Avg holes = Board quality during play (PRIMARY METRIC)
- Min holes = Best state achieved (shows potential)
- Final holes = At game-over (naturally higher, less important)
- Without line clears, final holes will remain high (40-50+)
- Line clears are the key to reducing final holes

## Debugging Tips

1. **High hole counts**: Check which metric - avg (during play) vs final (at game-over) - they differ significantly!
2. **Training not progressing**: Check epsilon value with `agent.epsilon` - should decrease over time
3. **Model not loading**: Verify checkpoint path, check `models/` directory
4. **Observation shape errors**: Verify wrapper is active with `use_complete_vision=True`
5. **Import errors**: Run `python tests/verify_imports.py`
6. **Action space issues**: Print `env.action_space` and verify 8 actions
7. **Board extraction bugs**: Test with `tests/test_board_extraction_fix.py`
8. **Comparing old vs new logs**: Pre-Nov 2025 logs only have final holes; new logs have avg/min/final - NOT directly comparable!

## Testing Guidelines

- Run relevant tests before committing changes
- Add new tests to `tests/` directory following pattern `test_<behavior>.py`
- Use deterministic seeds: `numpy.random.seed(42)` for reproducibility
- Keep tests standalone (can run via `python tests/<file>.py`)
- Smoke test suite: `test_actions_simple.py`, `test_reward_helpers.py`, environment check
