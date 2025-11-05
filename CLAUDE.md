# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) implementation for training AI agents to play Tetris using the Tetris Gymnasium environment. The project uses a multi-channel CNN architecture with progressive reward shaping to teach agents advanced Tetris strategies including hole avoidance, column spreading, and line clearing.

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
# Standard training (basic DQN)
python train.py --episodes 500

# Progressive training (recommended for long runs)
python train_progressive_improved.py --episodes 75000 --resume

# Resume from checkpoint
python train_progressive_improved.py --episodes 75000 --resume

# Fresh training (ignore checkpoints)
python train_progressive_improved.py --episodes 10000 --force_fresh
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

**1. 4-Channel Vision System**
- The environment wrapper (`CompleteVisionWrapper` in `config.py`) converts dict observations to 4-channel arrays
- Output shape: `(20, 10, 4)` representing height × width × channels
- Channel breakdown:
  - Channel 0: Board state (locked pieces)
  - Channel 1: Active tetromino (falling piece)
  - Channel 2: Holder (held piece for swap)
  - Channel 3: Queue (preview of next pieces)

**2. Board Extraction (CRITICAL)**
- Tetris Gymnasium raw board is `(24, 18)`:
  - Rows 0-19: Spawn + playable area (extract these)
  - Rows 20-23: Bottom wall (NOT playable, skip these)
  - Cols 4-13: Playable width
  - Cols 0-3, 14-17: Side walls
- Extract playable area: `board[0:20, 4:14]` to get `(20, 10)` array
- This extraction logic is in `CompleteVisionWrapper.observation()` and `extract_board_from_obs()` in `src/reward_shaping.py`

**3. DQN Model with CNN**
- Located in `src/model.py`
- Two architectures: Standard DQN and Dueling DQN
- CNN layers optimized for 20×10 Tetris boards:
  - Conv1: 32 filters, 3×3 kernel, stride=1, padding=1
  - Conv2: 64 filters, 4×4 kernel, stride=2, padding=1
  - Conv3: 64 filters, 3×3 kernel, stride=1, padding=1
- Fully connected layers: Conv output → 512 → 256 → n_actions
- **CRITICAL**: Dropout rate is 0.1 (not 0.3) for RL applications

**4. Critical Fixes Applied (See CRITICAL_FIXES_APPLIED.md)**
- **Dropout Fix**: Reduced from 0.3 to 0.1 in all model layers
- **Train/Eval Mode Fix**: Added `model.train()` in `agent.learn()` and `model.eval()` in `agent.act()`
  - Without these, dropout was ALWAYS active (even during inference)
  - This bug caused 30% random neurons to be off during play
  - Fix in `src/agent.py` lines 224 (eval mode) and 309 (train mode)

**5. Progressive Reward Shaping**
- Implemented in `src/progressive_reward_improved.py`
- 5-stage curriculum:
  - Stage 1 (0-500 episodes): Foundation - basic placement
  - Stage 2 (500-1000): Clean placement - reduce holes
  - Stage 3 (1000-2000): Spreading - use all columns
  - Stage 4 (2000-5000): Clean spreading - hole-free spreading
  - Stage 5 (5000+): Line clearing - efficient clearing
- Dynamically adjusts reward weights based on episode count
- Metrics tracked: holes, bumpiness, column heights, completable rows, clean rows, line clears

**6. Agent with Adaptive Epsilon**
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
├── config.py                           # Environment config, CompleteVisionWrapper
├── train.py                            # Basic training script
├── train_progressive_improved.py       # Progressive curriculum training (recommended)
├── evaluate.py                         # Model evaluation
├── monitor_training.py                 # Training monitoring utility
├── requirements.txt                    # Python dependencies
├── src/
│   ├── agent.py                       # DQN agent with adaptive epsilon
│   ├── model.py                       # DQN and Dueling DQN architectures
│   ├── reward_shaping.py              # Core reward shaping functions
│   ├── progressive_reward_improved.py # Progressive curriculum shaper
│   ├── utils.py                       # Logging, plotting utilities
│   └── env_wrapper.py                 # Additional environment wrappers
├── tests/
│   ├── test_*.py                      # Unit tests
│   ├── diagnose_*.py                  # Diagnostic scripts
│   └── verify_*.py                    # Verification scripts
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
- `14H_TRAINING_PLAN.md` - Long-term training strategy
- `CENTER_STACKING_FIXES.md` - Solutions to column spreading problem

## Known Issues & Solutions

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
- Episodes 0-500: Basic placement, ~20-40 steps/episode
- Episodes 500-2000: Improved placement, ~100-150 steps/episode
- Episodes 2000-12500: Column spreading mastered, ~175 steps/episode
- Episodes 12500-30000: Hole reduction (43 → 20-25 holes)
- Episodes 30000-50000: Line clearing emerges (0.5-1 lines/episode)
- Episodes 50000-75000: Expert play (2-5 lines/episode)

## Debugging Tips

1. **Training not progressing**: Check epsilon value with `agent.epsilon` - should decrease over time
2. **Model not loading**: Verify checkpoint path, check `models/` directory
3. **Observation shape errors**: Verify wrapper is active with `use_complete_vision=True`
4. **Import errors**: Run `python tests/verify_imports.py`
5. **Action space issues**: Print `env.action_space` and verify 8 actions
6. **Board extraction bugs**: Test with `tests/test_board_extraction_fix.py`

## Testing Guidelines

- Run relevant tests before committing changes
- Add new tests to `tests/` directory following pattern `test_<behavior>.py`
- Use deterministic seeds: `numpy.random.seed(42)` for reproducibility
- Keep tests standalone (can run via `python tests/<file>.py`)
- Smoke test suite: `test_actions_simple.py`, `test_reward_helpers.py`, environment check
