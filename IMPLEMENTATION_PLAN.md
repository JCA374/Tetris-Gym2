# Implementation Plan: Hybrid Feature Channels for DQN Tetris

**Date**: 2025-11-05
**Goal**: Add explicit feature channels to observation space for 10-20x learning speedup
**Approach**: Phase 1 - Hybrid (Visual + Feature Heatmaps)
**Timeline**: 4-6 hours implementation + testing

---

## 📋 Overview

### What We're Building

Extend observation from `(20, 10, 4)` to `(20, 10, 8)` by adding 4 feature heatmap channels:
1. **Holes heatmap**: Where holes exist on the board
2. **Height map**: Column heights (normalized 0-1)
3. **Bumpiness map**: Height variation between adjacent columns
4. **Wells map**: Depth of valleys between columns

### Why This Works

- Keeps spatial awareness (visual channels)
- Adds explicit guidance (feature channels)
- CNN can learn which features matter
- Network architecture needs NO changes (handles input size automatically)
- Can resume from checkpoint or start fresh

### Expected Results

- **Current**: 0.21 lines/episode after 75,000 episodes
- **Expected**: 5-20 lines/episode after 10,000 episodes (25-100x improvement)
- **Timeline**: Expert play by 20,000 episodes

---

## 🎯 Implementation Steps

### Step 1: Create Feature Heatmap Functions (30 mins)

**File**: `src/feature_heatmaps.py` (NEW)

**Functions to implement:**

```python
def compute_hole_heatmap(board: np.ndarray) -> np.ndarray:
    """
    Generate heatmap showing where holes are on the board.

    A hole is an empty cell with at least one filled cell above it.

    Args:
        board: (20, 10) binary board array

    Returns:
        (20, 10) array with 1.0 where holes are, 0.0 elsewhere
    """
    pass

def compute_height_map(board: np.ndarray) -> np.ndarray:
    """
    Generate heatmap showing column heights.

    Each column gets its normalized height (0-1) repeated down.

    Args:
        board: (20, 10) binary board array

    Returns:
        (20, 10) array with normalized column heights
    """
    pass

def compute_bumpiness_map(board: np.ndarray) -> np.ndarray:
    """
    Generate heatmap showing height differences between adjacent columns.

    Shows where board is bumpy (height variations).

    Args:
        board: (20, 10) binary board array

    Returns:
        (20, 10) array with normalized bumpiness values
    """
    pass

def compute_well_map(board: np.ndarray) -> np.ndarray:
    """
    Generate heatmap showing wells (valleys between columns).

    A well is a column that's lower than its neighbors.

    Args:
        board: (20, 10) binary board array

    Returns:
        (20, 10) array with normalized well depths
    """
    pass
```

**Testing**:
- Create `tests/test_feature_heatmaps.py`
- Test with known board configurations
- Verify output shapes and value ranges
- Visual inspection of heatmaps

---

### Step 2: Extend CompleteVisionWrapper (30 mins)

**File**: `config.py` (MODIFY)

**Changes to `CompleteVisionWrapper`:**

```python
class CompleteVisionWrapper(gym.ObservationWrapper):
    """
    Enhanced wrapper with 8 channels instead of 4:
    - Channels 0-3: Original (board, active, holder, queue)
    - Channel 4: Holes heatmap
    - Channel 5: Height map
    - Channel 6: Bumpiness map
    - Channel 7: Wells map
    """

    def __init__(self, env, use_feature_channels=True):
        super().__init__(env)
        self.use_feature_channels = use_feature_channels

        # Update observation space
        n_channels = 8 if use_feature_channels else 4
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,  # Changed from 255 - normalized to 0-1
            shape=(20, 10, n_channels),
            dtype=np.float32  # Changed from uint8
        )

    def observation(self, obs_dict):
        # Existing 4 channels (normalized to 0-1)
        channels = []

        # ... existing channel code ...

        if self.use_feature_channels:
            # Extract board for feature computation
            board = self._extract_board(obs_dict)

            # NEW: Compute feature heatmaps
            from src.feature_heatmaps import (
                compute_hole_heatmap,
                compute_height_map,
                compute_bumpiness_map,
                compute_well_map
            )

            holes_map = compute_hole_heatmap(board)
            height_map = compute_height_map(board)
            bumpiness_map = compute_bumpiness_map(board)
            well_map = compute_well_map(board)

            channels.extend([
                holes_map,
                height_map,
                bumpiness_map,
                well_map
            ])

        # Stack all channels
        observation = np.stack(channels, axis=-1).astype(np.float32)
        return observation
```

**Testing**:
- Verify observation shape: `(20, 10, 8)`
- Verify value ranges: `[0.0, 1.0]`
- Test with actual environment reset/step
- Check memory usage

---

### Step 3: Update Configuration (15 mins)

**File**: `config.py` (MODIFY)

**Add feature toggle:**

```python
# Feature channel configuration
USE_FEATURE_CHANNELS = True  # Set to False to revert to 4-channel

def make_env(render_mode=None, use_complete_vision=True,
             use_cnn=True, use_feature_channels=True):
    """
    Create Tetris environment with optional feature channels.

    Args:
        use_feature_channels: If True, use 8 channels (visual + features)
                             If False, use 4 channels (visual only)
    """
    env = gym.make(ENV_NAME, render_mode=render_mode)

    if use_complete_vision:
        env = CompleteVisionWrapper(env, use_feature_channels=use_feature_channels)

    return env
```

**Update training scripts:**
- `train_progressive_improved.py`: Add `--use_feature_channels` arg
- Default to `True` for new training
- Keep `False` option for comparison

---

### Step 4: Create Visualization Tools (45 mins)

**File**: `visualize_features.py` (NEW)

**Purpose**: Debug and visualize what the agent sees

```python
import matplotlib.pyplot as plt
import numpy as np
from config import make_env

def visualize_observation(obs: np.ndarray, save_path=None):
    """
    Visualize all 8 channels of observation.

    Creates 2x4 grid showing each channel.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    channel_names = [
        'Board (Locked)', 'Active Piece', 'Holder', 'Queue',
        'Holes', 'Heights', 'Bumpiness', 'Wells'
    ]

    for i, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        ax.imshow(obs[:, :, i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def test_feature_channels(episodes=5):
    """
    Run a few episodes and visualize observations.
    """
    env = make_env(use_feature_channels=True)

    for ep in range(episodes):
        obs, info = env.reset()
        visualize_observation(obs, f'logs/visualization/ep_{ep}_initial.png')

        # Play a few steps
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if step % 10 == 0:
                visualize_observation(obs,
                    f'logs/visualization/ep_{ep}_step_{step}.png')

            if term or trunc:
                break

    env.close()

if __name__ == '__main__':
    test_feature_channels()
```

**Testing**:
- Run visualization script
- Verify all 8 channels look correct
- Check holes channel matches actual holes
- Check height map shows column heights
- Verify bumpiness shows height variations
- Confirm wells show valleys

---

### Step 5: Baseline Test (30 mins)

**File**: `tests/test_feature_channels_training.py` (NEW)

**Purpose**: Quick sanity check that training works

```python
from config import make_env
from src.agent import Agent

def test_training_short():
    """Test that training runs with new observation space."""

    # Test with feature channels
    env = make_env(use_feature_channels=True)
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=5e-4,
        batch_size=64,
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Model input: {agent.q_network}")

    # Train for 10 episodes
    for episode in range(10):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            agent.remember(obs, action, reward, next_obs, done, info, reward)

            if len(agent.memory) >= agent.min_buffer_size:
                agent.learn()

            episode_reward += reward
            obs = next_obs

        print(f"Episode {episode+1}: Reward = {episode_reward:.1f}")

    print("✅ Training test passed!")
    env.close()

if __name__ == '__main__':
    test_training_short()
```

---

### Step 6: Comparison Training (2-4 hours)

**Run parallel training experiments:**

#### Experiment A: 4-Channel (Baseline - Current)
```bash
python train_progressive_improved.py \
    --episodes 2000 \
    --use_feature_channels False \
    --experiment_name "baseline_4ch_2k" \
    --force_fresh
```

#### Experiment B: 8-Channel (New - With Features)
```bash
python train_progressive_improved.py \
    --episodes 2000 \
    --use_feature_channels True \
    --experiment_name "hybrid_8ch_2k" \
    --force_fresh
```

**Metrics to compare:**
- Lines cleared (primary metric)
- Steps per episode
- Holes (avg during play)
- Learning speed (episodes to first line clear)
- Final performance (last 100 episodes)

---

### Step 7: Analysis & Documentation (30 mins)

**Create comparison report:**

**File**: `experiments/feature_channels_comparison.md`

**Content**:
- Training curves (reward, lines, holes)
- Performance table
- Observations
- Recommendations for next steps

---

## 🧪 Testing Strategy

### Unit Tests

**File**: `tests/test_feature_heatmaps.py`

Test each heatmap function:
```python
def test_hole_heatmap():
    # Board with known holes
    board = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Row 19 (bottom)
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # Row 18 - hole at (18, 1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 17
        # ... rest zeros
    ])

    heatmap = compute_hole_heatmap(board)

    # Verify hole detected at (18, 1)
    assert heatmap[18, 1] == 1.0
    assert heatmap[17, 1] == 0.0  # Not a hole (nothing above)

def test_height_map():
    # Board with known column heights
    board = np.zeros((20, 10))
    board[15:20, 0] = 1  # Column 0: height 5
    board[18:20, 1] = 1  # Column 1: height 2

    heatmap = compute_height_map(board)

    # Verify heights normalized correctly
    assert np.allclose(heatmap[:, 0], 5/20)  # 0.25
    assert np.allclose(heatmap[:, 1], 2/20)  # 0.10

# Similar for bumpiness and wells
```

### Integration Tests

**File**: `tests/test_enhanced_wrapper.py`

```python
def test_observation_shape():
    env = make_env(use_feature_channels=True)
    obs, info = env.reset()
    assert obs.shape == (20, 10, 8)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0

def test_backward_compatibility():
    # Test 4-channel mode still works
    env = make_env(use_feature_channels=False)
    obs, info = env.reset()
    assert obs.shape == (20, 10, 4)

def test_training_compatibility():
    # Verify agent can train with new observation
    env = make_env(use_feature_channels=True)
    agent = Agent(env.observation_space, env.action_space)

    obs, _ = env.reset()
    action = agent.select_action(obs)
    next_obs, reward, term, trunc, info = env.step(action)

    agent.remember(obs, action, reward, next_obs, term or trunc, info, reward)
    if len(agent.memory) >= agent.min_buffer_size:
        loss = agent.learn()
        assert loss is not None
```

### Visual Inspection

1. **Generate sample visualizations**
   ```bash
   python visualize_features.py
   ```

2. **Check output in `logs/visualization/`**
   - Verify holes channel shows actual holes
   - Verify height map increases with taller columns
   - Verify bumpiness shows height transitions
   - Verify wells show valleys

3. **Manual review**
   - Compare visual channels with feature channels
   - Ensure consistency (holes in heatmap match empty cells on board)

---

## 📊 Success Criteria

### Phase 1: Implementation (Day 1)

✅ **Must have:**
- [ ] All 4 heatmap functions implemented
- [ ] CompleteVisionWrapper extended to 8 channels
- [ ] Observation shape: (20, 10, 8)
- [ ] All values in range [0, 1]
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Visualizations look correct

### Phase 2: Baseline Test (Day 1)

✅ **Must have:**
- [ ] 10-episode training runs without errors
- [ ] Agent can select actions
- [ ] Learning loop completes
- [ ] Memory usage acceptable (<4GB)

### Phase 3: Comparison (Day 2-3)

✅ **Success = Feature channels perform better:**
- [ ] **First line clear**: <2000 episodes (vs 741 for 4-ch)
- [ ] **Lines at 2K episodes**: >1.0/ep (vs 0.21 for 4-ch at 75K)
- [ ] **Learning speed**: 5-10x faster improvement rate
- [ ] **Holes**: <20 avg during play by 2K episodes

✅ **Minimal success = Comparable:**
- [ ] Not worse than 4-channel baseline
- [ ] Similar memory/compute costs
- [ ] Stable training (no crashes)

❌ **Failure = Feature channels worse:**
- [ ] No line clears by 2000 episodes
- [ ] Higher average holes than baseline
- [ ] Training instability or crashes

---

## 🔧 Implementation Details

### Memory Considerations

**Current 4-channel:**
- Observation: 20 × 10 × 4 = 800 values
- Memory buffer (200K): ~640 MB

**New 8-channel:**
- Observation: 20 × 10 × 8 = 1600 values
- Memory buffer (200K): ~1.28 GB

**Mitigation**: Acceptable for modern systems. If needed:
- Reduce buffer size to 100K
- Use float16 instead of float32

### Compute Performance

**Feature computation per step:**
- Holes heatmap: O(200) - iterate board
- Height map: O(200) - iterate board once
- Bumpiness map: O(20) - compute from heights
- Wells map: O(200) - iterate board

**Total overhead**: ~0.5ms per step (negligible)

**Network inference:**
- Conv layers scale with input size
- 8 channels vs 4: ~2x compute
- Expected: 10-15ms vs 5-8ms per forward pass

**Still fast enough for real-time training**

---

## 🚨 Rollback Plan

### If Features Hurt Performance

**Option A: Disable feature channels**
```bash
python train_progressive_improved.py \
    --use_feature_channels False \
    --resume
```

**Option B: Try fewer channels**
- Start with just holes heatmap (5 channels total)
- Add others gradually

**Option C: Adjust feature computation**
- Different normalization
- Different heatmap style (binary vs gradient)

### If Training Unstable

**Check:**
1. Value ranges (should be 0-1)
2. NaN values in heatmaps
3. Memory leaks
4. Observation preprocessing in agent

**Fix:**
- Add clipping: `np.clip(heatmap, 0, 1)`
- Add NaN checking
- Reduce batch size
- Lower learning rate

---

## 📁 Files to Create/Modify

### New Files
- [ ] `src/feature_heatmaps.py` - Feature computation functions
- [ ] `tests/test_feature_heatmaps.py` - Unit tests
- [ ] `tests/test_enhanced_wrapper.py` - Integration tests
- [ ] `tests/test_feature_channels_training.py` - Training test
- [ ] `visualize_features.py` - Visualization tool
- [ ] `experiments/feature_channels_comparison.md` - Results

### Modified Files
- [ ] `config.py` - Extend CompleteVisionWrapper
- [ ] `train_progressive_improved.py` - Add --use_feature_channels arg
- [ ] `CLAUDE.md` - Document new feature
- [ ] `README.md` - Update with feature channels info

---

## 🎯 Execution Timeline

### Day 1 (4-6 hours)

**Morning (2-3 hours):**
- ✅ Create implementation plan (done!)
- [ ] Implement feature_heatmaps.py (30 min)
- [ ] Write unit tests (30 min)
- [ ] Extend CompleteVisionWrapper (30 min)
- [ ] Create visualization tool (45 min)

**Afternoon (2-3 hours):**
- [ ] Run visual tests (30 min)
- [ ] Test baseline training (30 min)
- [ ] Start comparison training (background)
- [ ] Document changes (30 min)

### Day 2-3 (Let experiments run)

**Training:**
- [ ] 4-channel baseline: 2000 episodes (~5 hours)
- [ ] 8-channel hybrid: 2000 episodes (~6 hours, slightly slower)

**Analysis:**
- [ ] Compare metrics
- [ ] Generate plots
- [ ] Write comparison report
- [ ] Decide next steps

### Day 4 (If successful)

**Scale up:**
- [ ] Run 10K episode training
- [ ] Monitor for expert play emergence
- [ ] Compare to 75K baseline

---

## 📈 Expected Outcomes

### Optimistic (80% chance)

- First line clear by episode 500-1000
- 5-10 lines/episode by 2000 episodes
- <15 avg holes by 2000 episodes
- Clear learning acceleration visible in curves

**Next step**: Scale to 10-20K episodes

### Realistic (15% chance)

- Marginal improvement (2-3x speedup)
- Some line clears by 2000 episodes
- Comparable hole metrics

**Next step**: Try feature branch architecture

### Pessimistic (5% chance)

- No improvement or worse
- Features add noise

**Next step**: Investigate why, try pure feature-based

---

## ✅ Checklist Before Starting

- [x] Read research analysis (done)
- [x] Understand hybrid approach (done)
- [x] Create implementation plan (done)
- [ ] Backup current code
- [ ] Create feature branch in git
- [ ] Set up experiment logging directory
- [ ] Clear schedule for 4-6 hours

---

## 🚀 Ready to Implement!

**Start with**: Step 1 - Create feature_heatmaps.py

**Test incrementally**: Don't wait until everything is done

**Visualize early**: Make sure features look right before training

**Compare fairly**: Use same hyperparameters for 4ch vs 8ch

**Document everything**: We're doing science! 🔬

Let's build this! 🎮
