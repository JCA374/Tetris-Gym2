# Tetris DQN - Feature Vector Approach

A Deep Q-Network (DQN) implementation for training AI agents to play Tetris using **feature vectors** (direct scalar features, not images) with the **Tetris Gymnasium** environment.

**Current Approach**: Feature Vector DQN - extracting 17 scalar features from the game board and feeding them to a simple fully-connected network. This is the proven approach used by 90%+ of successful Tetris DQN implementations.

## 🎯 Why Feature Vectors?

Based on competitive analysis of Tetris DQN implementations:
- **100-1000x better sample efficiency** than image-based approaches
- **Direct representation** of game state (holes, heights, bumpiness)
- **Smaller networks** (~46K parameters vs 1.2M for CNNs)
- **Proven approach** - vast majority of successful implementations use features

See [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) for detailed research findings.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Tetris-Gym2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Quick test (100 episodes, ~1 minute)
./venv/bin/python train_feature_vector.py --episodes 100 --log_freq 10

# Recommended training (5,000 episodes, ~3-5 hours)
./venv/bin/python train_feature_vector.py --episodes 5000 --model_type fc_dqn

# Full training (20,000 episodes, ~10-15 hours)
./venv/bin/python train_feature_vector.py --episodes 20000 --model_type fc_dqn

# Try Dueling DQN (may be 10-20% better)
./venv/bin/python train_feature_vector.py --episodes 5000 --model_type fc_dueling_dqn

# Force fresh start (ignore checkpoints)
./venv/bin/python train_feature_vector.py --episodes 5000 --force_fresh
```

### Analysis

```bash
# Analyze completed training run
python analyze_training.py logs/feature_vector_fc_dqn_<timestamp>
```

## 📊 Expected Performance

| Episode Range | Lines/Episode | Status |
|---------------|--------------|--------|
| 0-500 | 0-1 | Learning survival |
| 500-1,000 | 1-5 | Basic line clearing |
| 1,000-2,000 | 5-20 | Consistent clearing |
| 2,000-5,000 | 20-100 | Advanced strategy |
| 5,000+ | 100-1,000+ | Expert performance |

**Success at 5,000 episodes:**
- ✅ Lines/episode > 50
- ✅ First 100+ line episode before episode 3,000
- ✅ Consistent line clearing
- ✅ Reward trending upward

## 🏗️ Architecture Overview

### Feature Extraction (17 Features)
From board state → 17 normalized scalars:
- Aggregate height, max/min/std height
- Holes count
- Bumpiness (height differences)
- Wells (valley depths)
- Column heights (10 values)

See [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md) for complete details.

### Neural Network
Simple fully-connected network (NO CNNs):
```
Input: 17 features
  ↓
FC: 256 → 128 → 64 (with ReLU, Dropout 0.1)
  ↓
Output: 8 Q-values (one per action)

Total: ~46K parameters
```

### Reward Function
```python
reward = 1.0                      # Positive survival reward
       + lines_cleared * 100      # Huge bonus for lines
       - holes * 2.0              # Penalize holes
       - aggregate_height * 0.1   # Slight penalty for height
```

**Critical**: Always use positive survival reward, never negative per-step penalty!

## 📁 Project Structure

```
Tetris-Gym2/
├── README.md                           # This file
├── FEATURE_VECTOR_GUIDE.md            # Complete implementation guide
├── COMPETITIVE_ANALYSIS.md            # Why features beat CNNs
├── LOGGING_GUIDE.md                   # Monitoring and analysis
├── PROJECT_LOG.md                     # Project history and changes
├── CLAUDE.md                          # AI assistant guidance
├── INDEX.md                           # Documentation navigation
│
├── train_feature_vector.py            # Main training script
├── analyze_training.py                # Post-training analysis
├── evaluate.py                        # Model evaluation
├── requirements.txt                   # Python dependencies
│
├── src/                               # Core library
│   ├── agent.py                       # DQN agent
│   ├── feature_vector.py              # Feature extraction (17 scalars)
│   ├── model_fc.py                    # FC DQN models (current)
│   ├── env_feature_vector.py          # Feature vector wrapper
│   └── utils.py                       # Logging, plotting
│
├── archive_files/                     # Archived hybrid CNN implementation
├── logs/                              # Training logs (auto-created)
└── models/                            # Saved checkpoints (auto-created)
```

## 🔧 Configuration

Default hyperparameters (in `train_feature_vector.py`):
- Learning rate: 0.0001
- Gamma (discount): 0.99
- Batch size: 64
- Memory size: 100,000 transitions
- Epsilon: 1.0 → 0.05 (adaptive decay)
- Target network update: every 1,000 steps

Override with command-line args:
```bash
./venv/bin/python train_feature_vector.py \
    --episodes 10000 \
    --lr 0.0005 \
    --batch_size 128 \
    --epsilon_decay 0.999
```

## 🐛 Troubleshooting

### Agent not learning (zero lines after 1,000 episodes)
- Check reward function (should be positive for survival)
- Verify epsilon decaying properly (check console output)
- Ensure replay buffer filling (needs 1,000 transitions to start)
- See [CLAUDE.md](CLAUDE.md#debugging-tips) for detailed tips

### Import errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Model loading errors
- Feature vector models incompatible with CNN models
- Check model architecture matches (input_size=17, output_size=8)

## 📚 Documentation

**Essential Reading:**
- [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md) - Complete implementation guide
- [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) - Why this approach works
- [LOGGING_GUIDE.md](LOGGING_GUIDE.md) - How to monitor training
- [PROJECT_LOG.md](PROJECT_LOG.md) - Project history and bug fixes
- [CLAUDE.md](CLAUDE.md) - Developer guidance and patterns

**Historical Context:**
- [CLEANUP_PLAN.md](CLEANUP_PLAN.md) - What was archived and why
- `archive_files/` - Previous hybrid CNN implementation

## 🎯 Action Space

Tetris Gymnasium v0.3.0 actions:
```
0: LEFT, 1: RIGHT, 2: DOWN, 3: ROTATE_CW,
4: ROTATE_CCW, 5: HARD_DROP, 6: SWAP, 7: NOOP
```

## 🔬 Key Learnings

### Critical Bug Fixes (see PROJECT_LOG.md)

**Epsilon Decay Bug (Nov 2025):**
- Agent initialized without `max_episodes` parameter
- Caused epsilon to decay for 25K episodes when training only 5-6K
- Result: Agent stuck exploring (ε=0.59 at episode 6000 instead of 0.05)
- **Fix**: Pass `max_episodes=args.episodes` to Agent

**Broken Reward Function (Nov 2025):**
- Original used negative per-step penalty (-0.1 per step)
- Taught agent optimal strategy was dying immediately
- **Fix**: Use positive survival reward (+1.0 per step)

### Best Practices
- Always use positive survival rewards
- Let epsilon decay properly for episode count
- Monitor lines cleared as primary success metric
- Feature vectors are 100-1000x more sample efficient than images

## 📖 References

- [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium) - Modern Tetris RL environment
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581) - Dueling Network Architectures

## 🤝 Contributing

Contributions welcome! Focus areas:
- Feature engineering (new board features)
- Reward function improvements
- Hyperparameter optimization
- Documentation enhancements

## 📄 License

MIT License

---

**Ready to train?** Start with the quick test, then scale up to 5,000+ episodes for real results!

```bash
./venv/bin/python train_feature_vector.py --episodes 5000
```
