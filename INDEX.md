# Tetris-Gym2 Documentation Index

**Welcome!** This index helps you navigate the project documentation.

---

## 🚀 Getting Started

**New to this project?** Start here:

1. **[README.md](README.md)** - Project overview, quick start, installation
2. **[docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md)** - Understanding the hybrid DQN implementation
3. **[CLAUDE.md](CLAUDE.md)** - If you're Claude Code, read this for project guidance

---

## 📚 Documentation Structure

### Core Documentation (Root)

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Project overview, installation, quick start | All users |
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions and project context | Claude Code |
| [INDEX.md](INDEX.md) | This file - documentation navigation | All users |

### Architecture Documentation

Located in `docs/architecture/`

| File | Description |
|------|-------------|
| [hybrid-dqn.md](docs/architecture/hybrid-dqn.md) | Complete guide to hybrid dual-branch DQN architecture, usage, and expected results |

### Research & Analysis

Located in `docs/research/`

| File | Description |
|------|-------------|
| [dqn-research.md](docs/research/dqn-research.md) | Research findings on DQN approaches for Tetris (visual-only vs feature-based) |
| [curriculum-analysis.md](docs/research/curriculum-analysis.md) | Analysis of 5-stage progressive curriculum effectiveness |
| [decision-making.md](docs/research/decision-making.md) | Technical deep-dive: How Q-values work and how the agent decides actions |

### Project History

Located in `docs/history/`

| File | Description |
|------|-------------|
| [project-history.md](docs/history/project-history.md) | Complete project timeline, experiments, learnings, and evolution |
| [implementation-plan.md](docs/history/implementation-plan.md) | Original plan for feature channel implementation |

### Training Results

Located in `docs/history/training-results/`

| File | Description |
|------|-------------|
| [13k-analysis.md](docs/history/training-results/13k-analysis.md) | Analysis of 13,000 episode training run |
| [15k-analysis.md](docs/history/training-results/15k-analysis.md) | Analysis of 15,000 episode training run |

### Critical Bug Fixes (Historical)

Located in `reports/archive/`

| File | Description | Importance |
|------|-------------|------------|
| [CRITICAL_FIXES_APPLIED.md](reports/archive/CRITICAL_FIXES_APPLIED.md) | Dropout rate and train/eval mode fixes | **CRITICAL** |
| [HOLE_MEASUREMENT_FIX.md](reports/archive/HOLE_MEASUREMENT_FIX.md) | How holes are measured (during play vs game-over) | **CRITICAL** |

---

## 🎯 Documentation by Use Case

### "I want to train a model"

1. [README.md](README.md) - Installation and environment setup
2. [docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md) - Choose model type and understand options
3. [CLAUDE.md](CLAUDE.md#common-commands) - Training commands reference

### "I want to understand the architecture"

1. [docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md) - Architecture overview
2. [CLAUDE.md](CLAUDE.md#architecture-overview) - Technical details on 8-channel system
3. [docs/research/dqn-research.md](docs/research/dqn-research.md) - Why this architecture was chosen

### "I want to understand training results"

1. [docs/history/training-results/15k-analysis.md](docs/history/training-results/15k-analysis.md) - Latest results
2. [docs/history/training-results/13k-analysis.md](docs/history/training-results/13k-analysis.md) - Earlier results
3. [docs/history/project-history.md](docs/history/project-history.md) - Complete training history

### "I'm debugging an issue"

1. [CLAUDE.md](CLAUDE.md#known-issues--solutions) - Known issues and solutions
2. [CLAUDE.md](CLAUDE.md#debugging-tips) - Debugging tips
3. [docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md#troubleshooting) - Architecture-specific troubleshooting
4. [reports/archive/](reports/archive/) - Historical bug fixes

### "I want to understand how the agent learns"

1. [docs/research/decision-making.md](docs/research/decision-making.md) - How Q-values and decisions work
2. [docs/research/curriculum-analysis.md](docs/research/curriculum-analysis.md) - Progressive curriculum
3. [CLAUDE.md](CLAUDE.md#architecture-overview) - Reward shaping details

### "I want to see the project history"

1. [docs/history/project-history.md](docs/history/project-history.md) - Complete timeline and learnings
2. [docs/history/implementation-plan.md](docs/history/implementation-plan.md) - Feature implementation plan
3. [docs/history/training-results/](docs/history/training-results/) - Training run analyses

---

## 📁 Code Organization

### Active Scripts (Root)

| File | Purpose |
|------|---------|
| `train_progressive_improved.py` | **CURRENT** training script (use this one) |
| `test_hybrid_model.py` | Test hybrid architecture before training |
| `evaluate.py` | Evaluate trained models |
| `monitor_training.py` | Monitor ongoing training |
| `visualize_features.py` | Visualize 8-channel observations |
| `config.py` | Environment configuration |

### Deprecated Scripts

Located in `archive_scripts/` - **DO NOT USE**

| File | Status |
|------|--------|
| `train.py` | Deprecated (use `train_progressive_improved.py`) |
| `train_progressive.py` | Deprecated (use `train_progressive_improved.py`) |
| `debug_*.py` | Old debug scripts |

### Source Code

Located in `src/`

| File | Purpose |
|------|---------|
| `model_hybrid.py` | **Hybrid dual-branch DQN** (recommended) |
| `model.py` | Standard DQN and Dueling DQN |
| `agent.py` | DQN agent with adaptive epsilon |
| `progressive_reward_improved.py` | 5-stage curriculum |
| `reward_shaping.py` | Core reward functions |
| `feature_heatmaps.py` | Feature channel computation |
| `utils.py` | Logging and utilities |

### Tests

Located in `tests/`

Run tests to verify setup:
```bash
python tests/test_feature_heatmaps.py
python tests/test_feature_channels_training.py
```

---

## 🔍 Quick Reference

### Key Concepts

- **8-Channel Observation**: Visual (4) + Feature (4) channels
  - Visual: Board, Active piece, Holder, Queue
  - Features: Holes, Heights, Bumpiness, Wells

- **Hybrid Dual-Branch DQN**: Separate CNNs for visual and feature channels

- **Progressive Curriculum**: 5-stage reward shaping (Foundation → Clean placement → Spreading → Clean spreading → Line clearing)

- **Critical Fixes Applied**:
  - Dropout: 0.3 → 0.1 (RL-appropriate)
  - Train/eval modes: Properly set in agent
  - Hole measurement: Track during play, not just at game-over

### Performance Metrics

From latest training (15K episodes):
- Lines/episode: ~0.7 (improved over visual-only baseline of 0.21)
- Holes during play: 22-35 (target: <15)
- Training: Ongoing, may require 30K-50K episodes for full convergence

---

## 📊 Document Status

**Last Updated**: 2025-11-09

**Recent Changes**:
- Reorganized all documentation into `docs/` directory
- Removed duplicate files (Fix.md)
- Archived deprecated training scripts
- Created this index for easier navigation

**Documentation Health**:
- ✅ Core docs up-to-date (README, CLAUDE, INDEX)
- ✅ Architecture docs current
- ✅ Training results documented
- ⚠️ Some guides in `docs/guides/` are placeholders (future expansion)

---

## 🤝 Contributing to Documentation

When adding new documentation:

1. **Place files in appropriate `docs/` subdirectory**:
   - Architecture guides → `docs/architecture/`
   - Research/analysis → `docs/research/`
   - Training guides → `docs/guides/`
   - Historical content → `docs/history/`

2. **Update this INDEX.md** with links to new files

3. **Update CLAUDE.md** if changes affect AI assistant guidance

4. **Use clear, descriptive filenames**: `kebab-case.md`

---

## ❓ Questions?

- **General usage**: Start with [README.md](README.md)
- **Architecture questions**: See [docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md)
- **Training issues**: Check [CLAUDE.md](CLAUDE.md#debugging-tips)
- **Historical context**: See [docs/history/project-history.md](docs/history/project-history.md)

---

*This index is maintained to help navigate the growing documentation. If you find broken links or missing content, please update this file.*
