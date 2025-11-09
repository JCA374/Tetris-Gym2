# Tetris-Gym2 Documentation Index

**Welcome!** This index helps you navigate the project documentation.

> **Current Approach**: Feature Vector DQN (17 scalar features) - This is the active implementation.
> **Archived Approach**: Hybrid CNN (8-channel observations) - Archived November 2025, see `archive_files/`.

---

## 🚀 Getting Started

**New to this project?** Start here:

1. **[README.md](README.md)** - Project overview, installation, quick start with feature vectors
2. **[FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md)** - Complete guide to current implementation
3. **[COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)** - Why feature vectors beat CNNs (100-1000x better)
4. **[CLAUDE.md](CLAUDE.md)** - If you're Claude Code, read this for project guidance

---

## 📚 Current Documentation (Feature Vector Approach)

### Essential Guides

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Quick start, installation, training commands | All users |
| [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md) | Complete implementation guide | Developers |
| [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) | Research: why features > images | Researchers |
| [LOGGING_GUIDE.md](LOGGING_GUIDE.md) | Monitoring and analysis guide | All users |
| [PROJECT_LOG.md](PROJECT_LOG.md) | **NEW**: Complete project history and changelog | All users |
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions | Claude Code |
| [CLEANUP_PLAN.md](CLEANUP_PLAN.md) | Archive rationale and what was moved | Developers |

### Core Implementation Files

| File | Description |
|------|-------------|
| `train_feature_vector.py` | **CURRENT** training script - use this! |
| `analyze_training.py` | Post-training analysis tool |
| `evaluate.py` | Model evaluation script |
| `src/feature_vector.py` | Feature extraction (17 scalars) |
| `src/model_fc.py` | FC DQN models (current architecture) |
| `src/env_feature_vector.py` | Feature vector wrapper |
| `src/agent.py` | DQN agent (supports both FC and CNN models) |

---

## 📖 Historical Documentation (Archived Approaches)

> **⚠️ ARCHIVED**: The following documents describe the hybrid CNN approach that was archived on 2025-11-09.
> They are preserved for historical context and research reference only.
> **For current implementation, see Feature Vector documentation above.**

### Archived Architecture Docs

Located in `docs/architecture/`

| File | Description | Status |
|------|-------------|--------|
| [docs/architecture/hybrid-dqn.md](docs/architecture/hybrid-dqn.md) | Hybrid dual-branch CNN guide | ⚠️ ARCHIVED |

### Research & Analysis (Still Useful)

Located in `docs/research/`

| File | Description | Relevance |
|------|-------------|-----------|
| [docs/research/dqn-research.md](docs/research/dqn-research.md) | DQN approaches research | ✅ Still relevant |
| [docs/research/curriculum-analysis.md](docs/research/curriculum-analysis.md) | Progressive curriculum analysis | ⚠️ For CNN approach |
| [docs/research/decision-making.md](docs/research/decision-making.md) | How Q-values and decisions work | ✅ Still relevant |

### Project History

Located in `docs/history/`

| File | Description |
|------|-------------|
| [docs/history/project-history.md](docs/history/project-history.md) | Complete timeline and experiments |
| [docs/history/implementation-plan.md](docs/history/implementation-plan.md) | Original feature channel plan |

### Training Results (Historical)

Located in `docs/history/training-results/`

| File | Description | Approach |
|------|-------------|----------|
| [docs/history/training-results/13k-analysis.md](docs/history/training-results/13k-analysis.md) | 13K episode analysis | Hybrid CNN |
| [docs/history/training-results/15k-analysis.md](docs/history/training-results/15k-analysis.md) | 15K episode analysis | Hybrid CNN |

### Critical Bug Fixes (Historical Reference)

Located in `reports/archive/`

| File | Description |
|------|-------------|
| [reports/archive/CRITICAL_FIXES_APPLIED.md](reports/archive/CRITICAL_FIXES_APPLIED.md) | Dropout and train/eval mode fixes (CNN models) |
| [reports/archive/HOLE_MEASUREMENT_FIX.md](reports/archive/HOLE_MEASUREMENT_FIX.md) | Hole measurement methodology |

---

## 🎯 Documentation by Use Case

### "I want to train a model"

1. [README.md](README.md) - Installation and setup
2. [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md) - Complete training guide
3. [CLAUDE.md](CLAUDE.md#common-commands) - Command reference

**Quick command:**
```bash
./venv/bin/python train_feature_vector.py --episodes 5000
```

### "I want to understand the current architecture"

1. [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md) - Feature vector implementation
2. [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) - Why this approach works
3. [CLAUDE.md](CLAUDE.md#architecture-overview) - Technical details

### "I want to monitor training progress"

1. [LOGGING_GUIDE.md](LOGGING_GUIDE.md) - Complete logging guide
2. [README.md](README.md#expected-performance) - Performance expectations
3. [CLAUDE.md](CLAUDE.md#performance-expectations) - Success criteria

### "I'm debugging an issue"

1. [CLAUDE.md](CLAUDE.md#known-issues--solutions) - Known issues and fixes
2. [README.md](README.md#troubleshooting) - Common problems
3. [PROJECT_LOG.md](PROJECT_LOG.md) - Recent bug fixes

**Recent critical fixes:**
- **Epsilon decay bug**: Agent not receiving `max_episodes` parameter
- **Reward function bug**: Negative per-step penalty teaching agent to die

### "I want to understand project history"

1. [PROJECT_LOG.md](PROJECT_LOG.md) - **START HERE**: Complete changelog with dates
2. [CLEANUP_PLAN.md](CLEANUP_PLAN.md) - What was archived and why
3. [docs/history/project-history.md](docs/history/project-history.md) - Detailed timeline

### "I want to see research findings"

1. [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) - Feature vectors vs CNNs
2. [docs/research/dqn-research.md](docs/research/dqn-research.md) - DQN approaches
3. [docs/research/decision-making.md](docs/research/decision-making.md) - How Q-learning works

---

## 📁 Code Organization

### Active Scripts (Root Directory)

| File | Purpose | Status |
|------|---------|--------|
| `train_feature_vector.py` | **CURRENT** training script | ✅ USE THIS |
| `analyze_training.py` | Post-training analysis | ✅ Active |
| `evaluate.py` | Model evaluation | ✅ Active |

### Archived Scripts

Located in `archive_files/` - **DO NOT USE FOR NEW TRAINING**

| Directory | Contents |
|-----------|----------|
| `archive_files/hybrid_cnn/` | Hybrid CNN models, wrappers, tests |
| `archive_files/training_scripts/` | Old training scripts |
| `archive_files/tests/` | Diagnostic tests for archived code |

### Source Code (src/)

**Active Files:**

| File | Purpose |
|------|---------|
| `feature_vector.py` | **Feature extraction** (17 scalars) - CURRENT |
| `model_fc.py` | **FC DQN models** - CURRENT |
| `env_feature_vector.py` | **Feature vector wrapper** - CURRENT |
| `agent.py` | DQN agent (supports both FC and CNN) |
| `utils.py` | Logging, plotting, utilities |

**Legacy Files (still in src/):**

| File | Purpose | Status |
|------|---------|--------|
| `model.py` | Standard/Dueling DQN (CNNs) | Legacy, kept for compatibility |
| `progressive_reward_improved.py` | 5-stage curriculum | For hybrid CNN only |
| `reward_shaping.py` | Complex reward functions | For hybrid CNN only |

---

## 🔍 Quick Reference

### Current Approach: Feature Vector DQN

**17 Features Extracted:**
- Aggregate height, max/min/std height (4 features)
- Holes count (1 feature)
- Bumpiness (1 feature)
- Wells (1 feature)
- Column heights (10 features)

**Network Architecture:**
```
Input (17) → FC(256) → FC(128) → FC(64) → Output (8)
Total: ~46K parameters
```

**Reward Function:**
```python
reward = 1.0                      # Positive survival
       + lines_cleared * 100      # Line clear bonus
       - holes * 2.0              # Hole penalty
       - aggregate_height * 0.1   # Height penalty
```

### Performance Expectations (5,000 episodes)

| Episode | Lines/Ep | Status |
|---------|----------|--------|
| 500 | 1-5 | Learning |
| 1,000 | 5-20 | Improving |
| 2,000 | 20-50 | Consistent |
| 5,000 | 50-200+ | Good performance |

---

## 📊 Document Status

**Last Updated**: 2025-11-09

**Recent Changes**:
- ✅ Created PROJECT_LOG.md with complete project history
- ✅ Updated README.md for feature vector approach
- ✅ Updated INDEX.md (this file) to mark archived docs
- ✅ Reorganized documentation to feature vector first
- ⚠️ Hybrid CNN docs marked as archived but preserved

**Documentation Health**: 🟢 Excellent
- Up-to-date core documentation
- Clear separation of current vs archived
- Comprehensive project history
- All critical bug fixes documented

---

## 🤝 Contributing to Documentation

When adding new documentation:

1. **Place in appropriate location**:
   - Current implementation → Root directory
   - Historical content → `docs/history/`
   - Research findings → `docs/research/`
   - Archived code docs → `archive_files/docs/`

2. **Update this INDEX.md** with links to new files

3. **Update CLAUDE.md** if changes affect AI guidance

4. **Update PROJECT_LOG.md** for significant changes

5. **Use clear naming**: `kebab-case.md`

---

## ⚡ Quick Actions

**Start training:**
```bash
./venv/bin/python train_feature_vector.py --episodes 5000
```

**Analyze latest run:**
```bash
python analyze_training.py logs/feature_vector_fc_dqn_<timestamp>
```

**Test feature extraction:**
```bash
python src/feature_vector.py
```

**Check epsilon decay fix:**
```bash
grep "max_episodes" train_feature_vector.py
# Should see: max_episodes=args.episodes
```

---

## ❓ Questions?

- **Getting started**: [README.md](README.md)
- **Implementation details**: [FEATURE_VECTOR_GUIDE.md](FEATURE_VECTOR_GUIDE.md)
- **Why this approach**: [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)
- **Training issues**: [CLAUDE.md](CLAUDE.md#debugging-tips)
- **Project history**: [PROJECT_LOG.md](PROJECT_LOG.md)
- **What changed**: [CLEANUP_PLAN.md](CLEANUP_PLAN.md)

---

*This index is maintained to help navigate documentation. Last major reorganization: November 2025 (Feature Vector pivot).*
