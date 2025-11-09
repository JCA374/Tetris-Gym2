# Simple Baseline DQN Guide

This guide covers the simple feature-based DQN baseline implementation, which serves as a comparison point for our complex hybrid dual-branch architecture.

## Overview

The baseline implementation follows successful Tetris DQN approaches from the literature:
- **Input:** 4-8 scalar features (not raw pixels)
- **Network:** Small feedforward (64→64→actions, ~5k parameters)
- **Reward:** Simple sparse rewards (line clears + death penalty)
- **Training:** Fast (hours instead of 15+ hours)

## Quick Start

### Train Simple Baseline

```bash
# Basic training (5,000 episodes, ~2-4 hours)
python train_baseline_simple.py --episodes 5000

# With specific settings
python train_baseline_simple.py \
    --episodes 5000 \
    --feature_set basic \
    --model_type simple_dqn \
    --reward_variant quadratic \
    --experiment_name my_baseline
```

### Compare with Hybrid DQN

```bash
# Train both models
python train_baseline_simple.py --episodes 5000 --experiment_name baseline_5k
python train_progressive_improved.py --episodes 5000 --model_type hybrid_dqn --experiment_name hybrid_5k

# Compare results
python compare_models.py --log_dirs logs/baseline_5k logs/hybrid_5k
```

## Feature Sets

The baseline uses scalar features extracted from the board state:

### Basic (4 features) - DEFAULT, Recommended
```python
[holes, bumpiness, aggregate_height, completable_rows]
```
- Most common in literature
- Fast learning
- Good performance

### Standard (6 features)
```python
[holes, bumpiness, aggregate_height, completable_rows, max_height, height_variance]
```
- Additional height statistics
- Slightly more information

### Extended (8 features)
```python
[holes, bumpiness, aggregate_height, completable_rows,
 max_height, height_variance, wells, clean_rows]
```
- All available features
- More information, slower learning

### Minimal (3 features)
```python
[holes, aggregate_height, bumpiness]
```
- Absolute minimum
- Fastest learning
- May miss important patterns

## Model Types

### Simple DQN (Default)
```
Input (n_features) → Dense(64, relu) → Dense(64, relu) → Dense(8, linear)
```
- Standard feedforward DQN
- ~4,500 parameters (4 features)
- Proven effective

### Simple Dueling DQN
```
Input → Shared layers → Value stream (scalar)
                      → Advantage stream (per-action)
Q(s,a) = V(s) + A(s,a) - mean(A)
```
- Better value estimation
- Useful when not all actions matter equally
- ~5,000 parameters

## Reward Variants

### Quadratic (Default, from literature)
```python
reward = +1 (survival)
       + (lines_cleared)² × 10  # 10, 40, 90, 160
       - 10 (death)
```
- Most common in literature
- Strong preference for multi-line clears
- **Recommended starting point**

### Exponential
```python
reward = +1 (survival)
       + 10 × 2^lines_cleared  # 20, 40, 80, 160
       - 50 (death)
```
- Even stronger multi-line preference
- Larger reward range

### Linear
```python
reward = +1 (survival)
       + lines_cleared × 40
       - 10 (death)
```
- Proportional rewards
- Less preference for Tetris clears

### Sparse
```python
reward = (lines_cleared)² × 10  # NO survival bonus
       - 50 (death)
```
- Truly sparse rewards
- Harder to learn, but may discover better strategies

### Light Penalty
```python
reward = +1 (survival)
       + (lines_cleared)² × 10
       - 0.1 × holes
       - 0.01 × max_height
       - 10 (death)
```
- Quadratic + light structure penalties
- Middle ground between simple and curriculum

### Adaptive
```python
# Survival bonus decays from 1.0 → 0.0 over 5,000 episodes
reward = survival_factor × 1.0  # Decaying
       + (lines_cleared)² × 10  # Constant
       - 10 (death)
```
- Starts dense, becomes sparse
- Tests if reward frequency matters

## Command Line Options

### Training Parameters
```bash
--episodes N           # Number of training episodes (default: 5000)
--batch_size N         # Batch size for learning (default: 32)
--memory_size N        # Replay memory size (default: 50000)
```

### Model Parameters
```bash
--feature_set SET      # minimal, basic, standard, extended (default: basic)
--model_type TYPE      # simple_dqn, simple_dueling_dqn (default: simple_dqn)
--hidden_dims N1 N2    # Hidden layer sizes (default: 64 64)
```

### Reward Parameters
```bash
--reward_variant VAR   # quadratic, exponential, linear, sparse,
                      # light_penalty, adaptive (default: quadratic)
```

### Learning Parameters
```bash
--lr RATE             # Learning rate (default: 0.001, higher than hybrid)
--gamma GAMMA         # Discount factor (default: 0.95, from literature)
--epsilon_start EPS   # Initial epsilon (default: 1.0)
--epsilon_end EPS     # Final epsilon (default: 0.05)
--epsilon_decay_fraction F  # Fraction of episodes for decay (default: 0.75)
```

### Experiment Parameters
```bash
--experiment_name NAME  # Experiment name for logs/models
--resume               # Resume from checkpoint
--force_fresh          # Force fresh start
--save_freq N          # Save every N episodes (default: 500)
--log_freq N           # Log every N episodes (default: 10)
```

## Example Experiments

### Literature Baseline
```bash
# Recreate approach from nuno-faria/tetris-ai
python train_baseline_simple.py \
    --episodes 5000 \
    --feature_set basic \
    --model_type simple_dqn \
    --hidden_dims 32 32 \
    --reward_variant quadratic \
    --gamma 0.95 \
    --experiment_name lit_baseline
```

### Minimal Experiment (Fast Testing)
```bash
# Quick test with minimal features
python train_baseline_simple.py \
    --episodes 1000 \
    --feature_set minimal \
    --reward_variant sparse \
    --experiment_name quick_test
```

### Extended Experiment
```bash
# All features, dueling architecture, adaptive rewards
python train_baseline_simple.py \
    --episodes 10000 \
    --feature_set extended \
    --model_type simple_dueling_dqn \
    --hidden_dims 128 128 \
    --reward_variant adaptive \
    --experiment_name extended_test
```

## Ablation Studies

Run systematic ablation studies to determine which components matter:

### List Available Studies
```bash
python run_ablation_study.py --list
```

### Run Architecture Ablation
```bash
# Compare: simple features vs. CNN vs. hybrid
python run_ablation_study.py --study architecture
```

### Run Reward Ablation
```bash
# Compare: different reward functions
python run_ablation_study.py --study reward
```

### Run Hyperparameter Ablation
```bash
# Test: learning rates, gamma values
python run_ablation_study.py --study hyperparameter
```

### Run All Ablations
```bash
# Run all ablation studies sequentially
python run_ablation_study.py --study all
```

### Dry Run (Preview Commands)
```bash
# See what will be run without executing
python run_ablation_study.py --study architecture --dry_run
```

## Comparison and Evaluation

### Compare Multiple Models
```bash
# Compare baseline vs. hybrid
python compare_models.py \
    --log_dirs logs/baseline_5k logs/hybrid_5k \
    --names "Simple Baseline" "Hybrid DQN" \
    --output_dir comparison_baseline_vs_hybrid
```

### Compare Multiple Baselines
```bash
# Compare different reward functions
python compare_models.py \
    --log_dirs logs/reward_quadratic logs/reward_exponential logs/reward_sparse \
    --names "Quadratic" "Exponential" "Sparse"
```

## Expected Performance

Based on literature and our analysis:

### Simple Baseline (5,000 episodes, 2-4 hours)
- **Max lines:** 50-200
- **Avg lines (final 100):** 10-50
- **Learning speed:** Fast (10+ lines by episode 1000)
- **Parameters:** ~5,000
- **Training time:** 2-4 hours

### Hybrid DQN (5,000 episodes, 6-8 hours)
- **Max lines:** Unknown (untested at 5k)
- **Avg lines (final 100):** Unknown
- **Learning speed:** Slower (10+ lines by episode 2000-3000?)
- **Parameters:** ~2,800,000
- **Training time:** 6-8 hours

## Debugging Tips

### Model not learning?
1. Check if lines cleared is increasing at all
2. Verify epsilon is decaying (should see in logs)
3. Check reward values (print first 10 episodes)
4. Try simpler reward (sparse or quadratic)
5. Increase learning rate (try 0.005)

### Learning too slow?
1. Reduce feature set (try minimal)
2. Increase learning rate (0.001 → 0.005)
3. Use exponential rewards (stronger signal)
4. Check epsilon decay (may be too slow)

### Unstable training?
1. Reduce learning rate (0.001 → 0.0005)
2. Increase batch size (32 → 64)
3. Use simpler model (32,32 hidden dims)
4. Try different reward variant

### High variance?
1. Increase memory size (50k → 100k)
2. Longer epsilon decay (0.75 → 0.85 fraction)
3. Lower epsilon_end (0.05 → 0.02)

## Files Reference

- `train_baseline_simple.py` - Main training script
- `src/model_simple.py` - Simple feature-based DQN models
- `src/feature_extraction.py` - Feature extraction from board state
- `src/reward_simple.py` - Simple reward functions
- `compare_models.py` - Model comparison utilities
- `ablation_configs.py` - Ablation study configurations
- `run_ablation_study.py` - Ablation study runner

## Next Steps

1. **Train baseline:** Start with default settings (5k episodes)
2. **Compare with hybrid:** Train hybrid DQN with same episodes
3. **Analyze results:** Use compare_models.py to see which is better
4. **Run ablations:** Systematically test components
5. **Tune best approach:** Optimize the winner

## Questions?

See `reports/DQN_TETRIS_COMPREHENSIVE_ANALYSIS.md` for in-depth analysis of DQN requirements for Tetris and comparison with literature.
