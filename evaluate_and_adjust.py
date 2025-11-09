#!/usr/bin/env python3
"""
Evaluation and Adjustment Script
=================================

After training completes, this script:
1. Analyzes training results
2. Identifies issues preventing line clears
3. Suggests specific adjustments
4. Can automatically apply fixes and restart training

Goal: Achieve 10+ lines cleared in a single game
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_logs(log_dir):
    """Load training logs from directory."""
    log_file = Path(log_dir) / "episode_log.csv"

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return None

    df = pd.read_csv(log_file)
    return df

def analyze_performance(df):
    """Analyze training performance and identify issues."""
    print("=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80)

    total_episodes = len(df)
    max_lines = df['lines_cleared'].max()
    mean_lines = df['lines_cleared'].mean()
    episodes_with_lines = (df['lines_cleared'] > 0).sum()

    print(f"\n📊 Overall Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Max lines in single game: {max_lines}")
    print(f"  Mean lines per game: {mean_lines:.2f}")
    print(f"  Episodes with ANY lines: {episodes_with_lines} ({episodes_with_lines/total_episodes*100:.1f}%)")

    # Check by training phase
    print(f"\n📈 Performance by Phase:")
    phases = [
        ("Early (0-500)", 0, 500),
        ("Mid-Early (500-1000)", 500, 1000),
        ("Mid (1000-2500)", 1000, 2500),
        ("Late (2500-5000)", 2500, 5000)
    ]

    for phase_name, start, end in phases:
        if end > total_episodes:
            end = total_episodes
        if start >= total_episodes:
            continue

        phase_df = df.iloc[start:end]
        phase_max = phase_df['lines_cleared'].max()
        phase_mean = phase_df['lines_cleared'].mean()
        phase_with_lines = (phase_df['lines_cleared'] > 0).sum()

        print(f"  {phase_name:20s}: Max={phase_max:3d}, Mean={phase_mean:5.2f}, With lines={phase_with_lines:4d}")

    # Check reward trends
    print(f"\n💰 Reward Trends:")
    early_reward = df.iloc[0:500]['reward'].mean()
    late_reward = df.iloc[-500:]['reward'].mean()

    print(f"  Early (0-500): {early_reward:.1f}")
    print(f"  Late (last 500): {late_reward:.1f}")
    print(f"  Improvement: {late_reward - early_reward:+.1f}")

    # Check step trends (survival)
    print(f"\n⏱️  Survival (Steps per Episode):")
    early_steps = df.iloc[0:500]['steps'].mean()
    late_steps = df.iloc[-500:]['steps'].mean()

    print(f"  Early (0-500): {early_steps:.1f}")
    print(f"  Late (last 500): {late_steps:.1f}")
    print(f"  Change: {late_steps - early_steps:+.1f}")

    return {
        'max_lines': max_lines,
        'mean_lines': mean_lines,
        'episodes_with_lines': episodes_with_lines,
        'total_episodes': total_episodes,
        'reward_improvement': late_reward - early_reward,
        'survival_change': late_steps - early_steps
    }

def diagnose_issues(stats):
    """Diagnose why the agent isn't clearing lines."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    issues = []
    recommendations = []

    # Issue 1: No lines at all
    if stats['max_lines'] == 0:
        issues.append("❌ CRITICAL: No lines cleared in ANY game")
        recommendations.append({
            'issue': 'Never clearing lines',
            'likely_cause': 'Agent not learning to stack pieces properly',
            'solutions': [
                'Increase training episodes (try 10,000)',
                'Add intermediate rewards (height reduction, fewer holes)',
                'Reduce epsilon decay (explore more)',
                'Try Dueling DQN architecture'
            ]
        })

    # Issue 2: Very few lines
    elif stats['max_lines'] < 5:
        issues.append(f"⚠️  Low performance: Max {stats['max_lines']} lines")
        recommendations.append({
            'issue': 'Some lines cleared but very few',
            'likely_cause': 'Learning is happening but very slow',
            'solutions': [
                'Continue training to 10,000 episodes',
                'Add dense reward shaping (penalize holes/height)',
                'Increase line clear bonus (try +200 or +500)',
                'Adjust epsilon schedule (slower decay)'
            ]
        })

    # Issue 3: Reasonable progress but not 10 lines
    elif stats['max_lines'] < 10:
        issues.append(f"✓ Good progress: Max {stats['max_lines']} lines (target: 10)")
        recommendations.append({
            'issue': 'Close to target',
            'likely_cause': 'Learning is working, needs more time',
            'solutions': [
                'Continue training to 7,500-10,000 episodes',
                'Fine-tune epsilon (reduce to 0.01)',
                'Try Dueling DQN for better value estimation',
                'Add small penalty for game-over'
            ]
        })

    # Issue 4: Not improving over time
    if stats['reward_improvement'] < 10:
        issues.append("⚠️  Reward not improving (agent not learning)")
        recommendations.append({
            'issue': 'Flat learning curve',
            'likely_cause': 'Learning rate too low or exploration issues',
            'solutions': [
                'Increase learning rate (try 0.0005)',
                'Verify replay buffer is filling (check memory size)',
                'Reduce min_memory_size to start learning earlier',
                'Add reward for partial progress (fewer holes, etc.)'
            ]
        })

    # Issue 5: Dying too quickly
    if stats['survival_change'] < -10:
        issues.append("❌ Agent dying faster over time")
        recommendations.append({
            'issue': 'Negative survival trend',
            'likely_cause': 'Agent learning wrong behavior',
            'solutions': [
                'Check reward function (ensure no negative per-step)',
                'Add survival bonus',
                'Increase penalty for early death',
                'Reset epsilon periodically to re-explore'
            ]
        })

    # Print issues
    print("\n🔍 Issues Found:")
    for issue in issues:
        print(f"  {issue}")

    # Print recommendations
    print("\n💡 Recommended Adjustments:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']}")
        print(f"   Likely cause: {rec['likely_cause']}")
        print(f"   Solutions:")
        for sol in rec['solutions']:
            print(f"     • {sol}")

    return recommendations

def suggest_next_config(stats, recommendations):
    """Suggest concrete next training configuration."""
    print("\n" + "=" * 80)
    print("NEXT TRAINING CONFIGURATION")
    print("=" * 80)

    config = {
        'episodes': 5000,
        'learning_rate': 0.0001,
        'epsilon_decay': 0.9995,
        'line_clear_bonus': 100,
        'model_type': 'fc_dqn'
    }

    # Adjust based on performance
    if stats['max_lines'] == 0:
        # No lines at all - need more exploration and time
        config['episodes'] = 10000
        config['epsilon_decay'] = 0.999  # Slower decay
        config['line_clear_bonus'] = 200  # Bigger incentive
        print("\n🎯 Strategy: More exploration + stronger incentives")

    elif stats['max_lines'] < 5:
        # Some lines but very few
        config['episodes'] = 7500
        config['learning_rate'] = 0.0002  # Faster learning
        config['line_clear_bonus'] = 150
        config['model_type'] = 'fc_dueling_dqn'  # Try better architecture
        print("\n🎯 Strategy: Better architecture + more training")

    elif stats['max_lines'] < 10:
        # Close to target
        config['episodes'] = 7500
        config['epsilon_decay'] = 0.9997  # Fine-tune exploration
        print("\n🎯 Strategy: Fine-tuning to reach 10 lines")

    print(f"\n📝 Recommended Configuration:")
    print(f"  --episodes {config['episodes']}")
    print(f"  --lr {config['learning_rate']}")
    print(f"  --epsilon_decay {config['epsilon_decay']}")
    print(f"  --model_type {config['model_type']}")
    print(f"\nIn train_feature_vector.py simple_reward():")
    print(f"  Line clear bonus: {config['line_clear_bonus']} (change 'lines * 100' to 'lines * {config['line_clear_bonus']}')")

    return config

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_and_adjust.py <log_directory>")
        print("Example: python evaluate_and_adjust.py logs/feature_5k_bugfixed")
        sys.exit(1)

    log_dir = sys.argv[1]

    # Load and analyze
    df = load_training_logs(log_dir)
    if df is None:
        sys.exit(1)

    stats = analyze_performance(df)
    recommendations = diagnose_issues(stats)
    config = suggest_next_config(stats, recommendations)

    # Check if we reached the goal
    print("\n" + "=" * 80)
    if stats['max_lines'] >= 10:
        print("🎉 SUCCESS! Agent cleared 10+ lines in a single game!")
        print(f"   Max lines achieved: {stats['max_lines']}")
    else:
        print(f"🎯 Target: 10 lines | Current: {stats['max_lines']} lines")
        print(f"   Continue iterating with adjustments above")
    print("=" * 80)

if __name__ == "__main__":
    main()
