#!/usr/bin/env python3
"""
Analyze Feature Vector DQN Training Results

Loads episode logs and generates comprehensive analysis of training progress.

Usage:
    python analyze_training.py logs/feature_vector_fc_dqn_20251109_150324
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_episode_log(log_dir):
    """Load episode log CSV"""
    log_dir = Path(log_dir)

    # Try direct path first
    log_path = log_dir / "episode_log.csv"
    if not log_path.exists():
        # Try nested path (experiment_name subdir)
        subdirs = list(log_dir.glob("*/episode_log.csv"))
        if subdirs:
            log_path = subdirs[0]
        else:
            print(f"❌ Episode log not found in: {log_dir}")
            print(f"   Searched: {log_dir}/episode_log.csv")
            print(f"   Searched: {log_dir}/*/episode_log.csv")
            return None

    df = pd.read_csv(log_path)
    print(f"✅ Loaded {len(df)} episodes from {log_path}")
    return df


def analyze_learning_progress(df, window=100):
    """Analyze learning progress over time"""
    print("\n" + "=" * 80)
    print("LEARNING PROGRESS ANALYSIS")
    print("=" * 80)

    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"   Total episodes: {len(df)}")
    print(f"   Mean reward: {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
    print(f"   Mean steps: {df['steps'].mean():.1f} ± {df['steps'].std():.1f}")
    print(f"   Mean lines cleared: {df['lines_cleared'].mean():.2f} ± {df['lines_cleared'].std():.2f}")
    print(f"   Max lines cleared: {df['lines_cleared'].max():.0f}")
    print(f"   Epsilon range: {df['epsilon'].min():.4f} - {df['epsilon'].max():.4f}")

    # Recent performance (last 100 episodes)
    if len(df) >= window:
        recent = df.tail(window)
        print(f"\n📈 Recent Performance (last {window} episodes):")
        print(f"   Mean reward: {recent['reward'].mean():.2f} ± {recent['reward'].std():.2f}")
        print(f"   Mean steps: {recent['steps'].mean():.1f}")
        print(f"   Mean lines: {recent['lines_cleared'].mean():.2f}")
        print(f"   Max lines: {recent['lines_cleared'].max():.0f}")

    # Feature metrics (if available)
    if 'holes' in df.columns:
        print(f"\n🎯 Feature Metrics (final state averages):")
        print(f"   Aggregate height: {df['aggregate_height'].mean():.3f} ± {df['aggregate_height'].std():.3f}")
        print(f"   Holes: {df['holes'].mean():.3f} ± {df['holes'].std():.3f}")
        print(f"   Bumpiness: {df['bumpiness'].mean():.3f} ± {df['bumpiness'].std():.3f}")
        print(f"   Wells: {df['wells'].mean():.3f} ± {df['wells'].std():.3f}")
        print(f"   Max height: {df['max_height'].mean():.3f} ± {df['max_height'].std():.3f}")

    # Learning milestones
    print(f"\n🎯 Learning Milestones:")
    milestones = [1, 5, 10, 20, 50]
    for lines in milestones:
        episodes_with_lines = df[df['lines_cleared'] >= lines]
        if len(episodes_with_lines) > 0:
            first_episode = episodes_with_lines['episode'].min()
            count = len(episodes_with_lines)
            print(f"   ≥{lines:2d} lines: First at episode {first_episode}, achieved {count} times")


def plot_training_curves(df, output_dir):
    """Generate comprehensive training curves"""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Vector DQN Training Analysis', fontsize=16)

    # Moving average window
    window = min(50, len(df) // 10)

    # 1. Reward over time
    ax = axes[0, 0]
    ax.plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
    if len(df) >= window:
        ma = df['reward'].rolling(window=window).mean()
        ax.plot(df['episode'], ma, linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Steps over time
    ax = axes[0, 1]
    ax.plot(df['episode'], df['steps'], alpha=0.3, label='Raw')
    if len(df) >= window:
        ma = df['steps'].rolling(window=window).mean()
        ax.plot(df['episode'], ma, linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Lines cleared over time
    ax = axes[0, 2]
    ax.plot(df['episode'], df['lines_cleared'], alpha=0.3, label='Raw')
    if len(df) >= window:
        ma = df['lines_cleared'].rolling(window=window).mean()
        ax.plot(df['episode'], ma, linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Lines Cleared')
    ax.set_title('Line Clearing Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Epsilon decay
    ax = axes[1, 0]
    ax.plot(df['episode'], df['epsilon'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.grid(True, alpha=0.3)

    # 5. Holes over time (if available)
    if 'holes' in df.columns:
        ax = axes[1, 1]
        ax.plot(df['episode'], df['holes'], alpha=0.3, label='Raw')
        if len(df) >= window:
            ma = df['holes'].rolling(window=window).mean()
            ax.plot(df['episode'], ma, linewidth=2, label=f'{window}-episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Holes (normalized)')
        ax.set_title('Holes in Final State')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. Aggregate height over time (if available)
    if 'aggregate_height' in df.columns:
        ax = axes[1, 2]
        ax.plot(df['episode'], df['aggregate_height'], alpha=0.3, label='Raw')
        if len(df) >= window:
            ma = df['aggregate_height'].rolling(window=window).mean()
            ax.plot(df['episode'], ma, linewidth=2, label=f'{window}-episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Aggregate Height (normalized)')
        ax.set_title('Board Height Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Training curves saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze Feature Vector DQN training results')
    parser.add_argument('log_dir', type=str, help='Path to log directory')
    parser.add_argument('--window', type=int, default=100,
                        help='Window size for moving averages (default: 100)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        sys.exit(1)

    # Load data
    df = load_episode_log(log_dir)
    if df is None:
        sys.exit(1)

    # Analyze
    analyze_learning_progress(df, window=args.window)

    # Generate plots
    plot_training_curves(df, log_dir)

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
