#!/usr/bin/env python3
"""
Training Monitor
================

Quick check of training progress while it's running.
"""

import sys
from pathlib import Path
import pandas as pd

def monitor(log_dir):
    """Monitor training progress."""
    log_file = Path(log_dir) / "episode_log.csv"

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print("   Training may not have started yet or log_dir is wrong")
        return

    df = pd.read_csv(log_file)
    total = len(df)

    if total == 0:
        print("No episodes completed yet")
        return

    print("=" * 80)
    print(f"TRAINING PROGRESS: {log_dir}")
    print("=" * 80)

    # Latest episode
    latest = df.iloc[-1]
    print(f"\n📊 Latest Episode ({int(latest['episode'])}):")
    print(f"   Steps: {int(latest['steps'])}")
    print(f"   Lines cleared: {int(latest['lines_cleared'])}")
    print(f"   Reward: {latest['reward']:.1f}")
    print(f"   Epsilon: {latest['epsilon']:.3f}")

    # Overall stats
    max_lines = df['lines_cleared'].max()
    total_lines = df['lines_cleared'].sum()
    episodes_with_lines = (df['lines_cleared'] > 0).sum()

    print(f"\n🎯 Overall Performance:")
    print(f"   Total episodes: {total}")
    print(f"   Max lines (best game): {int(max_lines)}")
    print(f"   Total lines cleared: {int(total_lines)}")
    print(f"   Episodes with lines: {episodes_with_lines}")

    # Recent performance (last 100 or less)
    recent_n = min(100, total)
    recent = df.iloc[-recent_n:]
    recent_max = recent['lines_cleared'].max()
    recent_mean = recent['lines_cleared'].mean()
    recent_lines = recent['lines_cleared'].sum()

    print(f"\n📈 Recent Performance (last {recent_n} episodes):")
    print(f"   Max lines: {int(recent_max)}")
    print(f"   Mean lines: {recent_mean:.2f}")
    print(f"   Total lines: {int(recent_lines)}")
    print(f"   Mean reward: {recent['reward'].mean():.1f}")

    # Milestones
    print(f"\n✨ Milestones:")
    if max_lines >= 10:
        print(f"   ✅ GOAL REACHED: {int(max_lines)} lines cleared!")
    elif max_lines >= 5:
        print(f"   ✓ Halfway there: {int(max_lines)} / 10 lines")
    elif max_lines >= 1:
        print(f"   ✓ First line cleared! ({int(max_lines)} max)")
    else:
        print(f"   ⏳ No lines yet (this is normal early in training)")

    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <log_directory>")
        print("Example: python monitor_training.py logs/feature_5k_bugfixed")
        sys.exit(1)

    monitor(sys.argv[1])
