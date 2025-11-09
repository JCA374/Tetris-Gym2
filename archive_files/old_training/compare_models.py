"""
Model Comparison and Evaluation Framework

This script compares different Tetris DQN implementations:
- Simple feature-based DQN (baseline)
- Standard CNN DQN
- Hybrid dual-branch DQN

Metrics compared:
- Training efficiency (episodes to milestones)
- Sample efficiency (lines cleared per episode)
- Computational cost (training time, parameters)
- Final performance (max lines, average performance)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_logs(log_dir):
    """
    Load training logs from a directory.

    Args:
        log_dir: Path to log directory

    Returns:
        Dictionary with training metrics
    """
    log_file = os.path.join(log_dir, "training_log.json")

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return None

    with open(log_file, 'r') as f:
        data = json.load(f)

    return {
        'episodes': data.get('episodes', []),
        'rewards': data.get('rewards', []),
        'lines': data.get('lines_cleared', []),
        'steps': data.get('steps', []),
        'epsilon': data.get('epsilon', []),
        'pieces': data.get('pieces_placed', []),
    }


def calculate_metrics(logs):
    """
    Calculate summary metrics from training logs.

    Args:
        logs: Dictionary of training logs

    Returns:
        Dictionary of calculated metrics
    """
    if logs is None:
        return None

    lines = np.array(logs['lines'])
    steps = np.array(logs['steps'])
    rewards = np.array(logs['rewards'])

    metrics = {
        # Performance metrics
        'max_lines': float(np.max(lines)) if len(lines) > 0 else 0,
        'avg_lines_final_100': float(np.mean(lines[-100:])) if len(lines) >= 100 else 0,
        'avg_lines_all': float(np.mean(lines)) if len(lines) > 0 else 0,

        # Learning speed metrics
        'episodes_to_10_lines': int(np.argmax(lines >= 10)) if np.any(lines >= 10) else len(lines),
        'episodes_to_50_lines': int(np.argmax(lines >= 50)) if np.any(lines >= 50) else len(lines),
        'episodes_to_100_lines': int(np.argmax(lines >= 100)) if np.any(lines >= 100) else len(lines),

        # Survival metrics
        'max_steps': float(np.max(steps)) if len(steps) > 0 else 0,
        'avg_steps_final_100': float(np.mean(steps[-100:])) if len(steps) >= 100 else 0,

        # Reward metrics
        'max_reward': float(np.max(rewards)) if len(rewards) > 0 else 0,
        'avg_reward_final_100': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else 0,

        # Total stats
        'total_episodes': len(lines),
        'total_lines': int(np.sum(lines)),
    }

    return metrics


def compare_models(model_dirs, names=None, output_dir="comparison_results"):
    """
    Compare multiple models.

    Args:
        model_dirs: List of log directories to compare
        names: List of model names (optional, defaults to dir names)
        output_dir: Where to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)

    if names is None:
        names = [os.path.basename(os.path.normpath(d)) for d in model_dirs]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Load all logs
    all_logs = {}
    all_metrics = {}

    for name, log_dir in zip(names, model_dirs):
        print(f"\nLoading: {name}")
        print(f"  Log dir: {log_dir}")

        logs = load_training_logs(log_dir)
        if logs is not None:
            metrics = calculate_metrics(logs)
            all_logs[name] = logs
            all_metrics[name] = metrics
            print(f"  ✓ Loaded {len(logs['episodes'])} episodes")
        else:
            print(f"  ✗ Failed to load logs")

    if not all_metrics:
        print("\n❌ No valid logs found!")
        return

    # Print comparison table
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} " + " ".join(f"{name:>15}" for name in names))
    print("-" * (30 + 16 * len(names)))

    metric_names = [
        ('max_lines', 'Max Lines Cleared', '{:.0f}'),
        ('avg_lines_final_100', 'Avg Lines (final 100)', '{:.2f}'),
        ('avg_lines_all', 'Avg Lines (all)', '{:.2f}'),
        ('max_steps', 'Max Steps', '{:.0f}'),
        ('avg_steps_final_100', 'Avg Steps (final 100)', '{:.1f}'),
        ('episodes_to_10_lines', 'Episodes to 10 lines', '{:d}'),
        ('episodes_to_50_lines', 'Episodes to 50 lines', '{:d}'),
        ('episodes_to_100_lines', 'Episodes to 100 lines', '{:d}'),
        ('total_episodes', 'Total Episodes', '{:d}'),
        ('total_lines', 'Total Lines', '{:d}'),
    ]

    for key, label, fmt in metric_names:
        values = [all_metrics[name].get(key, 0) for name in names]
        print(f"{label:<30} " + " ".join(fmt.format(v).rjust(15) for v in values))

    # Create comparison plots
    print(f"\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Plot 1: Lines cleared over training
    plt.figure(figsize=(12, 6))
    for name in names:
        if name in all_logs:
            lines = all_logs[name]['lines']
            episodes = all_logs[name]['episodes']

            # Plot moving average
            window = 50
            if len(lines) >= window:
                smoothed = np.convolve(lines, np.ones(window)/window, mode='valid')
                plt.plot(episodes[window-1:], smoothed, label=name, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Lines Cleared (moving avg)', fontsize=12)
    plt.title('Training Progress: Lines Cleared', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_lines.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    # Plot 2: Learning speed comparison (episodes to milestones)
    milestones = [10, 50, 100, 200]
    plt.figure(figsize=(10, 6))

    x_pos = np.arange(len(milestones))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        if name in all_metrics:
            episodes_to_milestone = []
            for milestone in milestones:
                lines = np.array(all_logs[name]['lines'])
                if np.any(lines >= milestone):
                    episodes = np.argmax(lines >= milestone)
                else:
                    episodes = len(lines)
                episodes_to_milestone.append(episodes)

            offset = width * (i - len(names)/2 + 0.5)
            plt.bar(x_pos + offset, episodes_to_milestone, width, label=name)

    plt.xlabel('Milestone (lines cleared)', fontsize=12)
    plt.ylabel('Episodes Required', fontsize=12)
    plt.title('Learning Speed Comparison', fontsize=14)
    plt.xticks(x_pos, milestones)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_learning_speed.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    # Plot 3: Survival time (steps) comparison
    plt.figure(figsize=(12, 6))
    for name in names:
        if name in all_logs:
            steps = all_logs[name]['steps']
            episodes = all_logs[name]['episodes']

            # Plot moving average
            window = 50
            if len(steps) >= window:
                smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
                plt.plot(episodes[window-1:], smoothed, label=name, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps Survived (moving avg)', fontsize=12)
    plt.title('Training Progress: Survival Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_survival.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'comparison_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Saved: {metrics_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Tetris DQN models")

    parser.add_argument("--log_dirs", type=str, nargs="+", required=True,
                       help="Log directories to compare")
    parser.add_argument("--names", type=str, nargs="+",
                       help="Model names (optional, defaults to directory names)")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Output directory for comparison results")

    return parser.parse_args()


def main():
    """Main comparison function."""
    args = parse_args()

    compare_models(
        model_dirs=args.log_dirs,
        names=args.names,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Example usage if run directly
    import sys

    if len(sys.argv) == 1:
        print("\nExample usage:")
        print("  python compare_models.py --log_dirs logs/baseline_simple logs/hybrid_10k logs/dqn_10k")
        print("  python compare_models.py --log_dirs logs/exp1 logs/exp2 --names 'Simple DQN' 'Hybrid DQN'")
        print("\nRun with --help for more options")
    else:
        main()
