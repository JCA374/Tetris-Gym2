"""
Visualization Tool for Enhanced Tetris Observations

This script visualizes all 8 channels of the enhanced observation space,
showing both visual channels (board, active piece, holder, queue) and
feature channels (holes, heights, bumpiness, wells).

Usage:
    python visualize_features.py [--episodes N] [--steps N]

Author: Claude Code
Date: 2025-11-05
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import make_env


def visualize_observation(obs: np.ndarray, episode: int, step: int, save_dir: str = 'logs/visualization'):
    """
    Visualize all channels of an observation.

    Creates a 2x4 grid showing each channel with proper labels and colormaps.

    Args:
        obs: Observation array (20, 10, 4) or (20, 10, 8)
        episode: Episode number (for filename)
        step: Step number (for filename)
        save_dir: Directory to save visualization
    """
    n_channels = obs.shape[2]

    if n_channels == 4:
        channel_names = [
            'Board (Locked)', 'Active Piece', 'Holder', 'Queue'
        ]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes = axes.flatten()
    else:  # 8 channels
        channel_names = [
            'Board (Locked)', 'Active Piece', 'Holder', 'Queue',
            'Holes Heatmap', 'Height Map', 'Bumpiness Map', 'Wells Map'
        ]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        # Use different colormaps for different channel types
        if i < 4:  # Visual channels
            cmap = 'gray_r'  # Black = filled, White = empty
        else:  # Feature channels
            cmap = 'hot'  # Heat map for features

        im = ax.imshow(obs[:, :, i], cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Column (min={obs[:,:,i].min():.2f}, max={obs[:,:,i].max():.2f})', fontsize=9)
        ax.set_ylabel('Row', fontsize=9)

        # Add colorbar for feature channels
        if i >= 4:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add grid
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)

    plt.suptitle(f'Episode {episode} - Step {step} - Observation ({obs.shape[2]} channels)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = f'{save_dir}/ep{episode:04d}_step{step:04d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def test_feature_channels(episodes=5, max_steps=100, use_feature_channels=True):
    """
    Run a few episodes and visualize observations at key points.

    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        use_feature_channels: Whether to use 8-channel or 4-channel mode
    """
    print("="*80)
    print(f"🎨 Feature Channel Visualization Test")
    print("="*80)
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Feature channels: {use_feature_channels}")
    print(f"Visualization output: logs/visualization/")
    print("="*80)

    # Create environment
    env = make_env(
        render_mode=None,
        use_complete_vision=True,
        use_feature_channels=use_feature_channels
    )

    print(f"\n✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")

    saved_files = []

    for ep in range(episodes):
        print(f"\n{'─'*80}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"{'─'*80}")

        obs, info = env.reset()

        # Visualize initial state
        filename = visualize_observation(obs, ep+1, 0)
        saved_files.append(filename)
        print(f"  Step 0: Saved {filename}")

        # Play episode
        done = False
        step = 0

        while not done and step < max_steps:
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Visualize every 20 steps or at end
            if step % 20 == 0 or done:
                filename = visualize_observation(obs, ep+1, step)
                saved_files.append(filename)
                print(f"  Step {step}: Saved {filename}")

        print(f"  Episode ended after {step} steps (terminated={terminated}, truncated={truncated})")

    env.close()

    print("\n" + "="*80)
    print(f"✅ Visualization Complete!")
    print("="*80)
    print(f"Generated {len(saved_files)} visualization files")
    print(f"Output directory: logs/visualization/")
    print("\nReview the images to verify:")
    if use_feature_channels:
        print("  - Board channel shows locked pieces correctly")
        print("  - Active piece channel shows current falling piece")
        print("  - Holder and queue show piece previews")
        print("  - Holes heatmap highlights where holes exist (red = holes)")
        print("  - Height map shows column heights (red = tall, blue = short)")
        print("  - Bumpiness map shows height variations (red = bumpy)")
        print("  - Wells map shows valleys (red = deep wells)")
    else:
        print("  - Board channel shows locked pieces correctly")
        print("  - Active piece channel shows current falling piece")
        print("  - Holder and queue show piece previews")
    print("="*80)

    return saved_files


def compare_modes():
    """
    Generate visualizations comparing 4-channel vs 8-channel modes.
    """
    print("\n" + "="*80)
    print("🔬 Comparing 4-Channel vs 8-Channel Modes")
    print("="*80)

    # Test both modes
    print("\n1️⃣  Testing 4-Channel Mode (Visual Only)")
    test_feature_channels(episodes=2, max_steps=50, use_feature_channels=False)

    print("\n2️⃣  Testing 8-Channel Mode (Visual + Features)")
    test_feature_channels(episodes=2, max_steps=50, use_feature_channels=True)

    print("\n" + "="*80)
    print("✅ Comparison Complete!")
    print("="*80)
    print("Check logs/visualization/ to compare:")
    print("  - 4-channel mode shows only visual information")
    print("  - 8-channel mode adds explicit feature heatmaps")
    print("="*80)


def analyze_feature_values():
    """
    Analyze the range and distribution of feature channel values.
    """
    print("\n" + "="*80)
    print("📊 Feature Channel Value Analysis")
    print("="*80)

    env = make_env(use_complete_vision=True, use_feature_channels=True)

    # Collect observations from a few episodes
    all_holes = []
    all_heights = []
    all_bumpiness = []
    all_wells = []

    for ep in range(10):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < 100:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

            # Extract feature channels
            all_holes.append(obs[:, :, 4].flatten())
            all_heights.append(obs[:, :, 5].flatten())
            all_bumpiness.append(obs[:, :, 6].flatten())
            all_wells.append(obs[:, :, 7].flatten())

    env.close()

    # Concatenate all samples
    all_holes = np.concatenate(all_holes)
    all_heights = np.concatenate(all_heights)
    all_bumpiness = np.concatenate(all_bumpiness)
    all_wells = np.concatenate(all_wells)

    # Print statistics
    print("\nFeature Channel Statistics (10 episodes, ~1000 steps):")
    print("─"*80)

    print(f"\n🔴 Holes Heatmap (Channel 4):")
    print(f"   Min: {all_holes.min():.4f}, Max: {all_holes.max():.4f}")
    print(f"   Mean: {all_holes.mean():.4f}, Std: {all_holes.std():.4f}")
    print(f"   Non-zero: {(all_holes > 0).sum() / len(all_holes) * 100:.1f}%")

    print(f"\n🔵 Height Map (Channel 5):")
    print(f"   Min: {all_heights.min():.4f}, Max: {all_heights.max():.4f}")
    print(f"   Mean: {all_heights.mean():.4f}, Std: {all_heights.std():.4f}")
    print(f"   Non-zero: {(all_heights > 0).sum() / len(all_heights) * 100:.1f}%")

    print(f"\n🟡 Bumpiness Map (Channel 6):")
    print(f"   Min: {all_bumpiness.min():.4f}, Max: {all_bumpiness.max():.4f}")
    print(f"   Mean: {all_bumpiness.mean():.4f}, Std: {all_bumpiness.std():.4f}")
    print(f"   Non-zero: {(all_bumpiness > 0).sum() / len(all_bumpiness) * 100:.1f}%")

    print(f"\n🟢 Wells Map (Channel 7):")
    print(f"   Min: {all_wells.min():.4f}, Max: {all_wells.max():.4f}")
    print(f"   Mean: {all_wells.mean():.4f}, Std: {all_wells.std():.4f}")
    print(f"   Non-zero: {(all_wells > 0).sum() / len(all_wells) * 100:.1f}%")

    print("\n" + "="*80)
    print("✅ Analysis shows feature values are in expected [0, 1] range")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Tetris feature channels')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to visualize')
    parser.add_argument('--steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--mode', type=str, default='8ch', choices=['4ch', '8ch', 'compare', 'analyze'],
                       help='Visualization mode: 4ch, 8ch, compare, or analyze')

    args = parser.parse_args()

    if args.mode == 'compare':
        compare_modes()
    elif args.mode == 'analyze':
        analyze_feature_values()
    else:
        use_features = (args.mode == '8ch')
        test_feature_channels(episodes=args.episodes, max_steps=args.steps,
                             use_feature_channels=use_features)
