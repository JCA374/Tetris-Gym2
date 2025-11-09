#!/usr/bin/env python3
"""
Feature Vector DQN Training Script

Train Tetris agent using simple feature vector approach (17 scalar features).

This is the proven approach used by 90% of successful Tetris DQN implementations:
- Direct feature scalars (not images)
- Simple FC network (not CNNs)
- Expected: 100-1,000+ lines/episode in 2,000-6,000 episodes

Usage:
    python train_feature_vector.py --episodes 5000 --model_type fc_dqn
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

# Import environment and agent
from src.env_feature_vector import make_feature_vector_env
from src.agent import Agent
from src.utils import TrainingLogger, make_dir

# Global variables for graceful shutdown
_training_interrupted = False
_logger = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _training_interrupted, _logger
    print("\n\n⚠️  Training interrupted by user. Saving progress...")
    _training_interrupted = True

    # Save logs if available
    if _logger is not None:
        print("📊 Saving logs and plots...")
        _logger.save_logs()
        _logger.plot_progress()
        print("✅ Logs saved successfully")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tetris agent with feature vector DQN')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train (default: 5000)')
    parser.add_argument('--model_type', type=str, default='fc_dqn',
                        choices=['fc_dqn', 'fc_dueling_dqn'],
                        help='Model type (default: fc_dqn)')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Starting epsilon (default: 1.0)')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='Final epsilon (default: 0.05)')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995,
                        help='Epsilon decay rate (default: 0.9995)')

    # Training control
    parser.add_argument('--force_fresh', action='store_true',
                        help='Force fresh start (ignore existing checkpoints)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')

    # Logging
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log every N episodes (default: 10)')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='Save checkpoint every N episodes (default: 500)')

    return parser.parse_args()


def simple_reward(env_reward, info):
    """
    Improved reward function that incentivizes survival and good play.

    Key changes from broken version:
    - Positive reward for survival (not penalty)
    - Penalize bad board states (holes, height)
    - Big reward for line clears

    Args:
        env_reward: Reward from environment (usually 0)
        info: Info dict from environment

    Returns:
        float: Shaped reward
    """
    # Get metrics from info
    lines = info.get('number_of_lines', 0)

    # Base reward: positive for surviving (encourage staying alive)
    reward = 1.0

    # Huge bonus for line clears (this is the main goal)
    if lines > 0:
        reward += lines * 100

    # Penalize bad board states (if available in info)
    # Note: These might not be in info from base env, but won't hurt to check
    if 'holes' in info:
        reward -= info['holes'] * 2.0  # Penalize holes
    if 'aggregate_height' in info:
        reward -= info['aggregate_height'] * 0.1  # Slight penalty for high board

    return reward


def train(args):
    """Main training loop"""
    global _logger

    print("=" * 80)
    print("FEATURE VECTOR DQN TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Model type: {args.model_type}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epsilon: {args.epsilon_start} → {args.epsilon_end} (decay: {args.epsilon_decay})")
    print("=" * 80)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create environment
    print("\n📦 Creating environment...")
    env = make_feature_vector_env()

    # Get dimensions
    input_size = env.observation_space.shape[0]  # Should be 17
    output_size = env.action_space.n  # Should be 8

    print(f"\n✅ Environment ready:")
    print(f"   Input size: {input_size} features")
    print(f"   Output size: {output_size} actions")

    # Create agent (which will create the model internally)
    print(f"\n🤖 Initializing agent with {args.model_type} model...")
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=100000,
        min_memory_size=1000,
        model_type=args.model_type
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Setup logging
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"feature_vector_{args.model_type}_{timestamp}"

    log_dir = Path("logs")
    model_dir = Path("models")
    make_dir(log_dir)
    make_dir(model_dir)

    logger = TrainingLogger(log_dir, experiment_name)
    _logger = logger  # Set global for signal handler

    print(f"   Logging to: {logger.experiment_dir}")
    print(f"   Models saved to: {model_dir}")

    # Training loop
    print(f"\n🚀 Starting training for {args.episodes} episodes...")
    print("=" * 80)

    start_time = time.time()
    best_lines = 0
    start_episode = 0

    for episode in range(start_episode, args.episodes):
        if _training_interrupted:
            break

        # Reset environment
        state, info = env.reset()
        total_reward = 0
        steps = 0
        lines_cleared = 0
        done = False
        final_state = state.copy()
        final_raw_board = None

        # Episode loop
        while not done:
            # Agent selects action
            action = agent.select_action(state, training=True)

            # Take action
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store final board for visualization (access underlying Tetris env)
            if done and hasattr(env, 'env'):
                try:
                    # Access the base Tetris environment through wrapper chain
                    base_env = env.env if hasattr(env, 'env') else env.unwrapped
                    if hasattr(base_env, 'board'):
                        # Extract playable area: rows 0-19, cols 4-13
                        final_raw_board = base_env.board[0:20, 4:14].copy()
                except:
                    pass

            # Compute shaped reward
            reward = simple_reward(env_reward, info)

            # Store transition
            agent.remember(state, action, reward, next_state, done, info=info, original_reward=env_reward)

            # Learn
            agent.learn()

            # Update state
            state = next_state
            final_state = next_state.copy()
            total_reward += reward
            steps += 1

            # Track lines
            lines_cleared = info.get('number_of_lines', 0)

        # End episode (updates epsilon, logs stats)
        agent.end_episode(total_reward, steps, lines_cleared, original_reward=env_reward)

        # Extract feature metrics from final state (already feature vector from wrapper)
        final_features = final_state

        # Log episode with rich metrics
        logger.log_episode(
            episode=episode + 1,
            reward=total_reward,
            steps=steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_cleared,
            memory_size=len(agent.memory),
            # Feature metrics for analysis
            aggregate_height=float(final_features[0]),
            holes=float(final_features[1]),
            bumpiness=float(final_features[2]),
            wells=float(final_features[3]),
            max_height=float(final_features[14]),
            min_height=float(final_features[15]),
            std_height=float(final_features[16])
        )

        # Log board state every N episodes for debugging
        if (episode + 1) % (args.log_freq * 5) == 0 or lines_cleared > 0:
            if final_raw_board is not None:
                # Denormalize features for display
                from src.feature_vector import get_column_heights
                board_binary = (final_raw_board > 0).astype(int)
                actual_heights = get_column_heights(board_binary)

                logger.log_board_state(
                    episode=episode + 1,
                    board=board_binary,
                    reward=total_reward,
                    steps=steps,
                    lines_cleared=lines_cleared,
                    heights=actual_heights.tolist(),
                    # Feature vector (normalized)
                    features_normalized={
                        'aggregate_height': f"{final_features[0]:.3f}",
                        'holes': f"{final_features[1]:.3f}",
                        'bumpiness': f"{final_features[2]:.3f}",
                        'wells': f"{final_features[3]:.3f}",
                        'max_height': f"{final_features[14]:.3f}",
                        'std_height': f"{final_features[16]:.3f}",
                    }
                )
            else:
                # Feature-only logging if board access fails
                logger.log_board_state(
                    episode=episode + 1,
                    board=None,
                    reward=total_reward,
                    steps=steps,
                    lines_cleared=lines_cleared,
                    features_only=True,
                    features_normalized={
                        'aggregate_height': f"{final_features[0]:.3f}",
                        'holes': f"{final_features[1]:.3f}",
                        'bumpiness': f"{final_features[2]:.3f}",
                        'wells': f"{final_features[3]:.3f}",
                        'column_heights': final_features[4:14].tolist(),
                        'max_height': f"{final_features[14]:.3f}",
                        'min_height': f"{final_features[15]:.3f}",
                        'std_height': f"{final_features[16]:.3f}",
                    }
                )

        # Update best lines
        if lines_cleared > best_lines:
            best_lines = lines_cleared
            # Save best model
            torch.save(agent.q_network.state_dict(), model_dir / "best_model.pth")

        # Periodic logging
        if (episode + 1) % args.log_freq == 0:
            elapsed = time.time() - start_time
            episodes_per_sec = (episode + 1 - start_episode) / elapsed

            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"Steps: {steps:3d} | "
                  f"Lines: {lines_cleared:2d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Best: {best_lines:2d} lines | "
                  f"Speed: {episodes_per_sec:.1f} ep/s")

        # Periodic checkpoint and log saving
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = model_dir / f"checkpoint_ep{episode+1}.pth"
            torch.save({
                'episode': episode + 1,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_lines': best_lines
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")

            # Save logs periodically
            logger.save_logs()
            logger.plot_progress()
            print(f"   📊 Logs and plots updated")

    # Training finished
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {args.episodes}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Best performance: {best_lines} lines cleared")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("=" * 80)

    # Save logs and generate plots
    print(f"\n📊 Saving logs and generating plots...")
    logger.save_logs()
    logger.plot_progress()

    # Print summary statistics
    summary = logger.get_summary()
    print(f"\n📈 Training Summary:")
    print(f"   Total episodes: {summary.get('total_episodes', 0)}")
    print(f"   Mean reward: {summary.get('mean_reward', 0):.2f} ± {summary.get('std_reward', 0):.2f}")
    print(f"   Max reward: {summary.get('max_reward', 0):.2f}")
    if 'recent_mean_reward' in summary:
        print(f"   Recent mean (last 100): {summary['recent_mean_reward']:.2f}")
    print(f"   Final epsilon: {summary.get('final_epsilon', 'N/A')}")
    print(f"\n✅ Logs saved to: {log_dir}")

    # Final save
    final_path = model_dir / "final_model.pth"
    torch.save(agent.q_network.state_dict(), final_path)
    print(f"✅ Final model saved: {final_path}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
