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

# Import environment and model
from src.env_feature_vector import make_feature_vector_env
from src.model_fc import create_feature_vector_model
from src.agent import Agent
from src.utils import TrainingLogger, make_dir

# Global variable for graceful shutdown
_training_interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _training_interrupted
    print("\n\n⚠️  Training interrupted by user. Saving progress...")
    _training_interrupted = True


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
    Simple proven reward function.

    Based on research: lines_cleared - holes - bumpiness works well.

    Args:
        env_reward: Reward from environment (usually 0)
        info: Info dict from environment

    Returns:
        float: Shaped reward
    """
    # Get metrics
    lines = info.get('number_of_lines', 0)

    # Simple reward: heavily reward line clears
    reward = lines * 100

    # Small penalty for each step to encourage efficiency
    reward -= 0.1

    return reward


def train(args):
    """Main training loop"""

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

    # Create model
    print(f"\n🧠 Creating {args.model_type} model...")
    model = create_feature_vector_model(
        model_type=args.model_type,
        input_size=input_size,
        output_size=output_size
    )

    # Create agent
    print(f"\n🤖 Initializing agent...")
    agent = Agent(
        model=model,
        action_size=output_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=100000,
        min_memory_size=1000
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

    log_dir = Path("logs") / experiment_name
    model_dir = Path("models")
    make_dir(log_dir)
    make_dir(model_dir)

    logger = TrainingLogger(log_dir, experiment_name)

    print(f"   Logging to: {log_dir}")
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

        # Episode loop
        while not done:
            # Agent selects action
            action = agent.act(state)

            # Take action
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Compute shaped reward
            reward = simple_reward(env_reward, info)

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Learn
            agent.learn()

            # Update state
            state = next_state
            total_reward += reward
            steps += 1

            # Track lines
            lines_cleared = info.get('number_of_lines', 0)

        # Episode finished
        episode_data = {
            'episode': episode + 1,
            'steps': steps,
            'reward': total_reward,
            'lines_cleared': lines_cleared,
            'epsilon': agent.epsilon,
            'memory_size': len(agent.memory)
        }

        logger.log_episode(episode_data)

        # Update best lines
        if lines_cleared > best_lines:
            best_lines = lines_cleared
            # Save best model
            torch.save(agent.model.state_dict(), model_dir / "best_model.pth")

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

        # Periodic checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = model_dir / f"checkpoint_ep{episode+1}.pth"
            torch.save({
                'episode': episode + 1,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_lines': best_lines
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Training finished
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {args.episodes}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Best performance: {best_lines} lines cleared")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 80)

    # Final save
    final_path = model_dir / "final_model.pth"
    torch.save(agent.model.state_dict(), final_path)
    print(f"✅ Final model saved: {final_path}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
