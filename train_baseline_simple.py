"""
Simple Baseline Training Script for Tetris DQN

This script trains a simple feature-based DQN on Tetris using minimal reward shaping.
It serves as a baseline to compare against our complex hybrid CNN with progressive curriculum.

Key differences from train_progressive_improved.py:
- Simple 4-feature input (vs. 8-channel 20×10 images)
- Small network: 4→64→64→8 (~5k params vs. 2.8M)
- Simple reward: (lines²×10) + survival - death (vs. 10+ term curriculum)
- Linear epsilon decay (vs. adaptive 4-phase)
- Faster training expected (hours vs. 15+ hours)

Based on successful literature implementations:
- nuno-faria/tetris-ai
- ChesterHuynh/tetrisAI
"""

import argparse
import time
import numpy as np
import torch
import gymnasium as gym
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import create_env
from src.agent import Agent
from src.model_simple import create_simple_model
from src.feature_extraction import FeatureObservationWrapper
from src.reward_simple import create_simple_reward_shaper
from src.utils import Logger, make_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train simple baseline Tetris DQN")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=5000,
                       help="Number of episodes to train (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for learning (default: 32)")
    parser.add_argument("--memory_size", type=int, default=50000,
                       help="Replay memory size (default: 50000)")

    # Model parameters
    parser.add_argument("--feature_set", type=str, default="basic",
                       choices=["minimal", "basic", "standard", "extended"],
                       help="Feature set to use (default: basic=4 features)")
    parser.add_argument("--model_type", type=str, default="simple_dqn",
                       choices=["simple_dqn", "simple_dueling_dqn"],
                       help="Model architecture (default: simple_dqn)")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64],
                       help="Hidden layer sizes (default: 64 64)")

    # Reward parameters
    parser.add_argument("--reward_variant", type=str, default="quadratic",
                       choices=["quadratic", "exponential", "linear", "sparse",
                               "light_penalty", "adaptive"],
                       help="Reward function variant (default: quadratic)")

    # Learning parameters
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001, higher than hybrid)")
    parser.add_argument("--gamma", type=float, default=0.95,
                       help="Discount factor (default: 0.95, from literature)")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Initial epsilon (default: 1.0)")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                       help="Final epsilon (default: 0.05)")
    parser.add_argument("--epsilon_decay_fraction", type=float, default=0.75,
                       help="Fraction of episodes for epsilon decay (default: 0.75)")

    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default="baseline_simple",
                       help="Experiment name for logs/models")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--force_fresh", action="store_true",
                       help="Force fresh start, ignore checkpoint")

    # Save/log parameters
    parser.add_argument("--save_freq", type=int, default=500,
                       help="Save checkpoint every N episodes (default: 500)")
    parser.add_argument("--log_freq", type=int, default=10,
                       help="Log progress every N episodes (default: 10)")

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("SIMPLE BASELINE TETRIS DQN TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Feature set: {args.feature_set}")
    print(f"Model: {args.model_type}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Reward: {args.reward_variant}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print("=" * 80)

    # Setup directories
    MODEL_DIR = f"models/{args.experiment_name}/"
    LOG_DIR = f"logs/{args.experiment_name}/"
    make_dir(MODEL_DIR)
    make_dir(LOG_DIR)

    # Create environment with feature wrapper
    print("\nSetting up environment...")
    base_env = create_env(render_mode=None, use_complete_vision=False)
    env = FeatureObservationWrapper(base_env, feature_set=args.feature_set)
    print(f"✓ Environment ready")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Calculate epsilon decay for linear schedule
    epsilon_decay_episodes = int(args.epsilon_decay_fraction * args.episodes)
    epsilon_linear_step = (args.epsilon_start - args.epsilon_end) / epsilon_decay_episodes

    print(f"\n Epsilon schedule:")
    print(f"  Linear decay: {args.epsilon_start:.2f} → {args.epsilon_end:.2f}")
    print(f"  Decay episodes: {epsilon_decay_episodes} ({args.epsilon_decay_fraction*100:.0f}% of total)")
    print(f"  Step per episode: -{epsilon_linear_step:.6f}")

    # Create agent with simple model
    print("\nInitializing agent...")

    # Monkey-patch the agent's model creation to use simple models
    original_create_model = Agent.__init__

    def patched_init(self, obs_space, action_space, **kwargs):
        # Remove model_type from kwargs and save it
        model_type = kwargs.pop('model_type', 'simple_dqn')

        # Call original init
        original_create_model(self, obs_space, action_space, model_type="dqn", **kwargs)

        # Replace networks with simple models
        self.q_network = create_simple_model(
            obs_space, action_space, model_type=model_type,
            hidden_dims=args.hidden_dims, is_target=False
        ).to(self.device)

        self.target_network = create_simple_model(
            obs_space, action_space, model_type=model_type,
            hidden_dims=args.hidden_dims, is_target=True
        ).to(self.device)

        # Re-initialize optimizer with new network
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Override epsilon decay to linear
        self.epsilon_decay_method = "linear"
        self.epsilon_linear_step = epsilon_linear_step

    Agent.__init__ = patched_init

    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        min_memory_size=1000,
        target_update=1000,
        model_type=args.model_type,
        reward_shaping="none",  # Use external simple shaper
        max_episodes=args.episodes
    )

    # Restore original init
    Agent.__init__ = original_create_model

    print(f"✓ Agent initialized")

    # Create simple reward shaper
    print("\nInitializing reward shaper...")
    reward_shaper = create_simple_reward_shaper(variant=args.reward_variant)
    print(f"✓ Reward shaper ready")

    # Create logger
    logger = Logger(LOG_DIR, experiment_name=args.experiment_name)

    # Resume from checkpoint if requested
    start_episode = 0
    if not args.force_fresh and args.resume:
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            print(f"\n✓ Resumed from episode {start_episode}")
        else:
            print(f"\n⚠ No checkpoint found, starting fresh")

    # Training metrics
    lines_cleared_total = 0
    pieces_placed_total = 0

    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    start_time = time.time()

    try:
        for episode in range(start_episode, args.episodes):
            # Reset environment
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_lines = 0
            pieces_placed = 0

            reward_shaper.update_episode(episode)
            reward_shaper.reset()

            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(obs, training=True)

                # Step environment
                next_obs, raw_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track pieces
                pieces_placed += 1

                # Update info
                info['pieces_placed'] = pieces_placed
                info['steps'] = episode_steps

                # Apply simple reward shaping
                shaped_reward = reward_shaper.calculate_reward(
                    obs, action, raw_reward, done, info
                )

                # Store experience
                agent.remember(obs, action, shaped_reward, next_obs, done, info, raw_reward)

                # Learn every 4 steps
                if episode_steps % 4 == 0 and len(agent.memory) >= agent.min_buffer_size:
                    agent.learn()

                # Update metrics
                episode_reward += shaped_reward
                episode_steps += 1
                lines_this_step = info.get('lines_cleared', 0)
                episode_lines += lines_this_step

                # Move to next state
                obs = next_obs

            # Episode finished
            lines_cleared_total += episode_lines
            pieces_placed_total += pieces_placed

            # Update agent
            agent.end_episode(episode_reward, episode_steps, episode_lines)

            # Log episode
            logger.log_episode(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
                lines=episode_lines,
                epsilon=agent.epsilon,
                pieces_placed=pieces_placed
            )

            # Print progress
            if episode % args.log_freq == 0 or episode == args.episodes - 1:
                elapsed = time.time() - start_time
                avg_reward = np.mean(logger.rewards[-100:]) if logger.rewards else 0
                avg_lines = np.mean(logger.lines[-100:]) if logger.lines else 0
                avg_steps = np.mean(logger.steps[-100:]) if logger.steps else 0

                print(f"Ep {episode:5d} | "
                      f"R: {episode_reward:7.1f} (avg: {avg_reward:6.1f}) | "
                      f"Lines: {episode_lines:3d} (avg: {avg_lines:5.2f}) | "
                      f"Steps: {episode_steps:4d} (avg: {avg_steps:5.1f}) | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Mem: {len(agent.memory):6d} | "
                      f"Time: {elapsed:.1f}s")

            # Save checkpoint
            if episode % args.save_freq == 0 and episode > 0:
                agent.save_checkpoint(episode, MODEL_DIR)
                logger.save_logs()
                logger.plot_progress()

        # Training complete
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        total_time = time.time() - start_time
        print(f"Total episodes: {args.episodes}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Time per episode: {total_time/args.episodes:.2f}s")
        print(f"Total lines cleared: {lines_cleared_total}")
        print(f"Total pieces placed: {pieces_placed_total}")
        print(f"Lines per episode: {lines_cleared_total/args.episodes:.2f}")

        # Save final checkpoint
        agent.save_checkpoint(args.episodes, MODEL_DIR)
        logger.save_logs()
        logger.plot_progress()

        print(f"\n✓ Final checkpoint saved to {MODEL_DIR}")
        print(f"✓ Logs saved to {LOG_DIR}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        agent.save_checkpoint(episode if 'episode' in locals() else start_episode, MODEL_DIR)
        logger.save_logs()
        logger.plot_progress()
        print("✓ Progress saved")

    finally:
        env.close()


if __name__ == "__main__":
    main()
