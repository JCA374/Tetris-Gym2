#!/usr/bin/env python3
"""
Progressive Curriculum Training for Tetris AI

Uses 4-stage curriculum to teach:
1. Clean piece placement (avoid holes)
2. Height management
3. Spreading across columns
4. Balanced optimal play

Run: python train_progressive.py --episodes 1000
"""

from config import make_env, ENV_NAME, LR, GAMMA, BATCH_SIZE, MAX_EPISODES, MODEL_DIR, LOG_DIR
from src.agent import Agent
from src.utils import TrainingLogger, print_system_info, make_dir
from src.progressive_reward import ProgressiveRewardShaper
from src.reward_shaping import get_column_heights, count_holes, calculate_bumpiness

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser(
        description='Train Tetris AI with Progressive Curriculum')

    parser.add_argument('--episodes', type=int, default=20000,
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help=f'Discount factor (default: {GAMMA})')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'dueling_dqn'],
                        help='Model architecture type')

    # Complete vision options
    parser.add_argument('--use_complete_vision', action='store_true', default=True,
                        help='Use complete 4-channel vision (REQUIRED)')
    parser.add_argument('--use_cnn', action='store_true', default=True,
                        help='Use CNN processing')

    # Epsilon settings for curriculum learning
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Starting epsilon for exploration (default: 1.0)')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995,
                        help='Epsilon decay rate (default: 0.9995)')

    # Training control
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--force_fresh', action='store_true',
                        help='Force fresh start even if checkpoint exists')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    # Curriculum settings
    parser.add_argument('--stage_basic', type=int, default=200,
                        help='Episodes for basic stage (default: 200)')
    parser.add_argument('--stage_height', type=int, default=400,
                        help='Episodes for height stage (default: 400)')
    parser.add_argument('--stage_spreading', type=int, default=600,
                        help='Episodes for spreading stage (default: 600)')

    return parser.parse_args()


def train(args):
    """Main training function with progressive curriculum"""
    start_time = time.time()
    print("="*80)
    print("🎓 TETRIS AI TRAINING - PROGRESSIVE CURRICULUM")
    print("="*80)
    print("✅ 4-Stage Curriculum Learning:")
    print(f"   Stage 1 (Basic):     Episodes 0-{args.stage_basic} - Learn clean placement")
    print(f"   Stage 2 (Height):    Episodes {args.stage_basic}-{args.stage_height} - Height management")
    print(f"   Stage 3 (Spreading): Episodes {args.stage_height}-{args.stage_spreading} - Spread pieces")
    print(f"   Stage 4 (Balanced):  Episodes {args.stage_spreading}+ - Optimal play")
    print("="*80)

    if not args.use_complete_vision:
        print("⚠️  WARNING: Complete vision disabled! This will likely fail!")
        print("   Add --use_complete_vision flag")

    # Create environment
    env = make_env(
        use_complete_vision=args.use_complete_vision,
        use_cnn=args.use_cnn
    )
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")

    if len(env.observation_space.shape) == 3:
        channels = env.observation_space.shape[-1]
        if channels == 4:
            print(f"   ✅ 4-channel complete vision confirmed!")
        else:
            print(f"   ⚠️  Only {channels} channels - may be missing information!")

    # Create progressive reward shaper
    stage_thresholds = {
        "basic": args.stage_basic,
        "height": args.stage_height,
        "spreading": args.stage_spreading,
        "balanced": float('inf')
    }
    reward_shaper = ProgressiveRewardShaper(stage_thresholds)
    print(f"\n✅ Progressive reward shaper initialized")

    # Create agent
    agent = Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        model_type=args.model_type,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        reward_shaping="none",  # We handle shaping manually with curriculum
        max_episodes=args.episodes
    )

    # Handle resume/fresh start
    start_episode = 0
    if args.force_fresh:
        print("🆕 Forcing fresh start (ignoring any checkpoints)")
    elif args.resume:
        print(f"\n🔄 Attempting to load checkpoint...")
        if agent.load_checkpoint(latest=True, model_dir=MODEL_DIR):
            start_episode = agent.episodes_done
            reward_shaper.episode_count = start_episode
            reward_shaper.update_stage(start_episode)
            print(f"✅ Resumed from episode {start_episode}")
            print(f"   Current stage: {reward_shaper.current_stage}")

            # Force higher epsilon for exploration with new shaping
            if agent.epsilon < 0.3:
                print(f"⚠️  Epsilon too low ({agent.epsilon:.3f}), boosting to 0.5")
                agent.epsilon = 0.5
        else:
            print("❌ No checkpoint found - starting fresh")
    else:
        print("🆕 Starting fresh training")

    # Setup experiment logging
    experiment_name = args.experiment_name or f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = []
    recent_lines = []
    recent_steps = []
    recent_holes = []
    recent_columns = []

    print(f"\n🚀 Starting progressive training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    print(f"Initial stage: {reward_shaper.current_stage}")
    print("-" * 80)

    # MAIN TRAINING LOOP
    for episode in range(start_episode, args.episodes):
        # Update reward shaper with current episode (critical for stage advancement!)
        reward_shaper.episode_count = episode

        obs, info = env.reset()
        episode_reward = 0
        original_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store raw reward
            raw_reward = reward
            original_reward += raw_reward

            # Apply PROGRESSIVE reward shaping
            shaped_reward = reward_shaper.shape_reward(obs, action, raw_reward, done, info)

            # Store experience with shaped reward
            agent.remember(obs, action, shaped_reward, next_obs, done, info, raw_reward)

            # Learn every 4 steps
            if episode_steps % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.learn()

            # Update metrics
            episode_reward += shaped_reward
            episode_steps += 1

            # Track line clears
            lines = info.get('lines_cleared', 0)
            if lines > 0:
                lines_this_episode += lines
                lines_cleared_total += lines
                if first_line_episode is None:
                    first_line_episode = episode + 1
                    print(f"\n🎉🎉🎉 FIRST LINE CLEARED! Episode {first_line_episode} 🎉🎉🎉\n")

            obs = next_obs

        # End of episode - log stats
        agent.end_episode(episode_reward, episode_steps, lines_this_episode, original_reward)

        # Extract final board stats
        from src.reward_shaping import extract_board_from_obs
        board = extract_board_from_obs(next_obs)
        heights = get_column_heights(board)
        holes = count_holes(board)
        bumpiness = calculate_bumpiness(board)
        columns_used = sum(1 for h in heights if h > 0)
        max_height = max(heights) if heights else 0
        outer_unused = sum(1 for c in [0,1,2,7,8,9] if heights[c] == 0)

        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        recent_holes.append(holes)
        recent_columns.append(columns_used)

        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lines.pop(0)
            recent_steps.pop(0)
            recent_holes.pop(0)
            recent_columns.pop(0)

        # Log episode data
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total,
            shaped_reward_used=True
        )

        # Log board state periodically
        if (episode + 1) % args.log_freq == 0:
            max_row_fullness = 0
            for r in range(board.shape[0]):
                filled = int((board[r, :] > 0).sum())
                if filled > max_row_fullness:
                    max_row_fullness = filled

            logger.log_board_state(
                episode=episode + 1,
                board=board,
                reward=episode_reward,
                steps=episode_steps,
                lines_cleared=lines_this_episode,
                heights=heights,
                holes=holes,
                bumpiness=bumpiness,
                max_height=max_height,
                max_row_fullness=max_row_fullness
            )

        # Print progress
        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            avg_reward = np.mean(recent_rewards)
            avg_lines = np.mean(recent_lines)
            avg_steps = np.mean(recent_steps)
            avg_holes = np.mean(recent_holes) if recent_holes else 0
            avg_cols = np.mean(recent_columns) if recent_columns else 0

            stage_info = reward_shaper.get_stage_info()

            print(f"Ep {episode+1:4d} | "
                  f"Stage: {stage_info['stage']:10s} | "
                  f"Cols: {columns_used:2d}/10 (avg {avg_cols:.1f}) | "
                  f"Outer: {outer_unused}/6 | "
                  f"Holes: {holes:3d} (avg {avg_holes:.1f}) | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"ε: {agent.epsilon:.3f}")

        # Save checkpoint periodically
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()
            print(f"💾 Checkpoint saved at episode {episode + 1}")

    # Training complete
    training_time = time.time() - start_time
    env.close()

    # Save final checkpoint
    agent.save_checkpoint(args.episodes, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    # Print final summary
    print(f"\n" + "="*80)
    print(f"TRAINING COMPLETE - PROGRESSIVE CURRICULUM")
    print("="*80)
    episodes_trained = args.episodes - start_episode
    print(f"Total episodes: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")

    if episodes_trained > 0:
        avg_lines_all = lines_cleared_total / episodes_trained
        print(f"Average lines per episode: {avg_lines_all:.3f}")

    print(f"First line at episode: {first_line_episode or 'Never'}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Final stage: {reward_shaper.current_stage}")

    if len(recent_rewards) > 0:
        print(f"\nRecent performance (last {len(recent_rewards)} episodes):")
        print(f"  Average reward: {np.mean(recent_rewards):.1f}")
        print(f"  Average steps: {np.mean(recent_steps):.1f}")
        print(f"  Average lines/episode: {np.mean(recent_lines):.3f}")
        print(f"  Average holes: {np.mean(recent_holes):.1f}")
        print(f"  Average columns used: {np.mean(recent_columns):.1f}/10")

    # Stage progression summary
    if reward_shaper.stage_history:
        print(f"\n📈 Curriculum Progression:")
        for stage_change in reward_shaper.stage_history:
            print(f"   Episode {stage_change['episode']:4d}: "
                  f"{stage_change['old_stage']:10s} → {stage_change['new_stage']:10s}")

    # Provide feedback
    avg_holes_final = np.mean(recent_holes) if recent_holes else 999
    avg_cols_final = np.mean(recent_columns) if recent_columns else 0

    print(f"\n{'='*80}")
    print("CURRICULUM SUCCESS METRICS:")
    print("="*80)

    # Check Stage 1 success (hole reduction)
    if avg_holes_final < 20:
        print("✅ Stage 1 (Clean Placement): SUCCESS")
        print(f"   Holes reduced to {avg_holes_final:.1f} (target: <20)")
    else:
        print("⚠️  Stage 1 (Clean Placement): NEEDS MORE TRAINING")
        print(f"   Holes: {avg_holes_final:.1f} (target: <20)")

    # Check Stage 3 success (spreading)
    if avg_cols_final >= 7:
        print("✅ Stage 3 (Spreading): SUCCESS")
        print(f"   Columns used: {avg_cols_final:.1f}/10 (target: ≥7)")
    else:
        print("⚠️  Stage 3 (Spreading): NEEDS MORE TRAINING")
        print(f"   Columns used: {avg_cols_final:.1f}/10 (target: ≥7)")

    # Check final success (line clears)
    if lines_cleared_total > episodes_trained * 0.1:
        print("✅ Final Performance: SUCCESS")
        print(f"   Line clearing rate: {lines_cleared_total/episodes_trained:.2f} lines/episode")
    else:
        print("⚠️  Final Performance: NEEDS MORE TRAINING")
        print(f"   Line clearing rate: {lines_cleared_total/episodes_trained:.2f} lines/episode")
        print("   Recommendation: Continue training for more episodes")

    print("="*80)


def main():
    """Main entry point"""
    args = parse_args()

    print("🎓 Tetris AI Training - Progressive Curriculum")
    print("="*80)
    print("Key features:")
    print("  ✅ 4-stage curriculum learning")
    print("  ✅ Skill-based progression (motor control → strategy)")
    print("  ✅ Adaptive reward shaping per stage")
    print("  ✅ Automatic stage advancement")
    print()

    train(args)


if __name__ == "__main__":
    main()
