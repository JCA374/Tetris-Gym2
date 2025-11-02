#!/usr/bin/env python3
"""
Improved Progressive Training Script for Tetris RL
Uses 5-stage curriculum with better reward shaping to fix center stacking problem

Run: python train_progressive_improved.py --episodes 10000 --force_fresh
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Import from existing codebase
from config import make_env, ENV_NAME, LR, GAMMA, BATCH_SIZE, MODEL_DIR, LOG_DIR
from src.agent import Agent
from src.utils import TrainingLogger, make_dir

# Import our improved reward shaper
from src.progressive_reward_improved import ImprovedProgressiveRewardShaper
from src.reward_shaping import get_column_heights, count_holes, calculate_bumpiness, extract_board_from_obs


def parse_args():
    """Parse training arguments"""
    parser = argparse.ArgumentParser(
        description='Train Tetris AI with Improved 5-Stage Curriculum')

    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train (default: 10000)')
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
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')

    return parser.parse_args()


def train(args):
    """Main training function with improved 5-stage curriculum"""
    start_time = time.time()
    print("="*80)
    print("🎓 TETRIS AI TRAINING - IMPROVED 5-STAGE CURRICULUM")
    print("="*80)
    print("✅ 5-Stage Curriculum Learning:")
    print("   Stage 1 (Foundation):        Episodes 0-500     - Basic survival")
    print("   Stage 2 (Clean Placement):   Episodes 500-1000  - Reduce holes")
    print("   Stage 3 (Spreading Found):   Episodes 1000-2000 - Learn spreading")
    print("   Stage 4 (Clean Spreading):   Episodes 2000-5000 - Clean + spread")
    print("   Stage 5 (Line Clearing):     Episodes 5000+     - Maximize lines")
    print("="*80)
    print()
    print("🔧 Key Improvements:")
    print("   ✅ Gentler hole penalty at start (avoid learned helplessness)")
    print("   ✅ Completable rows reward (8-9 filled, no holes)")
    print("   ✅ Clean rows reward (encourage tidy placement)")
    print("   ✅ Conditional survival bonus (only if holes < 30)")
    print("   ✅ Progressive hole penalty (0.3 → 2.0)")
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

    # Create improved progressive reward shaper
    reward_shaper = ImprovedProgressiveRewardShaper()
    print(f"\n✅ Improved progressive reward shaper initialized")

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
        reward_shaping="none",  # We handle shaping manually with improved curriculum
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
            reward_shaper.update_episode(start_episode)
            print(f"✅ Resumed from episode {start_episode}")
            print(f"   Current stage: {reward_shaper.get_current_stage()}")
        else:
            print("❌ No checkpoint found - starting fresh")
    else:
        print("🆕 Starting fresh training")

    # Setup experiment logging
    experiment_name = args.experiment_name or f"improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(LOG_DIR, experiment_name)

    # Training metrics
    lines_cleared_total = 0
    first_line_episode = None
    recent_rewards = []
    recent_lines = []
    recent_steps = []
    recent_holes = []
    recent_columns = []
    recent_completable_rows = []
    recent_clean_rows = []

    print(f"\n🚀 Starting improved progressive training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    print(f"Initial stage: {reward_shaper.get_current_stage()}")
    print("-" * 80)

    # Track previous stage for transition detection
    previous_stage = reward_shaper.get_current_stage()

    # MAIN TRAINING LOOP
    for episode in range(start_episode, args.episodes):
        # Update reward shaper with current episode (CRITICAL!)
        reward_shaper.update_episode(episode)
        current_stage = reward_shaper.get_current_stage()

        # Detect stage transitions
        if current_stage != previous_stage:
            print(f"\n{'='*80}")
            print(f"🎓 CURRICULUM ADVANCEMENT: {previous_stage} → {current_stage}")
            print(f"   Episode: {episode}")
            if recent_holes:
                print(f"   Recent avg holes: {np.mean(recent_holes):.1f}")
            if recent_columns:
                print(f"   Recent avg columns used: {np.mean(recent_columns):.1f}/10")
            if recent_completable_rows:
                print(f"   Recent avg completable rows: {np.mean(recent_completable_rows):.1f}")
            print(f"{'='*80}\n")
            previous_stage = current_stage

        obs, info = env.reset()
        episode_reward = 0
        original_reward = 0
        episode_steps = 0
        lines_this_episode = 0
        done = False
        pieces_placed = 0

        # Track metrics within episode
        final_board = None
        final_metrics = None

        while not done:
            # Select action
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store raw reward
            raw_reward = reward
            original_reward += raw_reward

            # Add pieces_placed to info for efficiency calculation
            if action == 5:  # Hard drop action
                pieces_placed += 1
            info['pieces_placed'] = pieces_placed
            info['steps'] = episode_steps

            # Apply IMPROVED PROGRESSIVE reward shaping
            shaped_reward = reward_shaper.calculate_reward(obs, action, raw_reward, done, info)

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

            # Store final board state for logging
            if done:
                final_board = extract_board_from_obs(next_obs)
                final_metrics = reward_shaper.calculate_metrics(final_board, info)

            obs = next_obs

        # End of episode - log stats
        agent.end_episode(episode_reward, episode_steps, lines_this_episode, original_reward)

        # Extract final board stats
        if final_board is None:
            final_board = extract_board_from_obs(obs)
            final_metrics = reward_shaper.calculate_metrics(final_board, info)

        heights = final_metrics['column_heights']
        holes = final_metrics['holes']
        bumpiness = final_metrics['bumpiness']
        columns_used = final_metrics['columns_used']
        completable_rows = final_metrics['completable_rows']
        clean_rows = final_metrics['clean_rows']
        max_height = max(heights) if heights else 0
        outer_unused = final_metrics['outer_unused']

        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        recent_holes.append(holes)
        recent_columns.append(columns_used)
        recent_completable_rows.append(completable_rows)
        recent_clean_rows.append(clean_rows)

        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_lines.pop(0)
            recent_steps.pop(0)
            recent_holes.pop(0)
            recent_columns.pop(0)
            recent_completable_rows.pop(0)
            recent_clean_rows.pop(0)

        # Log episode data
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=agent.epsilon,
            lines_cleared=lines_this_episode,
            original_reward=original_reward,
            total_lines=lines_cleared_total,
            shaped_reward_used=True,
            stage=current_stage,
            holes=holes,
            columns_used=columns_used,
            completable_rows=completable_rows,
            clean_rows=clean_rows
        )

        # Log board state periodically (SAME AS ORIGINAL)
        if (episode + 1) % args.log_freq == 0:
            logger.log_board_state(
                episode=episode + 1,
                board=final_board,
                reward=episode_reward,
                steps=episode_steps,
                lines_cleared=lines_this_episode,
                heights=heights,
                holes=holes,
                bumpiness=bumpiness,
                max_height=max_height
            )

        # Print progress
        if (episode + 1) % args.log_freq == 0 or lines_this_episode > 0:
            avg_reward = np.mean(recent_rewards)
            avg_lines = np.mean(recent_lines)
            avg_steps = np.mean(recent_steps)
            avg_holes = np.mean(recent_holes) if recent_holes else 0
            avg_cols = np.mean(recent_columns) if recent_columns else 0
            avg_completable = np.mean(recent_completable_rows) if recent_completable_rows else 0
            avg_clean = np.mean(recent_clean_rows) if recent_clean_rows else 0

            print(f"Ep {episode+1:4d} | "
                  f"Stage: {current_stage:20s} | "
                  f"Steps: {episode_steps:3d} (avg {avg_steps:.1f}) | "
                  f"Holes: {holes:3d} (avg {avg_holes:.1f}) | "
                  f"Cols: {columns_used:2d}/10 (avg {avg_cols:.1f}) | "
                  f"Compl: {completable_rows:2d} (avg {avg_completable:.1f}) | "
                  f"Lines: {lines_this_episode} | "
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
    print(f"TRAINING COMPLETE - IMPROVED 5-STAGE CURRICULUM")
    print("="*80)
    episodes_trained = args.episodes - start_episode
    print(f"Total episodes: {episodes_trained}")
    print(f"Total lines cleared: {lines_cleared_total}")

    if episodes_trained > 0:
        avg_lines_all = lines_cleared_total / episodes_trained
        print(f"Average lines per episode: {avg_lines_all:.3f}")

    print(f"First line at episode: {first_line_episode or 'Never'}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Final stage: {reward_shaper.get_current_stage()}")

    if len(recent_rewards) > 0:
        print(f"\nRecent performance (last {len(recent_rewards)} episodes):")
        print(f"  Average reward: {np.mean(recent_rewards):.1f}")
        print(f"  Average steps: {np.mean(recent_steps):.1f}")
        print(f"  Average lines/episode: {np.mean(recent_lines):.3f}")
        print(f"  Average holes: {np.mean(recent_holes):.1f}")
        print(f"  Average columns used: {np.mean(recent_columns):.1f}/10")
        print(f"  Average completable rows: {np.mean(recent_completable_rows):.1f}")
        print(f"  Average clean rows: {np.mean(recent_clean_rows):.1f}")

    # Stage progression summary
    if reward_shaper.stage_transitions:
        print(f"\n📈 Curriculum Progression:")
        for transition in reward_shaper.stage_transitions:
            print(f"   Episode {transition['episode']:4d}: "
                  f"{transition['from_stage']:20s} → {transition['to_stage']:20s}")

    # Success criteria check
    avg_holes_final = np.mean(recent_holes) if recent_holes else 999
    avg_cols_final = np.mean(recent_columns) if recent_columns else 0
    avg_lines_final = np.mean(recent_lines) if recent_lines else 0
    avg_steps_final = np.mean(recent_steps) if recent_steps else 0

    print(f"\n{'='*80}")
    print("IMPROVED CURRICULUM SUCCESS METRICS:")
    print("="*80)

    # Check Stage 2 success (clean placement)
    if avg_holes_final < 15:
        print("✅ Stage 2 (Clean Placement): SUCCESS")
        print(f"   Holes: {avg_holes_final:.1f} (target: <15)")
    else:
        print("⚠️  Stage 2 (Clean Placement): NEEDS MORE TRAINING")
        print(f"   Holes: {avg_holes_final:.1f} (target: <15)")

    # Check Stage 4 success (spreading)
    if avg_cols_final >= 8:
        print("✅ Stage 4 (Clean Spreading): SUCCESS")
        print(f"   Columns used: {avg_cols_final:.1f}/10 (target: ≥8)")
    else:
        print("⚠️  Stage 4 (Clean Spreading): NEEDS MORE TRAINING")
        print(f"   Columns used: {avg_cols_final:.1f}/10 (target: ≥8)")

    # Check Stage 5 success (line clears)
    if avg_lines_final >= 2.0:
        print("✅ Stage 5 (Line Clearing): SUCCESS")
        print(f"   Lines/episode: {avg_lines_final:.2f} (target: ≥2.0)")
    elif lines_cleared_total > 0:
        print("⚠️  Stage 5 (Line Clearing): IN PROGRESS")
        print(f"   Lines/episode: {avg_lines_final:.2f} (target: ≥2.0)")
    else:
        print("⚠️  Stage 5 (Line Clearing): NOT STARTED")
        print("   No lines cleared yet - agent needs more training")

    # Overall performance
    if avg_steps_final >= 100 and avg_holes_final < 10 and avg_lines_final >= 5:
        print("\n🎉🎉🎉 EXCELLENT PERFORMANCE! 🎉🎉🎉")
    elif avg_steps_final >= 80 and avg_cols_final >= 8:
        print("\n👍 GOOD PROGRESS - Continue training for better line clearing")
    else:
        print("\n📚 LEARNING IN PROGRESS - Give the curriculum more time")

    print("="*80)


def main():
    """Main entry point"""
    args = parse_args()

    print("🎓 Tetris AI Training - Improved 5-Stage Curriculum")
    print("="*80)
    print("Key improvements over 4-stage curriculum:")
    print("  ✅ Gentler start to avoid learned helplessness")
    print("  ✅ Completable rows metric (8-9 filled, no holes)")
    print("  ✅ Clean rows metric (encourage tidy placement)")
    print("  ✅ Conditional survival bonus (penalize messy play)")
    print("  ✅ Progressive hole penalty (0.3 → 2.0)")
    print("  ✅ Longer stages for better learning")
    print()

    train(args)


if __name__ == "__main__":
    main()
