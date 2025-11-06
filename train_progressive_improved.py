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
import signal
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

# Global variable for graceful shutdown
_training_interrupted = False


def generate_debug_summary(args, start_episode, total_episodes, training_time,
                          lines_cleared_total, first_line_episode,
                          recent_rewards, recent_steps, recent_lines,
                          recent_holes, recent_columns, recent_completable_rows,
                          recent_clean_rows, reward_shaper, agent, logger):
    """Generate comprehensive debug summary for post-training analysis"""

    episodes_trained = total_episodes - start_episode

    # Calculate statistics
    recent_tail = lambda data: data[-100:] if len(data) >= 100 else data

    tail_rewards = recent_tail(recent_rewards)
    tail_steps = recent_tail(recent_steps)
    tail_lines = recent_tail(recent_lines)
    tail_holes = recent_tail(recent_holes)
    tail_cols = recent_tail(recent_columns)
    tail_completable = recent_tail(recent_completable_rows)
    tail_clean = recent_tail(recent_clean_rows)

    avg_reward = np.mean(tail_rewards) if tail_rewards else 0
    avg_steps = np.mean(tail_steps) if tail_steps else 0
    avg_lines = np.mean(tail_lines) if tail_lines else 0
    avg_holes = np.mean(tail_holes) if tail_holes else 0
    avg_cols = np.mean(tail_cols) if tail_cols else 0
    avg_completable = np.mean(tail_completable) if tail_completable else 0
    avg_clean = np.mean(tail_clean) if tail_clean else 0

    lines_per_episode = lines_cleared_total / episodes_trained if episodes_trained > 0 else 0

    summary = []
    summary.append("=" * 80)
    summary.append("TETRIS AI TRAINING - DEBUG SUMMARY")
    summary.append("=" * 80)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Experiment: {logger.experiment_name}")
    summary.append("")

    # Training Configuration
    summary.append("=" * 80)
    summary.append("TRAINING CONFIGURATION")
    summary.append("=" * 80)
    summary.append(f"Total episodes:        {total_episodes}")
    summary.append(f"Episodes trained:      {episodes_trained} (from {start_episode} to {total_episodes})")
    summary.append(f"Training time:         {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)")
    summary.append(f"Time per episode:      {training_time/episodes_trained:.2f} seconds" if episodes_trained > 0 else "N/A")
    summary.append(f"Learning rate:         {args.lr}")
    summary.append(f"Batch size:            {args.batch_size}")
    summary.append(f"Gamma (discount):      {args.gamma}")
    summary.append(f"Epsilon start:         {args.epsilon_start}")
    summary.append(f"Epsilon end:           {args.epsilon_end}")
    summary.append(f"Epsilon decay:         {args.epsilon_decay}")
    summary.append(f"Final epsilon:         {agent.epsilon:.4f}")
    summary.append(f"Model type:            {args.model_type}")
    summary.append(f"Complete vision:       {args.use_complete_vision}")
    summary.append(f"CNN enabled:           {args.use_cnn}")
    summary.append("")

    # Curriculum Stage Information
    summary.append("=" * 80)
    summary.append("CURRICULUM PROGRESSION")
    summary.append("=" * 80)
    summary.append(f"Final stage:           {reward_shaper.get_current_stage()}")
    summary.append("")
    summary.append("Stage Thresholds:")
    summary.append("  Stage 1 (Foundation):        Episodes 0-500")
    summary.append("  Stage 2 (Clean Placement):   Episodes 500-1000")
    summary.append("  Stage 3 (Spreading Found):   Episodes 1000-2000")
    summary.append("  Stage 4 (Clean Spreading):   Episodes 2000-5000")
    summary.append("  Stage 5 (Line Clearing):     Episodes 5000+")
    summary.append("")

    if reward_shaper.stage_transitions:
        summary.append("Stage Transitions:")
        for transition in reward_shaper.stage_transitions:
            summary.append(f"  Episode {transition['episode']:5d}: {transition['from_stage']:20s} → {transition['to_stage']}")
    else:
        summary.append("Stage Transitions: None (training too short or didn't advance)")
    summary.append("")

    # Performance Metrics
    summary.append("=" * 80)
    summary.append("PERFORMANCE METRICS (Last 100 Episodes)")
    summary.append("=" * 80)
    summary.append(f"Average reward:        {avg_reward:8.1f}")
    summary.append(f"Average steps:         {avg_steps:8.1f}")
    summary.append(f"Average lines/ep:      {avg_lines:8.2f}")
    summary.append(f"Total lines cleared:   {lines_cleared_total:8d}")
    summary.append(f"Overall lines/ep:      {lines_per_episode:8.3f}")
    summary.append(f"First line at:         Episode {first_line_episode if first_line_episode else 'Never'}")
    summary.append("")
    summary.append("Board Quality Metrics:")
    summary.append(f"  Holes:               {avg_holes:8.1f}  (target: <15)")
    summary.append(f"  Columns used:        {avg_cols:8.1f}/10  (target: ≥8)")
    summary.append(f"  Completable rows:    {avg_completable:8.1f}  (target: 3-5)")
    summary.append(f"  Clean rows:          {avg_clean:8.1f}  (target: 10-15)")
    summary.append("")

    # Success Criteria Evaluation
    summary.append("=" * 80)
    summary.append("SUCCESS CRITERIA EVALUATION")
    summary.append("=" * 80)

    stage2_success = avg_holes < 15
    stage4_success = avg_cols >= 8
    stage5_success = avg_lines >= 2.0
    overall_success = avg_steps >= 100 and avg_holes < 10 and avg_lines >= 5

    summary.append(f"Stage 2 (Clean Placement):  {'✅ SUCCESS' if stage2_success else '❌ FAILED'}")
    summary.append(f"  Holes: {avg_holes:.1f} (target: <15)")
    summary.append("")
    summary.append(f"Stage 4 (Clean Spreading):  {'✅ SUCCESS' if stage4_success else '❌ FAILED'}")
    summary.append(f"  Columns used: {avg_cols:.1f}/10 (target: ≥8)")
    summary.append("")
    summary.append(f"Stage 5 (Line Clearing):    {'✅ SUCCESS' if stage5_success else '❌ FAILED'}")
    summary.append(f"  Lines/episode: {avg_lines:.2f} (target: ≥2.0)")
    summary.append("")
    summary.append(f"Overall Performance:        {'🎉 EXCELLENT!' if overall_success else '👍 GOOD PROGRESS' if (stage4_success and avg_steps >= 80) else '📚 NEEDS MORE TRAINING'}")
    summary.append("")

    # Problem Analysis
    summary.append("=" * 80)
    summary.append("PROBLEM ANALYSIS & RECOMMENDATIONS")
    summary.append("=" * 80)

    problems = []
    recommendations = []

    # Analyze holes
    if avg_holes > 50:
        problems.append("❌ CRITICAL: Holes too high (>50) - Board is swiss cheese")
        recommendations.append("→ Agent not learning clean placement")
        recommendations.append("→ Increase hole penalty in current stage by 50%")
        recommendations.append("→ Reduce survival bonus if holes > 30")
    elif avg_holes > 30:
        problems.append("⚠️  WARNING: Holes high (30-50) - Board quality poor")
        recommendations.append("→ Agent needs more time in earlier stages")
        recommendations.append("→ Consider increasing hole penalty by 20-30%")
    elif avg_holes > 15:
        problems.append("⚠️  Holes moderate (15-30) - Room for improvement")
        recommendations.append("→ Agent progressing but not mastered clean play")
        recommendations.append("→ Continue training in current stage")
    else:
        problems.append("✅ Holes low (<15) - Clean placement achieved!")

    # Analyze spreading
    if avg_cols < 6:
        problems.append("❌ CRITICAL: Not spreading (<6 columns) - Center stacking")
        recommendations.append("→ Increase spread bonus and columns_used reward")
        recommendations.append("→ Increase outer_unused penalty")
        recommendations.append("→ Check if Stage 3+ rewards are active")
    elif avg_cols < 8:
        problems.append("⚠️  WARNING: Limited spreading (6-8 columns)")
        recommendations.append("→ Agent learning to spread but not fully")
        recommendations.append("→ Continue training in spreading stages")
    else:
        problems.append("✅ Spreading achieved (≥8 columns)!")

    # Analyze line clearing
    if avg_lines < 0.1 and total_episodes > 5000:
        problems.append("❌ CRITICAL: No line clears despite 5000+ episodes")
        recommendations.append("→ Agent likely has too many holes to clear lines")
        recommendations.append("→ Check completable_rows metric (should be >0)")
        recommendations.append("→ May need to restart with stronger hole penalties")
    elif avg_lines < 1.0 and total_episodes > 5000:
        problems.append("⚠️  WARNING: Low line clears (<1/episode) in Stage 5")
        recommendations.append("→ Agent needs to reduce holes first")
        recommendations.append("→ Increase completable_rows bonus")
        recommendations.append("→ Continue training for 2000-3000 more episodes")
    elif avg_lines < 2.0:
        problems.append("⚠️  Line clears happening but infrequent")
        recommendations.append("→ Good progress, continue training")
    else:
        problems.append("✅ Consistent line clearing (≥2/episode)!")

    # Analyze completable rows
    if avg_completable < 0.5 and total_episodes > 3000:
        problems.append("⚠️  WARNING: No completable rows (rows with 8-9 filled, no holes)")
        recommendations.append("→ Agent not learning to set up line clears")
        recommendations.append("→ Increase completable_rows bonus significantly")
        recommendations.append("→ This is the key metric bridging placement → line clears")

    # Check for tall towers
    if avg_holes > 30 and avg_steps > 100:
        problems.append("⚠️  WARNING: Building tall towers with many holes")
        recommendations.append("→ Agent getting survival bonus despite bad board")
        recommendations.append("→ Make survival bonus more conditional (only if holes <20)")
        recommendations.append("→ Add explicit height penalty for towers >15")

    if problems:
        summary.append("Problems Detected:")
        for problem in problems:
            summary.append(f"  {problem}")
        summary.append("")

    if recommendations:
        summary.append("Recommendations:")
        for rec in recommendations:
            summary.append(f"  {rec}")
        summary.append("")

    if not recommendations:
        summary.append("✅ No critical issues detected!")
        summary.append("→ Continue training to further improve performance")
        summary.append("")

    # Next Steps
    summary.append("=" * 80)
    summary.append("NEXT STEPS")
    summary.append("=" * 80)

    if overall_success:
        summary.append("🎉 Training successful! Agent has mastered Tetris basics.")
        summary.append("")
        summary.append("Suggested next steps:")
        summary.append("  1. Continue training for 5000-10000 more episodes to optimize")
        summary.append("  2. Experiment with different epsilon decay rates")
        summary.append("  3. Try different model architectures (dueling DQN)")
        summary.append("  4. Test the agent in different game modes")
    elif stage4_success and avg_steps >= 80:
        summary.append("👍 Good progress! Agent has learned spreading.")
        summary.append("")
        summary.append("Suggested next steps:")
        summary.append(f"  1. Continue training for {max(10000 - total_episodes, 3000)} more episodes")
        summary.append("  2. Focus on reducing holes to enable line clears")
        summary.append("  3. Monitor completable_rows metric - should increase")
        summary.append("  4. Expect first consistent line clears around episode 6000-8000")
    else:
        summary.append("📚 Training in progress. Agent still learning fundamentals.")
        summary.append("")
        summary.append("Suggested next steps:")
        if total_episodes < 5000:
            summary.append(f"  1. Continue training to at least episode 5000 ({5000 - total_episodes} more episodes)")
            summary.append("  2. Let curriculum fully progress through all 5 stages")
            summary.append("  3. Monitor stage transitions and metrics")
        else:
            summary.append("  1. Review problems and recommendations above")
            summary.append("  2. Consider adjusting reward function based on analysis")
            summary.append("  3. If holes still >50, consider restarting with stronger penalties")
            summary.append("  4. If spreading not happening, increase spread bonuses")

    summary.append("")

    # Reward Function Analysis
    summary.append("=" * 80)
    summary.append("REWARD FUNCTION EFFECTIVENESS")
    summary.append("=" * 80)
    summary.append(f"Current stage: {reward_shaper.get_current_stage()}")
    summary.append("")

    # Example reward calculation for current performance
    current_stage = reward_shaper.get_current_stage()
    summary.append(f"Typical reward for current performance (stage: {current_stage}):")
    summary.append(f"  Assumptions: {int(avg_holes)} holes, {int(avg_cols)} columns, {int(avg_steps)} steps")
    summary.append("")

    if current_stage == "line_clearing_focus":
        summary.append("  Hole penalty:        -3.5 × {:.0f} = {:.1f}".format(avg_holes, -3.5 * avg_holes))
        summary.append("  Completable rows:    +15.0 × {:.1f} = {:.1f}".format(avg_completable, 15.0 * avg_completable))
        summary.append("  Clean rows:          +12.0 × {:.1f} = {:.1f}".format(avg_clean, 12.0 * avg_clean))
        summary.append("  Spread bonus:        ~+20.0")
        summary.append("  Columns bonus:       +4.0 × {:.0f} = {:.1f}".format(avg_cols, 4.0 * avg_cols))
        if avg_holes < 10:
            survival = min(avg_steps * 0.5, 40.0)
            summary.append(f"  Survival bonus:      +{survival:.1f} (full bonus, holes <10)")
        elif avg_holes < 20:
            survival = min(avg_steps * 0.3, 25.0)
            summary.append(f"  Survival bonus:      +{survival:.1f} (reduced, holes 10-20)")
        elif avg_holes < 30:
            survival = min(avg_steps * 0.1, 10.0)
            summary.append(f"  Survival bonus:      +{survival:.1f} (minimal, holes 20-30)")
        else:
            summary.append("  Survival bonus:      +0.0 (NO bonus, holes ≥30)")
    elif current_stage == "clean_spreading":
        summary.append("  Hole penalty:        -2.5 × {:.0f} = {:.1f}".format(avg_holes, -2.5 * avg_holes))
        summary.append("  Completable rows:    +10.0 × {:.1f} = {:.1f}".format(avg_completable, 10.0 * avg_completable))
        summary.append("  Clean rows:          +7.0 × {:.1f} = {:.1f}".format(avg_clean, 7.0 * avg_clean))
        summary.append("  Spread bonus:        ~+25.0")
        summary.append("  Columns bonus:       +5.0 × {:.0f} = {:.1f}".format(avg_cols, 5.0 * avg_cols))

    summary.append("")
    summary.append("If rewards are consistently negative:")
    summary.append("  → Agent is being penalized more than rewarded")
    summary.append("  → This is OK in early stages (learning from mistakes)")
    summary.append("  → Should become positive by episode 3000-5000")
    summary.append("")
    summary.append("If rewards are too high (>10000) without line clears:")
    summary.append("  → Agent may be exploiting survival bonus")
    summary.append("  → Check if holes are high - survival should be conditional")
    summary.append("  → Hole penalty may need to be stronger")
    summary.append("")

    # File References
    summary.append("=" * 80)
    summary.append("OUTPUT FILES")
    summary.append("=" * 80)
    summary.append(f"Log directory:         {logger.experiment_dir}")
    summary.append(f"Board states:          board_states.txt")
    summary.append(f"Episode log (CSV):     episode_log.csv")
    summary.append(f"Reward plot:           reward_progress.png")
    summary.append(f"Metrics plot:          training_metrics.png")
    summary.append(f"Debug summary:         DEBUG_SUMMARY.txt (this file)")
    summary.append("")

    # Useful Commands
    summary.append("=" * 80)
    summary.append("USEFUL DEBUG COMMANDS")
    summary.append("=" * 80)
    summary.append("View recent board states:")
    summary.append(f"  tail -100 {logger.experiment_dir}/board_states.txt")
    summary.append("")
    summary.append("Analyze hole progression:")
    summary.append(f"  awk -F',' 'NR>1 {{print $4,$6}}' {logger.experiment_dir}/episode_log.csv | tail -100")
    summary.append("")
    summary.append("Check line clearing progress:")
    summary.append(f"  awk -F',' 'NR>1 {{sum+=$7}} NR%1000==0 {{print NR,sum}}' {logger.experiment_dir}/episode_log.csv")
    summary.append("")
    summary.append("Resume training:")
    summary.append(f"  python train_progressive_improved.py --episodes {total_episodes + 5000} --resume")
    summary.append("")

    summary.append("=" * 80)
    summary.append("END OF DEBUG SUMMARY")
    summary.append("=" * 80)

    return "\n".join(summary)


def save_debug_summary(logger, args, start_episode, current_episode, training_time,
                       lines_cleared_total, first_line_episode,
                       recent_rewards, recent_steps, recent_lines,
                       recent_holes, recent_columns, recent_completable_rows,
                       recent_clean_rows, reward_shaper, agent, is_interrupted=False):
    """Save debug summary to file with current training state"""
    try:
        debug_summary = generate_debug_summary(
            args=args,
            start_episode=start_episode,
            total_episodes=current_episode,
            training_time=training_time,
            lines_cleared_total=lines_cleared_total,
            first_line_episode=first_line_episode,
            recent_rewards=recent_rewards,
            recent_steps=recent_steps,
            recent_lines=recent_lines,
            recent_holes=recent_holes,
            recent_columns=recent_columns,
            recent_completable_rows=recent_completable_rows,
            recent_clean_rows=recent_clean_rows,
            reward_shaper=reward_shaper,
            agent=agent,
            logger=logger
        )

        # Add interruption note if applicable
        if is_interrupted:
            debug_summary = f"⚠️  TRAINING INTERRUPTED AT EPISODE {current_episode}\n\n" + debug_summary

        # Save to file
        debug_path = logger.experiment_dir / "DEBUG_SUMMARY.txt"
        with open(debug_path, 'w') as f:
            f.write(debug_summary)

        print(f"\n📊 Debug summary saved: {debug_path}")
        return debug_path
    except Exception as e:
        print(f"\n⚠️  Error saving debug summary: {e}")
        return None


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
    parser.add_argument('--model_type', type=str, default='dqn',
                        choices=['dqn', 'dueling_dqn', 'hybrid_dqn', 'hybrid_dueling_dqn'],
                        help='Model architecture type (hybrid models require 8-channel mode)')

    # Complete vision options
    parser.add_argument('--use_complete_vision', action='store_true', default=True,
                        help='Use complete 4-channel vision (REQUIRED)')
    parser.add_argument('--use_cnn', action='store_true', default=True,
                        help='Use CNN processing')
    parser.add_argument('--use_feature_channels', action='store_true', default=True,
                        help='Use 8-channel hybrid mode (visual + features) vs 4-channel (visual only)')
    parser.add_argument('--no_feature_channels', dest='use_feature_channels', action='store_false',
                        help='Disable feature channels (use 4-channel visual-only mode)')

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
        use_cnn=args.use_cnn,
        use_feature_channels=args.use_feature_channels
    )
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")

    if len(env.observation_space.shape) == 3:
        channels = env.observation_space.shape[-1]
        if channels == 8 and args.use_feature_channels:
            print(f"   ✅ 8-channel HYBRID mode confirmed (visual + features)!")
            if args.model_type in ['hybrid_dqn', 'hybrid_dueling_dqn']:
                print(f"   ✅ Using {args.model_type.upper()} - optimized for hybrid mode!")
        elif channels == 4 and not args.use_feature_channels:
            print(f"   ✅ 4-channel VISUAL-ONLY mode confirmed!")
            if args.model_type in ['hybrid_dqn', 'hybrid_dueling_dqn']:
                print(f"   ⚠️  WARNING: Hybrid model requires 8 channels! Use --use_feature_channels")
                sys.exit(1)
        elif channels == 4 and args.use_feature_channels:
            print(f"   ⚠️  Expected 8 channels but got 4 - features may not be enabled!")
        else:
            print(f"   ⚠️  Unexpected channel count: {channels}")

        # Validate hybrid model requirements
        if args.model_type in ['hybrid_dqn', 'hybrid_dueling_dqn'] and channels != 8:
            print(f"\n❌ ERROR: Hybrid models require 8-channel mode!")
            print(f"   Current: {channels} channels")
            print(f"   Solution: Use --use_feature_channels flag")
            sys.exit(1)

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
        memory_size=200000,
        min_memory_size=20000,
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
    recent_holes = []  # Now tracks average holes during play
    recent_holes_final = []  # NEW: Holes at game-over
    recent_holes_min = []  # NEW: Minimum holes during play
    recent_columns = []
    recent_completable_rows = []  # Now tracks average during play
    recent_completable_final = []  # NEW: At game-over
    recent_clean_rows = []  # Now tracks average during play
    recent_clean_rows_final = []  # NEW: At game-over
    recent_bumpiness = []  # NEW: Average during play
    recent_bumpiness_final = []  # NEW: At game-over
    recent_max_height = []  # NEW: Average during play
    recent_max_height_final = []  # NEW: At game-over
    RECENT_WINDOW = 200
    CURRICULUM_GATE_WINDOW = 150

    # Setup signal handler for graceful Ctrl+C shutdown
    def signal_handler(sig, frame):
        global _training_interrupted
        _training_interrupted = True
        print("\n\n" + "="*80)
        print("⚠️  TRAINING INTERRUPTED (Ctrl+C)")
        print("="*80)
        print("Saving progress and generating debug summary...")

        current_time = time.time() - start_time
        current_episode = episode if 'episode' in locals() else start_episode

        # Save checkpoint
        try:
            agent.save_checkpoint(current_episode, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()
            print("✅ Checkpoint and logs saved")
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")

        # Generate debug summary
        save_debug_summary(
            logger=logger,
            args=args,
            start_episode=start_episode,
            current_episode=current_episode,
            training_time=current_time,
            lines_cleared_total=lines_cleared_total,
            first_line_episode=first_line_episode,
            recent_rewards=recent_rewards,
            recent_steps=recent_steps,
            recent_lines=recent_lines,
            recent_holes=recent_holes,
            recent_columns=recent_columns,
            recent_completable_rows=recent_completable_rows,
            recent_clean_rows=recent_clean_rows,
            reward_shaper=reward_shaper,
            agent=agent,
            is_interrupted=True
        )

        print("\n✅ Training interrupted gracefully. You can resume with --resume flag.")
        print("="*80)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print("📋 Ctrl+C handler registered (will save progress before exit)")

    print(f"\n🚀 Starting improved progressive training")
    print(f"Episodes: {start_episode + 1} to {args.episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    print(f"Initial stage: {reward_shaper.get_current_stage()}")
    print("-" * 80)

    # Track previous stage for transition detection
    previous_stage = reward_shaper.get_current_stage()

    # Calculate 50% checkpoint
    total_episodes_to_train = args.episodes - start_episode
    halfway_episode = start_episode + (total_episodes_to_train // 2)
    summary_generated_at_50 = False

    # MAIN TRAINING LOOP
    for episode in range(start_episode, args.episodes):
        # Check for interruption
        if _training_interrupted:
            break

        gate_holes = gate_completable = gate_clean = None
        if recent_holes:
            hole_window = recent_holes[-CURRICULUM_GATE_WINDOW:] if len(recent_holes) >= CURRICULUM_GATE_WINDOW else recent_holes
            gate_holes = float(np.mean(hole_window))
        if recent_completable_rows:
            compl_window = recent_completable_rows[-CURRICULUM_GATE_WINDOW:] if len(recent_completable_rows) >= CURRICULUM_GATE_WINDOW else recent_completable_rows
            gate_completable = float(np.mean(compl_window))
        if recent_clean_rows:
            clean_window = recent_clean_rows[-CURRICULUM_GATE_WINDOW:] if len(recent_clean_rows) >= CURRICULUM_GATE_WINDOW else recent_clean_rows
            gate_clean = float(np.mean(clean_window))

        reward_shaper.update_curriculum_metrics(
            hole_avg=gate_holes,
            completable_avg=gate_completable,
            clean_avg=gate_clean
        )
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
        episode_component_totals = {}

        # NEW: Track metrics throughout episode (not just at game-over)
        hole_samples = []  # Sample holes every 20 steps
        hole_at_step_50 = None
        hole_at_step_100 = None
        hole_at_step_150 = None
        min_holes = float('inf')  # Track best board state

        # NEW: Track other metrics during play
        bumpiness_samples = []
        completable_samples = []
        clean_rows_samples = []
        max_height_samples = []

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

            components = getattr(reward_shaper, "last_reward_components", None)
            if components:
                for key, value in components.items():
                    if key in ('stage',):
                        continue
                    episode_component_totals[key] = episode_component_totals.get(key, 0.0) + float(value)

            # Store experience with shaped reward
            agent.remember(obs, action, shaped_reward, next_obs, done, info, raw_reward)

            # Learn every 4 steps
            if episode_steps % 4 == 0 and len(agent.memory) >= agent.min_buffer_size:
                agent.learn()

            # Update metrics
            episode_reward += shaped_reward
            episode_steps += 1

            # NEW: Sample metrics during play (every 20 steps)
            if episode_steps % 20 == 0:
                current_board = extract_board_from_obs(obs)
                current_holes = count_holes(current_board)
                hole_samples.append(current_holes)
                min_holes = min(min_holes, current_holes)

                # Track other metrics
                current_bumpiness = calculate_bumpiness(current_board)
                bumpiness_samples.append(current_bumpiness)

                current_metrics = reward_shaper.calculate_metrics(current_board, info)
                completable_samples.append(current_metrics['completable_rows'])
                clean_rows_samples.append(current_metrics['clean_rows'])

                current_heights = get_column_heights(current_board)
                current_max_height = max(current_heights) if current_heights else 0
                max_height_samples.append(current_max_height)

            # NEW: Capture holes at specific checkpoints
            if episode_steps == 50:
                checkpoint_board = extract_board_from_obs(obs)
                hole_at_step_50 = count_holes(checkpoint_board)
            elif episode_steps == 100:
                checkpoint_board = extract_board_from_obs(obs)
                hole_at_step_100 = count_holes(checkpoint_board)
            elif episode_steps == 150:
                checkpoint_board = extract_board_from_obs(obs)
                hole_at_step_150 = count_holes(checkpoint_board)

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
        holes_final = final_metrics['holes']  # Renamed: holes at game-over
        bumpiness = final_metrics['bumpiness']
        columns_used = final_metrics['columns_used']
        completable_rows = final_metrics['completable_rows']
        clean_rows = final_metrics['clean_rows']
        max_height = max(heights) if heights else 0
        outer_unused = final_metrics['outer_unused']

        # NEW: Calculate metrics from samples (use average during play as primary)
        holes_avg = np.mean(hole_samples) if hole_samples else holes_final
        holes_min = min_holes if min_holes != float('inf') else holes_final
        holes = holes_avg  # Use average for primary metric (backward compat)

        # NEW: Calculate other metric averages
        bumpiness_avg = np.mean(bumpiness_samples) if bumpiness_samples else bumpiness
        completable_avg = np.mean(completable_samples) if completable_samples else completable_rows
        clean_rows_avg = np.mean(clean_rows_samples) if clean_rows_samples else clean_rows
        max_height_avg = np.mean(max_height_samples) if max_height_samples else max_height

        # Track recent performance
        recent_rewards.append(episode_reward)
        recent_lines.append(lines_this_episode)
        recent_steps.append(episode_steps)
        recent_holes.append(holes)  # Average holes
        recent_holes_final.append(holes_final)  # Final holes
        recent_holes_min.append(holes_min)  # Min holes
        recent_columns.append(columns_used)
        recent_completable_rows.append(completable_avg)  # Average during play
        recent_completable_final.append(completable_rows)  # Final
        recent_clean_rows.append(clean_rows_avg)  # Average during play
        recent_clean_rows_final.append(clean_rows)  # Final
        recent_bumpiness.append(bumpiness_avg)  # Average during play
        recent_bumpiness_final.append(bumpiness)  # Final
        recent_max_height.append(max_height_avg)  # Average during play
        recent_max_height_final.append(max_height)  # Final

        if len(recent_rewards) > RECENT_WINDOW:
            recent_rewards.pop(0)
            recent_lines.pop(0)
            recent_steps.pop(0)
            recent_holes.pop(0)
            recent_holes_final.pop(0)
            recent_holes_min.pop(0)
            recent_columns.pop(0)
            recent_completable_rows.pop(0)
            recent_completable_final.pop(0)
            recent_clean_rows.pop(0)
            recent_clean_rows_final.pop(0)
            recent_bumpiness.pop(0)
            recent_bumpiness_final.pop(0)
            recent_max_height.pop(0)
            recent_max_height_final.pop(0)

        component_fields = {f"rc_{k}": v for k, v in episode_component_totals.items()}

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
            holes=holes,  # Average holes during play
            holes_final=holes_final,  # Holes at game-over
            holes_min=holes_min,  # Minimum holes
            holes_at_step_50=hole_at_step_50 if hole_at_step_50 is not None else '',
            holes_at_step_100=hole_at_step_100 if hole_at_step_100 is not None else '',
            holes_at_step_150=hole_at_step_150 if hole_at_step_150 is not None else '',
            columns_used=columns_used,
            completable_rows=completable_avg,  # Average during play
            completable_rows_final=completable_rows,  # Final
            clean_rows=clean_rows_avg,  # Average during play
            clean_rows_final=clean_rows,  # Final
            bumpiness=bumpiness_avg,  # Average during play
            bumpiness_final=bumpiness,  # Final
            max_height=max_height_avg,  # Average during play
            max_height_final=max_height,  # Final
            curriculum_gate_holes=gate_holes if gate_holes is not None else '',
            curriculum_gate_completable=gate_completable if gate_completable is not None else '',
            curriculum_gate_clean=gate_clean if gate_clean is not None else '',
            **component_fields
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
            avg_holes_final = np.mean(recent_holes_final) if recent_holes_final else 0
            avg_holes_min = np.mean(recent_holes_min) if recent_holes_min else 0
            avg_cols = np.mean(recent_columns) if recent_columns else 0
            avg_completable = np.mean(recent_completable_rows) if recent_completable_rows else 0
            avg_completable_final = np.mean(recent_completable_final) if recent_completable_final else 0
            avg_clean = np.mean(recent_clean_rows) if recent_clean_rows else 0
            avg_clean_final = np.mean(recent_clean_rows_final) if recent_clean_rows_final else 0
            avg_bumpiness = np.mean(recent_bumpiness) if recent_bumpiness else 0
            avg_max_height = np.mean(recent_max_height) if recent_max_height else 0

            # TWO-LINE format for better readability
            # Line 1: Episode info, reward, steps, lines
            print(f"Ep {episode+1:5d} │ {current_stage:20s} │ "
                  f"R:{episode_reward:7.1f} (Σ{avg_reward:6.1f}) │ "
                  f"Steps:{episode_steps:3d} (Σ{avg_steps:5.1f}) │ "
                  f"Lines:{lines_this_episode} (Σ{avg_lines:.2f}, Σ={lines_cleared_total}) │ "
                  f"ε:{agent.epsilon:.3f}")

            # Line 2: Board quality metrics (showing play vs final)
            print(f"       │ Holes:{holes:.0f}→{holes_final} [min:{holes_min:.0f}] (Σ{avg_holes:.1f}) │ "
                  f"Compl:{completable_avg:.1f}→{completable_rows} (Σ{avg_completable:.1f}) │ "
                  f"Clean:{clean_rows_avg:.0f}→{clean_rows} (Σ{avg_clean:.1f}) │ "
                  f"Bump:{bumpiness_avg:.0f}→{bumpiness:.0f} │ "
                  f"H:{max_height_avg:.0f}→{max_height} │ "
                  f"Cols:{columns_used}/10")

        # Save checkpoint periodically
        if (episode + 1) % args.save_freq == 0:
            agent.save_checkpoint(episode + 1, MODEL_DIR)
            logger.save_logs()
            logger.plot_progress()
            print(f"💾 Checkpoint saved at episode {episode + 1}")

        # Generate debug summary at 50% progress
        if not summary_generated_at_50 and episode + 1 >= halfway_episode:
            summary_generated_at_50 = True
            current_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"📊 HALFWAY CHECKPOINT - Episode {episode + 1}/{args.episodes} (50%)")
            print(f"{'='*80}")
            save_debug_summary(
                logger=logger,
                args=args,
                start_episode=start_episode,
                current_episode=episode + 1,
                training_time=current_time,
                lines_cleared_total=lines_cleared_total,
                first_line_episode=first_line_episode,
                recent_rewards=recent_rewards,
                recent_steps=recent_steps,
                recent_lines=recent_lines,
                recent_holes=recent_holes,
                recent_columns=recent_columns,
                recent_completable_rows=recent_completable_rows,
                recent_clean_rows=recent_clean_rows,
                reward_shaper=reward_shaper,
                agent=agent,
                is_interrupted=False
            )
            print(f"{'='*80}\n")

    # Training complete
    training_time = time.time() - start_time
    env.close()

    # Save final checkpoint
    agent.save_checkpoint(args.episodes, MODEL_DIR)
    logger.save_logs()
    logger.plot_progress()

    # Generate final comprehensive debug summary
    save_debug_summary(
        logger=logger,
        args=args,
        start_episode=start_episode,
        current_episode=args.episodes,
        training_time=training_time,
        lines_cleared_total=lines_cleared_total,
        first_line_episode=first_line_episode,
        recent_rewards=recent_rewards,
        recent_steps=recent_steps,
        recent_lines=recent_lines,
        recent_holes=recent_holes,
        recent_columns=recent_columns,
        recent_completable_rows=recent_completable_rows,
        recent_clean_rows=recent_clean_rows,
        reward_shaper=reward_shaper,
        agent=agent,
        is_interrupted=False
    )

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
