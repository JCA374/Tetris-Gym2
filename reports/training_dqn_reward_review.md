# Tetris DQN Training & Reward System Review

Generated: 2025-11-04  
Log source: `logs/improved_20251103_214736`

---

## 1. Run Snapshot
- Episodes: 0 → 10 000 (Stage 5 reached at episode 5000)
- Final epsilon: 0.010 (adaptive schedule)
- Replay buffer size: 10 000 (default `Agent`)
- Optimizer: Adam, lr = 5e-4 in this run
- Frames per episode: mean 160.9 (peaks 300+)  
  ↳ Learner updates every 4 environment steps (≈40–70 updates/episode once buffer fills)
- Shaped-return range (Stage 5): clipped to ±1000; empirical 5th–95th percentile ≈ -900 to +27 000 due to large bonuses

## 2. Curriculum Outcomes

| Stage | Episodes | Avg holes | Lines/ep | Avg completable rows | Avg clean rows | Avg steps |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| foundation | 500 | 42.63 | 0.0000 | 0.006 | 6.18 | 38.8 |
| clean_placement | 500 | 46.52 | 0.0000 | 0.006 | 6.46 | 32.4 |
| spreading_foundation | 1000 | 63.18 | 0.0150 | 0.117 | 4.42 | 97.0 |
| clean_spreading | 3000 | 54.46 | 0.0617 | 0.170 | 3.42 | 147.9 |
| line_clearing_focus | 5000 | 57.09 | 0.0680 | 0.182 | 2.08 | 206.5 |

Key observations:
- Hole count never drops below ~40 in any stage; Stage 5 average is higher than Stage 4.
- Completable rows remain <0.2 on average, so the agent rarely builds almost-full rows.
- Lines cleared remain near zero across the run (total 540 lines in 10 000 episodes, 0.054 lines/episode).
- Despite poor board quality, shaped rewards climb to 20 000+ because spread and column bonuses dominate penalties.

## 3. Training Loop Assessment

**Replay and batching**
- Memory size 10 000 is tight for 10 000+ episodes with long trajectories. Fresh experiences rapidly evict early-stage clean placements, limiting variety during line-clearing focus.
- Uniform sampling plus 4-step update cadence means many experiences are revisited while still correlated; consider prioritized replay or larger buffers.

**Epsilon schedule**
- Adaptive schedule is tuned for ≥25 k episodes, but this run only spans 10 k. Epsilon drops to <0.1 by ~6000 episodes, potentially curtailing exploration of clean-up strategies once Stage 5 begins.
- Exploration distribution in `select_action` heavily favors lateral moves and hard drops, which is good for spreading but reduces incentive to soft drop/rotate that often produce cleaner stacks.

**Reward scaling & stability**
- Stage rewards multiply base env reward by 100 and add large bonuses (e.g., +400 for Tetris before additional multipliers). Q-targets therefore reach tens of thousands, while MSE loss uses default unscaled targets—can cause exploding gradients or value overestimation without gradient rescaling.
- Clipping final shaped reward to ±1000 in Stage 5 helps, but earlier stages clip at ±600; as soon as Stage 5 is entered, bootstrapping jumps in magnitude which destabilizes target network alignment.

**Logging & diagnostics**
- Episode CSV includes `clean_rows`, `completable_rows`, and `holes`, which proved critical. Recommend also logging `reward_components` to trace which bonuses dominate.

## 4. DQN Architecture Review

| Component | Assessment |
| --- | --- |
| Encoder | 3-layer CNN (8×8/4 stride, 4×4/2 stride, 3×3/1). On a 20×10 input this downsamples to 2×2 feature map; heavy early stride may discard detail (especially vertical structure). |
| Fully connected head | 512 → 256 → actions with dropout 0.1; adequate capacity, but lacks skip connections/residuals common in modern Tetris agents. |
| Target network | Soft-updated every 1000 gradient steps (hard copy). Works but could be improved with Polyak averaging. |
| Regularization | Dropout in training but not inference; no batch-norm. With large reward magnitudes this can still overfit to noisy targets. |
| Observation preprocessing | Agent converts obs to float32, normalizes by /255 if needed, and permutes to CHW. However, CompleteVisionWrapper already outputs binary {0,1}; dividing by 255 leaves values unchanged but is harmless. |

Implications:
- Aggressive strides mean single-cell holes may disappear after conv1, making it hard for the network to tell whether a column is hole-free.
- Replay buffer limit + high dropout can hamper consistent value estimates once Stage 5 reward signals change sharply.

## 5. Reward Shaping & Curriculum Review

**Stage 1–2 (Foundation / Clean Placement)**
- Base reward: `base_reward * 100`, mild hole penalty (0.3→1.0 per hole). Since base env reward is often zero, shaped reward becomes dominated by survival bonuses (up to +40) even with 40+ holes. Agent learns to survive, not to clean.

**Stage 3 (Spreading Foundation)**
- Spread bonuses explode (`+40*spread`, `+8*columns`, `-15*outer_unused`). With 9–10 columns active, rewards exceed +100 even if holes >60. Center-stacking penalty helps coverage but not cleanliness.

**Stage 4 (Clean Spreading)**
- Hole penalty increases to `-2.5*holes` (≈ -150 for 60 holes) but spread/columns bonuses (~+50*spread + 12*columns) still net positive (~+170) before survival bonus. Conditional survival only turns off at 30+ holes, so the agent still accumulates positive reward while maintaining messy boards.

**Stage 5 (Line Clearing Focus)**
- Prior to the fix just applied, hole penalty (-3.5*holes) was insufficient against +60*spread + 15*columns + survival, yielding rewards >20 000 with zero lines. Completablerow bonus (+30 each) rarely triggered because the agent never built them.
- After the fix (current repo state), Stage 5 now:  
  - `-5*holes` penalty  
  - `+45*completable_rows`  
  - Spread/column bonuses scaled by cleanliness factor (fade out once holes >20)  
  - Survival bonus pays only when holes <20  
  Expectation: agent should now be forced to flatten board before receiving large positive reward.

**Cross-stage consistency**
- Stage transitions depend solely on episode index; there is no performance gating. Agent can reach Stage 5 while still failing Stage 2 criteria, which happened here. Consider gating on rolling metrics (e.g., average holes <20) before advancing.

## 6. Root Cause: Why Lines Are Not Clearing

1. **Reward imbalance** – Spread/column bonuses outweigh hole penalties across all stages, so the agent receives positive shaped rewards despite extremely messy boards. High survival bonuses when `holes >30` further reinforce this behavior.
2. **Curriculum advancement without mastery** – Stage advancement is time-based, so the agent enters spreading phases with hole counts already >40. Line-clearing signals are never experienced on clean boards, so Q-values for clearing setups stay untrained.
3. **Sparse positive feedback for near-complete rows** – Completable rows average 0.18/episode. Without substantial bonuses, the agent has little incentive to invest 3–4 moves building one.
4. **Network observation fidelity** – CNN strides reduce the ability to spot single-cell gaps; the policy may not "see" that it needs to fill holes before clearing.
5. **Replay buffer churn** – Small buffer recycles late-stage, hole-heavy states, so the learner keeps reinforcing survival-in-holes behavior even if earlier episodes briefly produced cleaner stacks.

## 7. Additional Risks

- **Value explosion**: shaped rewards reaching 20 000+ lead to large TD targets, risking gradient explosions despite clipping. Monitor loss spikes; consider reward normalization (e.g., divide shaped reward by 100 before feeding into DQN).
- **Epsilon floor**: final epsilon 0.01 means ~1% random actions; once the agent converges to hole-heavy survival, it rarely explores drastic clean-up sequences.
- **Logging gaps**: no per-component reward breakdown, so diagnosing which bonus dominates required manual reasoning.

## 8. Recommendations

### Immediate (already partly implemented)
1. **Stage 5 penalties/bonuses** – Completed in `src/progressive_reward_improved.py`: higher hole penalties, bigger completable-row bonus, cleanliness scaling of spread/column, tighter survival gates.
2. **Retrain** – Resume for 3000+ episodes or restart from Stage 4 checkpoint to let the policy adapt to new reward landscape. Monitor `holes`, `completable_rows`, and `lines_cleared` every 100 episodes.

### Short Term
1. **Add metric-gated curriculum** – Require rolling averages (e.g., `holes < 25` & `completable_rows > 0.5`) before moving beyond clean placement.
2. **Increase replay buffer** – Set `memory_size` ≥ 100 000 and ensure `MIN_MEMORY_SIZE` is met before learning. This retains rare clean states.
3. **Log reward components** – Modify `calculate_reward` to return component breakdown for debug CSV.
4. **Normalize shaped rewards** – Divide shaped reward by constant (e.g., 50) before storing to replay; adjust bonuses accordingly to keep Q-targets comparable across stages.

### Medium Term
1. **Refine CNN encoder** – Replace first conv with stride-2 kernels or add small-kernel conv before stride-4 to preserve hole detail. Alternatively, feed engineered features (column heights, hole mask) alongside image input.
2. **Introduce performance-based curricula** – Evaluate stage success conditions every 200 episodes and, if unmet, delay advancement or temporarily revert to earlier stage shaping.
3. **Adopt prioritized replay** – Helps the agent revisit rare clean-board experiences and line clears once reward signals become sparser.
4. **Adjust exploration distribution** – Increase probability for soft drop and rotations during exploration to encourage cleaner placements rather than constant hard drops.

### Monitoring
- Track `avg holes`, `avg completable_rows`, and `avg lines` over a rolling 200-episode window. Training should stay in the current stage until `holes` trend downward and `completable_rows` trend upward.
- Capture sample boards every 200 episodes for visual regression.
- Flag runs where shaped reward > 15 000 but `lines_cleared == 0`—this indicates the agent is again exploiting survival bonuses.

---

Prepared by: Codex training agent audit  
Repository state: post Stage 5 reward rebalancing (`src/progressive_reward_improved.py`)  
Next action: retrain with updated shaping and monitor for rising completable rows before expecting consistent line clears.

## 9. Implementation Progress (Current Repo State)

- **Curriculum gating** – Stage 5 entry now requires rolling averages of `holes ≤ 25`, `completable_rows ≥ 0.5`, and `clean_rows ≥ 5`, preventing premature progression while the board is still messy.
- **Replay buffer & warmup** – Agent buffer enlarged to 200 k transitions with a 20 k warmup; learning waits until the buffer contains a broad mix of experiences.
- **Exploration tuning** – Adaptive epsilon schedule keeps exploration ≥0.18 through 70 % of training and the exploratory action prior now favors horizontal moves, soft drops, and rotations over constant hard drops.
- **CNN encoder refresh** – Convolutional stack replaced with small-stride filters so single-cell holes survive the feature extractor; dueling variant updated accordingly.
- **Reward diagnostics** – Stage 4/5 shapers record per-step component breakdowns; episode logs now include aggregated reward-component totals along with curriculum gate metrics for regression analysis.

Monitor the next run with these changes to verify that rolling hole averages fall below the gate threshold and that `rc_completable_bonus` terms grow before line clears appear.
