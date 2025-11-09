# DQN Decision-Making: How the Agent Evaluates Piece Placements

**Author**: Analysis of Tetris-Gym2 Hybrid DQN Architecture
**Date**: 2025-11-07
**Purpose**: Detailed technical explanation of how the DQN agent decides where to place pieces and evaluates whether placements are good or bad

---

## Executive Summary

The DQN agent makes placement decisions through a **two-part evaluation system**:

1. **Q-Value Prediction** (Neural Network): Estimates long-term value of each action
2. **Reward Signal** (Immediate Feedback): Tells the agent if the placement was good/bad

The key insight: **The agent doesn't directly "calculate" if a placement is good** - instead, it **learns** through trial and error by:
- Trying millions of placements
- Receiving immediate reward feedback
- Training a neural network to predict which actions lead to high future rewards
- Gradually improving its placement strategy through experience

---

## Part 1: The Q-Value System (Future Value Estimation)

### What Are Q-Values?

Q-values (Quality values) estimate the **total future reward** expected from taking an action in a given state.

**Formula:**
```
Q(state, action) = Expected cumulative reward from taking 'action' in 'state'
```

**Example:**
```
State: Tetris board with I-piece at top
Actions available: LEFT, RIGHT, DOWN, ROTATE_CW, ROTATE_CCW, HARD_DROP, SWAP, NOOP

Q-values (predicted by neural network):
- Q(state, LEFT)      = 45.2  ← Low value, not promising
- Q(state, RIGHT)     = 52.1  ← Medium value
- Q(state, HARD_DROP) = 87.3  ← HIGH value, agent chooses this!
- Q(state, SWAP)      = 38.5  ← Low value
- ...
```

The agent picks the action with the **highest Q-value** (greedy policy).

### How Q-Values Are Computed

The Q-values are computed by a **Convolutional Neural Network (CNN)** that processes the Tetris board observation.

**Complete Pipeline** (`src/model_hybrid.py:109-157`):

```
1. INPUT: Observation (20×10×8 array)
   - Visual channels (0-3): Board, Active piece, Holder, Queue
   - Feature channels (4-7): Holes, Heights, Bumpiness, Wells

2. DUAL-BRANCH CNN PROCESSING:

   Visual Branch (Channels 0-3):
   ┌─────────────────────────────────────────┐
   │ Conv2d(4→32) + ReLU                     │  Learn spatial patterns
   │ Conv2d(32→64) + ReLU + Stride 2         │  (edges, shapes, Tetromino forms)
   │ Conv2d(64→64) + ReLU                    │
   │ Flatten → 3,200 features                │
   └─────────────────────────────────────────┘

   Feature Branch (Channels 4-7):
   ┌─────────────────────────────────────────┐
   │ Conv2d(4→16) + ReLU                     │  Understand spatial distribution
   │ Conv2d(16→32) + ReLU + Stride 2         │  of pre-computed features
   │ Flatten → 1,600 features                │  (simpler - already meaningful)
   └─────────────────────────────────────────┘

3. FUSION AND Q-VALUE OUTPUT:

   Concatenate [visual, feature] → 4,800 features
           ↓
   FC Layer: 4800 → 512 (+ ReLU + Dropout 0.1)
           ↓
   FC Layer: 512 → 256 (+ ReLU + Dropout 0.1)
           ↓
   FC Layer: 256 → 8 Q-values (one per action)
           ↓
   OUTPUT: [Q_left, Q_right, Q_down, Q_rotate_cw, Q_rotate_ccw, Q_hard_drop, Q_swap, Q_noop]
```

**Code Reference**: `src/model_hybrid.py:109-157`

### Key CNN Design Choices

**Why dual-branch?**
- Visual channels need different processing than feature channels
- Visual CNN learns spatial patterns (edges, shapes)
- Feature CNN just needs to understand distribution of already-meaningful values
- Separate processing preserves signal strength of explicit features

**Why these layer sizes?**
- Visual branch: 4→32→64→64 (deeper, more parameters for complex pattern recognition)
- Feature branch: 4→16→32 (simpler, features already meaningful)
- Dropout: 0.1 (not 0.3) - RL needs consistent exploration

---

## Part 2: Action Selection (Epsilon-Greedy)

Once the network outputs 8 Q-values, the agent must **select an action**.

### Greedy Selection (Exploitation)

**Code**: `src/agent.py:220-226`

```python
# Exploitation: Choose action with highest Q-value
self.q_network.eval()  # Turn OFF dropout for inference
with torch.no_grad():
    state_tensor = self._preprocess_state(state)
    q_values = self.q_network(state_tensor)
    return q_values.max(1)[1].item()  # argmax Q-value
```

**Example:**
```
Q-values: [45.2, 52.1, 60.3, 55.0, 48.7, 87.3, 38.5, 25.1]
          LEFT  RIGHT DOWN  ROT_CW ROT_CCW HARD  SWAP  NOOP
                                            ^^^^
                                            Max value at index 5
Action chosen: HARD_DROP (index 5)
```

### Epsilon-Greedy Exploration

**Problem**: If the agent always picks the highest Q-value, it never tries new things!

**Solution**: With probability **epsilon**, take a **random action** instead.

**Code**: `src/agent.py:210-226`

```python
if np.random.rand() > self.epsilon:
    # Exploit: Use Q-values (greedy)
    return argmax(q_values)
else:
    # Explore: Random action (biased distribution)
    return random_action()
```

**Epsilon Schedule** (Adaptive):
- Episodes 0-8,750: ε = 1.0 → 0.15 (high exploration)
- Episodes 8,750-17,500: ε = 0.15 → 0.05 (moderate exploration)
- Episodes 17,500+: ε = 0.05 → 0.01 (mostly exploitation)

**Code Reference**: `src/agent.py:94-145`

### Exploration Action Distribution

During exploration, actions are **not uniformly random** - they're biased toward meaningful moves:

```
LEFT:        22%  (horizontal positioning important)
RIGHT:       22%
DOWN:        15%  (controlled descent)
ROTATE_CW:   13%  (rotation crucial)
ROTATE_CCW:  13%
HARD_DROP:   10%  (risky but fast)
SWAP:        5%   (strategic, less frequent)
NOOP:        0%   (disabled - waste of time)
```

**Code Reference**: `src/agent.py:229-260`

---

## Part 3: The Reward Signal (How Agent Learns Good vs Bad)

### The Core Learning Mechanism

**Q-values are initially random** - the network starts knowing nothing!

The agent learns through **reward signals** that tell it if placements were good or bad.

**Reward Pipeline:**

```
1. Agent takes action (e.g., HARD_DROP)
2. Environment updates (piece placed)
3. Reward calculator analyzes the NEW board state
4. Reward signal sent back to agent (+100 for good, -50 for bad, etc.)
5. Agent updates Q-values to predict these rewards better
```

### What Makes a Placement Good or Bad?

The reward system evaluates placements using **multiple metrics** calculated from the board state:

**Core Metrics** (`src/reward_shaping.py`):

| Metric | Calculation | Good Value | Bad Value |
|--------|-------------|------------|-----------|
| **Holes** | Empty cells with blocks above | 0-5 | 30+ |
| **Height** | Sum of column heights | <100 | >200 |
| **Bumpiness** | Sum of height differences | <10 | >30 |
| **Lines Cleared** | Rows completed | 1-4 | 0 |
| **Completable Rows** | Rows nearly full | 3-5 | 0 |
| **Clean Rows** | Rows without holes | 10+ | <3 |
| **Column Spread** | # columns used | 10 | <5 |
| **Center Stacking** | Overuse of center columns | 0% | >60% |

### Progressive Reward Shaping (5 Stages)

The reward function **changes over training** to teach concepts progressively:

**Stage 1: Foundation (0-500 episodes)**
```python
reward = base_reward * 100.0
reward -= 0.3 * holes          # Gentle hole penalty
reward -= 0.02 * height        # Minimal height penalty
reward += min(steps * 0.8, 40) # Strong survival bonus
```
**Goal**: Don't die immediately, explore actions

**Stage 2: Clean Placement (500-1000 episodes)**
```python
reward -= 2.0 * holes          # Stronger hole penalty
reward += hole_reduction * 20  # Reward fixing holes
reward -= 0.05 * height
reward += survival_bonus
```
**Goal**: Minimize holes, learn clean placement

**Stage 3: Spreading Foundation (1000-2000 episodes)**
```python
reward -= 3.0 * holes
reward += columns_used * 10    # Reward using many columns
reward -= center_penalty       # Penalize center stacking
reward += spread_bonus
```
**Goal**: Use all columns, avoid center stacking

**Stage 4: Clean Spreading (2000-5000 episodes)**
```python
reward -= 4.0 * holes          # Even stronger hole penalty
reward += completable_rows * 35
reward += clean_rows * 5
reward += spread * 50          # Strong spread bonus
reward -= outer_unused * 25    # Penalize unused edges
```
**Goal**: Spread placement while maintaining cleanliness

**Stage 5: Line Clearing (5000+ episodes)**
```python
reward -= 5.0 * holes
reward += lines_cleared * 150 * quality
reward += completable_rows * 45
reward += clean_rows * 6
reward += spread * 60
# Bonus for multi-line clears:
if lines == 2: reward += 50
if lines == 3: reward += 150
if lines == 4: reward += 400  # Tetris!
```
**Goal**: Maximize line clears with quality placement

**Code Reference**: `src/progressive_reward_improved.py:162-465`

### Reward Examples

**Good Placement Example:**
```
Action: HARD_DROP (places I-piece vertically in hole)
Result:
- Holes: 8 → 5 (reduced by 3)
- Lines cleared: 1
- Height: 120 → 110
- Bumpiness: 12 → 8

Reward calculation (Stage 5):
+ Base: 0.0 * 100 = 0
- Holes: -5.0 * 5 = -25
+ Hole reduction: 3 * 25 = +75
+ Line bonus: 1 * 150 * 0.9 = +135
- Height: -0.06 * 110 = -6.6
+ Survival: +20
─────────────────────
Total: +198.4 ✅ GOOD!
```

**Bad Placement Example:**
```
Action: HARD_DROP (places O-piece creating holes)
Result:
- Holes: 8 → 12 (increased by 4)
- Lines cleared: 0
- Height: 120 → 125
- Bumpiness: 12 → 18

Reward calculation (Stage 5):
+ Base: 0.0 * 100 = 0
- Holes: -5.0 * 12 = -60
+ Hole reduction: -4 * 25 = 0 (no bonus for increase)
+ Line bonus: 0
- Height: -0.06 * 125 = -7.5
- Bumpiness: -0.5 * 18 = -9
+ Survival: +15
─────────────────────
Total: -61.5 ❌ BAD!
```

---

## Part 4: The Learning Algorithm (Temporal Difference Learning)

### How Q-Values Are Updated

The agent doesn't just use rewards - it uses them to **update Q-value predictions**.

**Bellman Equation** (core of Q-learning):
```
Q(s, a) ← Q(s, a) + α [r + γ * max Q(s', a') - Q(s, a)]
                        └─────────────────┘
                          TD Target
```

**In words:**
1. Get current Q-value prediction: `Q(s, a)`
2. Take action, observe reward `r` and next state `s'`
3. Calculate TD Target: `r + γ * max Q(s', a')`
   - `r`: Immediate reward (how good was this placement?)
   - `γ * max Q(s', a')`: Best future Q-value from next state (discounted)
4. Update Q-value toward the target

**Implementation** (`src/agent.py:301-348`):

```python
def learn(self):
    # Sample batch from replay memory
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states_tensor = torch.tensor(states)
    actions_tensor = torch.tensor(actions)
    rewards_tensor = torch.tensor(rewards)
    next_states_tensor = torch.tensor(next_states)
    dones_tensor = torch.tensor(dones)

    # Current Q-values: Q(s, a)
    current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

    # Target Q-values: r + γ * max Q(s', a')
    with torch.no_grad():
        next_q_values = self.target_network(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

    # Loss: Mean squared error between current and target
    loss = mse_loss(current_q_values, target_q_values)

    # Backpropagation: Update network weights
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Experience Replay

**Problem**: Learning from consecutive experiences is inefficient (correlated).

**Solution**: Store experiences in a **replay buffer** and sample randomly.

**Process:**
1. Agent takes action, stores `(state, action, reward, next_state, done)` in memory
2. Memory holds up to 200,000 experiences
3. Every step, sample random batch of 32 experiences
4. Learn from diverse experiences, not just recent ones

**Code Reference**: `src/agent.py:277-285`

### Target Network

**Problem**: Q-value target keeps changing as we update the network (unstable).

**Solution**: Use a **separate target network** that updates slowly.

**Process:**
1. Main network (q_network): Updated every step
2. Target network (target_network): Copied from main every 1000 steps
3. Target network provides stable Q-value targets during learning

**Code Reference**: `src/agent.py:340-342`

---

## Part 5: The Complete Decision Loop

### Training Loop (Single Episode)

```
Episode Start:
    state = reset_environment()

    While not done:
        1. ACTION SELECTION (src/agent.py:199-260)
           ├─ With probability ε: random action (explore)
           └─ With probability 1-ε: argmax Q(state) (exploit)

        2. ENVIRONMENT STEP
           next_state, base_reward, done, info = env.step(action)

        3. REWARD CALCULATION (src/progressive_reward_improved.py:102-465)
           shaped_reward = calculate_progressive_reward(
               state, action, base_reward, done, info
           )

        4. STORE EXPERIENCE (src/agent.py:277-285)
           memory.append((state, action, shaped_reward, next_state, done))

        5. LEARN (src/agent.py:301-348)
           IF memory has enough samples:
               batch = sample_random(memory, batch_size=32)
               current_q = q_network(states)
               target_q = rewards + γ * max(target_network(next_states))
               loss = mse_loss(current_q, target_q)
               backpropagate(loss)
               update_q_network_weights()

        6. UPDATE STATE
           state = next_state

    End Episode:
        - Decay epsilon
        - Log metrics
        - Save checkpoint
```

### How the Agent "Sees" Good Placements

The agent doesn't have explicit rules like "fill holes" or "avoid height" - instead:

1. **Early Training**: Random exploration
   - Q-values are random noise
   - Agent tries everything
   - Reward signals start teaching: +200 for line clear, -60 for holes

2. **Mid Training (1000-5000 episodes)**: Pattern emergence
   - Q-network learns: "HARD_DROP in this pattern → +150 reward"
   - Weights adjust to recognize good board states
   - Q-values become meaningful: Q(fill_hole) = 85, Q(create_hole) = 20

3. **Late Training (10,000+ episodes)**: Strategic play
   - Network recognizes complex patterns
   - Q-values predict long-term consequences
   - Agent "knows" that clearing lines → survival → more future reward

**The key insight**: The agent never calculates "this placement has 3 holes so it's bad" - instead, it learned through 10,000+ episodes that "placements creating holes led to low rewards, so Q-values for hole-creating actions should be low."

---

## Part 6: Visualizing the Decision Process

### Example: Deciding Where to Place an I-Piece

**Scenario**: I-piece at top, board has a 4-deep hole on the left

**Step 1: Observe State**
```
Board (20×10×8):
Channel 0 (Board):     Channel 4 (Holes heatmap):
█ · · · · · · · · █    0 1 0 0 0 0 0 0 0 0
█ · · · · · · · · █    0 1 0 0 0 0 0 0 0 0
█ · · █ █ · · · · █    0 1 0 0 0 0 0 0 0 0
█ · · █ █ █ · · █ █    0 1 0 0 0 0 0 0 0 0
█ █ █ █ █ █ █ · █ █    0 0 0 0 0 0 0 1 0 0
                       (High values = holes)

Channel 1 (Active):    Channel 5 (Height map):
· · I · · · · · · ·    0 5 3 4 4 3 2 3 2 4
· · I · · · · · · ·    (normalized 0-1)
· · I · · · · · · ·
· · I · · · · · · ·
```

**Step 2: CNN Processing**
```
Visual Branch: Recognizes I-piece shape and board structure
Feature Branch: Sees hole at column 1, height variance
Combined: 4,800 features → 512 → 256 → 8 Q-values
```

**Step 3: Q-Value Output**
```
Action         Q-Value   Interpretation
─────────────────────────────────────────
LEFT           92.5      HIGHEST - fills hole!
RIGHT          45.3      Meh
DOWN           38.7      Wastes time
ROTATE_CW      25.1      Wrong orientation
ROTATE_CCW     24.8      Wrong orientation
HARD_DROP      12.3      Current position is bad
SWAP           48.2      Maybe save I-piece
NOOP           5.1       Useless
```

**Step 4: Action Selection**
- If ε = 0.1: 90% chance pick LEFT (exploit), 10% chance random (explore)
- Assume exploitation: Pick LEFT

**Step 5: Execute and Reward**
```
Agent moves left, then hard drops
Result: I-piece fills 4-deep hole, clears 1 line

Reward calculation:
+ Hole reduction: 4 * 25 = +100
+ Line clear: 1 * 150 * 0.95 = +142.5
+ Height reduction: +15
+ Survival: +20
Total: +277.5 ✅ EXCELLENT!
```

**Step 6: Learning Update**
```
Before:  Q(state, LEFT) = 92.5
Target:  reward + γ * max Q(next_state) = 277.5 + 0.99 * 85 = +361.65
Update:  Q(state, LEFT) ← 92.5 + α * (361.65 - 92.5)
After:   Q(state, LEFT) ≈ 95.8

Next time agent sees similar state → even more likely to pick LEFT!
```

---

## Part 7: Key Insights and Design Decisions

### Why This Architecture Works

**1. Dual-Branch CNN**
- Visual channels (board, pieces) need complex pattern recognition
- Feature channels (holes, heights) are already meaningful
- Separate processing preserves feature signal strength
- Result: 5x faster learning than visual-only (1.0 vs 0.21 lines/ep at 10K)

**2. Progressive Reward Shaping**
- Can't learn everything at once
- 5-stage curriculum teaches concepts incrementally
- Each stage builds on previous mastery
- Prevents "learned helplessness" from overwhelming penalties

**3. Epsilon-Greedy Exploration**
- Pure exploitation → stuck in local optimum
- Pure exploration → never improves
- Adaptive schedule: high exploration early, low exploitation late
- Biased exploration focuses on meaningful actions

**4. Experience Replay + Target Network**
- Experience replay breaks temporal correlation
- Target network provides stable learning targets
- Together: prevents catastrophic forgetting and oscillation

**5. Dropout = 0.1 (Critical Fix!)**
- Original bug: 0.3 dropout was too high
- Result: 30% neurons randomly off during inference
- Fix: 0.1 dropout + proper train/eval modes
- Impact: Consistent Q-values, stable learning

### Current Performance

**At 10,000 episodes (Hybrid DQN):**
- Lines cleared: ~1.0 per episode (5x baseline of 0.21)
- Holes: ~24 (2x better than baseline 48)
- Stage: 4 (clean_spreading)
- Epsilon: ~0.12 (mostly exploitation)

**Expected at 15,000 episodes (with curriculum fix):**
- Lines cleared: 2-3 per episode (10-15x baseline)
- Holes: <15 during play
- Stage: 5 (line_clearing_focus)
- Strategic play: Tetris clears, combo chains

---

## Part 8: Limitations and Future Work

### Current Limitations

**1. No Explicit Planning**
- Agent doesn't "think ahead" multiple pieces
- Only considers immediate next action
- No lookahead like "if I place here, next O-piece will..."

**2. Limited Long-Term Memory**
- Replay buffer: 200,000 transitions (~400 episodes)
- Forgets old experiences
- Can't reference "that one time 5000 episodes ago..."

**3. Action Granularity**
- 8 discrete actions (LEFT, RIGHT, etc.)
- No direct "place piece at position X with rotation Y"
- Must chain multiple actions for complex placements

**4. Curriculum Still Episode-Based**
- Stages transition at fixed episodes
- Doesn't adapt to individual agent learning rate
- Could benefit from fully performance-based transitions

### Potential Improvements

**1. Monte Carlo Tree Search (MCTS)**
- Look ahead 2-3 pieces
- Simulate placements, pick best sequence
- Used by champion Tetris bots

**2. Prioritized Experience Replay**
- Learn more from surprising transitions
- Faster learning on rare but important events

**3. Dueling Network Architecture**
- Separate value and advantage streams
- Already implemented in `HybridDuelingDQN`
- Expected 10-20% improvement

**4. Multi-Piece Observation**
- Show next 3-5 pieces in observation
- Agent can plan better sequences
- Requires larger CNN or LSTM

---

## Conclusion

### How Does the Agent Decide Where to Place Pieces?

**Short Answer:**
The agent uses a trained neural network to predict "Q-values" (long-term rewards) for each possible action, then picks the action with the highest Q-value.

**Complete Answer:**
1. **Observe**: Convert board to 8-channel array (visual + features)
2. **Process**: Dual-branch CNN extracts patterns → 8 Q-values
3. **Decide**: Pick action with highest Q-value (90% of the time) or random (10%)
4. **Execute**: Take action, observe result
5. **Evaluate**: Calculate shaped reward based on metrics (holes, lines, spread)
6. **Learn**: Update Q-network to better predict rewards
7. **Repeat**: 10,000+ episodes until expert play emerges

### How Does It Know If a Placement Is Good or Bad?

**Short Answer:**
It receives immediate reward signals that evaluate the board state after placement using metrics like holes, lines cleared, height, and spread.

**Complete Answer:**
The agent uses a **two-part evaluation**:

1. **Immediate Reward (explicit calculation)**:
   - Calculate holes: `count_holes(board)` → 12 holes → penalty -60
   - Calculate lines: `count_lines_cleared()` → 2 lines → bonus +300
   - Calculate spread: `column_spread()` → 0.8 → bonus +48
   - Total reward: +288 → GOOD placement!

2. **Future Value (learned prediction)**:
   - CNN predicts: "similar boards led to +450 total reward"
   - Q-value encodes this prediction
   - Agent learns: "boards with few holes → survive longer → more lines → higher reward"

**The Magic:**
Through 10,000+ episodes of trial and error, the agent learns to recognize good board states **without** explicit rules. The reward system teaches, the Q-network learns, and strategic play emerges naturally.

---

## Appendix: Key Code References

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Hybrid CNN | `src/model_hybrid.py` | 19-157 | Dual-branch architecture |
| Action selection | `src/agent.py` | 199-260 | Epsilon-greedy + exploration bias |
| Q-learning | `src/agent.py` | 301-348 | TD learning + replay + target network |
| Reward Stage 1 | `src/progressive_reward_improved.py` | 162-187 | Foundation stage |
| Reward Stage 2 | `src/progressive_reward_improved.py` | 189-231 | Clean placement stage |
| Reward Stage 3 | `src/progressive_reward_improved.py` | 233-284 | Spreading foundation stage |
| Reward Stage 4 | `src/progressive_reward_improved.py` | 286-352 | Clean spreading stage |
| Reward Stage 5 | `src/progressive_reward_improved.py` | 354-465 | Line clearing stage |
| Metrics calculation | `src/reward_shaping.py` | All | Holes, heights, bumpiness, etc. |
| Epsilon schedule | `src/agent.py` | 94-145 | Adaptive decay schedule |

---

**End of Report**
