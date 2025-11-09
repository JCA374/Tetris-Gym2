# Debug Tests - Comprehensive Debugging Suite

This directory contains a systematic debugging suite to diagnose why the Tetris DQN agent has failed to clear any lines after 18,500 episodes.

## Problem Statement

After 18,500 training episodes:
- **Lines cleared**: 0 (zero)
- **Steps/episode**: 90-318 (decent survival)
- **Reward/episode**: Equals steps (only getting +1.0 survival reward)
- **Expected performance**: 100-1,000+ lines/episode at this point

The agent has learned to survive but never discovered the actual objective of Tetris.

## Test Suite Overview

### Test 1: Basic Environment (`test_1_basic_env.py`)
**Critical: Yes**

Tests if the base Tetris environment can clear lines at all.

- **Test 1A**: Random actions - can random play clear lines?
- **Test 1B**: Hard drop strategy - does biasing toward HARD_DROP help?
- **Test 1C**: Info dict contents - what's actually in the info dict?

**What it checks**:
- Is line clearing possible in the environment?
- Does the environment correctly report line clears in info dict?

**Expected result**: Random actions should clear some lines (even if rarely)

**If it fails**: Critical environment issue - check Tetris Gymnasium installation


### Test 2: Environment Wrapper (`test_2_wrapper.py`)
**Critical: Yes**

Tests if the FeatureVectorWrapper preserves the info dict, especially `number_of_lines`.

- **Test 2A**: Info dict passthrough - does wrapper preserve line clear info?
- **Test 2B**: Feature extraction - are features calculated correctly?
- **Test 2C**: Wrapped vs base comparison - do both behave identically?

**What it checks**:
- Does the wrapper break the info dict?
- Are features normalized to [0, 1]?
- Do features change over time (not static)?

**Expected result**: Wrapper should preserve all info dict fields

**If it fails**: Critical wrapper issue - fix FeatureVectorWrapper


### Test 3: Feature Extraction (`test_3_features.py`)
**Critical: No (but important)**

Tests if feature extraction correctly identifies holes, heights, bumpiness, wells.

- **Test 3A**: Empty board - all features should be zero
- **Test 3B**: Simple board - known board with predictable features
- **Test 3C**: Hole detection - specifically test hole counting
- **Test 3D**: Real gameplay - features during actual game
- **Test 3E**: Normalization - values properly in [0, 1]

**What it checks**:
- Are features extracted correctly?
- Is normalization working?

**Expected result**: All feature calculations should match expected values

**If it fails**: Feature bugs could mislead the agent


### Test 4: Reward Function (`test_4_reward.py`)
**Critical: Yes**

Tests if the reward function receives line clear info and calculates rewards correctly.

- **Test 4A**: Reward calculation - test reward function with different scenarios
- **Test 4B**: Reward during gameplay - actual rewards during play
- **Test 4C**: Info dict completeness - what's available to reward function
- **Test 4D**: Training loop simulation - exact simulation of training

**What it checks**:
- Does reward function receive `number_of_lines` from info dict?
- Are line clears rewarded correctly (+100 per line)?
- Does reward function match what's used in training?

**Expected result**: Line clears should result in rewards of 101+ (base 1 + lines * 100)

**If it fails**: Agent never learns line clearing is valuable


### Test 5: Action Distribution (`test_5_actions.py`)
**Critical: Yes**

Analyzes which actions the trained agent actually uses.

- **Test 5A**: Random baseline - what does uniform random look like?
- **Test 5B**: Trained agent actions - what does the agent actually do?
- **Test 5C**: Q-value analysis - are Q-values meaningful?
- **Test 5D**: Action sequences - patterns in agent behavior

**What it checks**:
- Does agent use HARD_DROP (action 5)? Essential for efficient play
- Is agent stuck on one action (NOOP)?
- Are Q-values differentiated (learned preferences)?
- Are there repeated patterns (stuck in loops)?

**Expected result**: Agent should use all actions, especially HARD_DROP

**If it fails**: Agent may have learned degenerate policy


## Running the Tests

### Run All Tests (Recommended)
```bash
cd /home/jonas/Code/Tetris-Gym2
python debug_tests/run_all_tests.py
```

This runs all tests in sequence with summaries and pauses between tests.

### Run Individual Tests
```bash
# Test 1: Basic environment
python debug_tests/test_1_basic_env.py

# Test 2: Wrapper
python debug_tests/test_2_wrapper.py

# Test 3: Features
python debug_tests/test_3_features.py

# Test 4: Reward function
python debug_tests/test_4_reward.py

# Test 5: Action distribution
python debug_tests/test_5_actions.py
```

### Quick Diagnostic (Tests 1, 2, 4 only)
```bash
# Run only critical tests
python debug_tests/test_1_basic_env.py
python debug_tests/test_2_wrapper.py
python debug_tests/test_4_reward.py
```

## Interpreting Results

### All Tests Pass
If all tests pass, the issue is likely:
- **Insufficient exploration**: Epsilon decayed too fast, agent never discovered line clearing
- **Action distribution problem**: Agent doesn't use HARD_DROP
- **Local optimum**: Agent learned survival-only strategy

**Solutions**:
1. Check Test 5 output - is HARD_DROP usage < 5%?
2. Reset epsilon to 0.3-0.5 temporarily to force exploration
3. Add shaped rewards (reward for filling rows, not just clearing)
4. Use curriculum learning
5. Add expert demonstrations to replay buffer

### Test 1 Fails (Environment)
Critical issue - environment can't clear lines at all.

**Solutions**:
1. Check Tetris Gymnasium version: `pip show tetris-gymnasium`
2. Reinstall: `pip install --upgrade tetris-gymnasium`
3. Test with official examples from tetris-gymnasium repo

### Test 2 Fails (Wrapper)
Critical issue - wrapper breaks info dict.

**Solutions**:
1. Review `src/env_feature_vector.py`
2. Ensure wrapper's `step()` returns original info dict
3. Check if wrapper is being used (should see "Feature vector mode ACTIVE" message)

### Test 4 Fails (Reward)
Critical issue - reward function not receiving line clear info.

**Solutions**:
1. Check info dict keys in Test 2 and Test 4 outputs
2. Verify `train_feature_vector.py` passes info dict to `simple_reward()`
3. Add debug logging to reward function

### Test 5 Shows Problems (Actions)
Agent learned degenerate policy.

**Common patterns and solutions**:
- **HARD_DROP < 5%**: Agent doesn't know how to finish placing pieces
  - Solution: Force HARD_DROP during warmup, or bias exploration
- **NOOP > 50%**: Agent learned to do nothing
  - Solution: Penalize NOOP, or remove it from action space
- **One action > 80%**: Agent stuck repeating same action
  - Solution: Reset training, increase exploration
- **Q-values all similar**: Agent didn't learn action preferences
  - Solution: Check learning is happening (test replay buffer, loss)

## Additional Diagnostic Tests (TODO)

The following tests are planned but not yet implemented:

### Test 6: Agent Exploration
- Test epsilon decay schedule
- Verify action selection (epsilon-greedy working correctly)
- Check exploration vs exploitation balance

### Test 7: Replay Buffer
- Verify transitions stored correctly
- Check if info dict preserved in buffer
- Ensure sufficient diversity in buffer

### Test 8: Model Learning
- Monitor loss values during training
- Verify Q-values update (not static)
- Check gradient flow

### Test 9: Short Diagnostic Training
- Run 100 episodes with verbose logging
- Log every action, reward, Q-value
- Verify learning loop works end-to-end

### Test 10: Baseline Comparison
- Simple rule-based agent (always HARD_DROP)
- Reward-hacking agent
- Compare to see if problem is in environment or agent

## Next Steps After Diagnosis

Based on test results, here are recommended next steps:

### Scenario 1: Environment/Wrapper Issue (Tests 1-2 fail)
1. Fix the critical bugs identified
2. Re-run tests to verify fix
3. Start fresh training run

### Scenario 2: Exploration Problem (All tests pass, Test 5 shows no HARD_DROP)
1. Reset epsilon to 0.3 or higher
2. Add epsilon floor (don't decay below 0.1)
3. Run for 1,000 more episodes with increased exploration
4. Check if agent starts clearing lines

### Scenario 3: Reward Shaping Needed (Tests pass, but no learning progress)
1. Add dense rewards:
   - Reward for filling rows (even if not cleared)
   - Reward for reducing holes
   - Penalty that grows with height
2. Use curriculum learning:
   - Start with forced line clear scenarios
   - Gradually increase difficulty
3. Consider expert demonstrations

### Scenario 4: Architecture Problem (Q-values not learning)
1. Check model is updating (run Test 8 when implemented)
2. Verify gradients flowing
3. Try different learning rate
4. Try different network architecture

## Files

```
debug_tests/
├── README.md                  # This file
├── run_all_tests.py          # Master test runner
├── test_1_basic_env.py       # Test environment can clear lines
├── test_2_wrapper.py         # Test wrapper preserves info dict
├── test_3_features.py        # Test feature extraction
├── test_4_reward.py          # Test reward function
└── test_5_actions.py         # Test action distribution
```

## Contributing

To add new tests:

1. Create `test_N_name.py` following the existing pattern
2. Add test description to this README
3. Add test to `TESTS` list in `run_all_tests.py`
4. Use clear section headers and print statements
5. Provide a "FINAL VERDICT" section with pass/fail

## Notes

- All tests are designed to be run from the project root directory
- Tests use the actual training environment and models
- Test 5 requires a trained model checkpoint (models/best_model.pth)
- Tests are non-destructive (don't modify models or logs)
