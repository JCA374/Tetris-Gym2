# Codebase Cleanup Plan

## Files to Remove (Obsolete with Feature Vector Approach)

### 1. Hybrid CNN-Specific Files (No longer needed)

**Source Files:**
- `src/feature_heatmaps.py` - Generates spatial heatmaps for CNN input
- `src/model_hybrid.py` - Hybrid dual-branch CNN model
- `src/env_wrapper.py` - 8-channel observation wrapper for CNNs

**Test Files:**
- `test_hybrid_model.py` - Tests hybrid model
- `tests/test_feature_heatmaps.py` - Tests heatmap generation
- `tests/test_feature_channels_training.py` - Tests 8-channel training
- `tests/test_4channel_wrapper.py` - Tests 4-channel wrapper

**Training/Visualization:**
- `train_progressive_improved.py` - Trains hybrid CNN models
- `visualize_features.py` - Visualizes feature heatmaps
- `config.py` - CompleteVisionWrapper for 8-channel approach

### 2. Old/Duplicate Implementations

- `src/feature_extraction.py` - Old feature extraction (replaced by `src/feature_vector.py`)
- `src/progressive_reward.py` - Old 4-stage progressive (replaced by `src/progressive_reward_improved.py`)
- `src/model_simple.py` - Simple CNN model (we use FC models now)
- `src/reward_simple.py` - Simple reward function (if exists)

### 3. Old Training/Analysis Scripts

- `train_baseline_simple.py` - Baseline CNN training
- `compare_models.py` - Compares CNN architectures
- `ablation_configs.py` - Ablation study configs for CNNs
- `run_ablation_study.py` - Runs CNN ablation studies
- `monitor_training.py` - Old training monitor (check if still used)

### 4. Old Test Files (for deprecated approaches)

- `tests/diagnose_model.py` - Diagnoses CNN models
- `tests/diagnose_training.py` - Diagnoses old training
- `tests/test_actual_line_clear.py` - Specific test (check if still relevant)
- `tests/test_agent_exploration_mix.py` - Agent exploration test (check if relevant)
- `tests/test_board_extraction_fix.py` - Board extraction test (may be useful)
- `tests/test_reward_system_complete.py` - Complete reward system test

## Files to Keep

### Core Implementation:
- `train_feature_vector.py` ✅ Main training script
- `analyze_training.py` ✅ Analysis tool
- `evaluate.py` ✅ Model evaluation
- `src/feature_vector.py` ✅ Feature extraction
- `src/model_fc.py` ✅ FC DQN models
- `src/env_feature_vector.py` ✅ Feature vector wrapper
- `src/agent.py` ✅ Core agent
- `src/model.py` ✅ Model factory (updated to support FC models)
- `src/utils.py` ✅ Logging and utilities
- `src/reward_shaping.py` ✅ Core reward functions
- `src/progressive_reward_improved.py` ✅ 5-stage curriculum

### Useful Tests:
- `tests/test_actions_simple.py` ✅ Basic action tests
- `tests/test_reward_helpers.py` ✅ Reward function tests
- `tests/verify_imports.py` ✅ Import verification

### Documentation:
- All `.md` files in root and `docs/` ✅

## Execution Plan

```bash
# 1. Move obsolete files to archive (for reference)
mkdir -p archive_files/hybrid_cnn
mkdir -p archive_files/old_training
mkdir -p archive_files/old_tests

# 2. Archive hybrid CNN files
mv src/feature_heatmaps.py archive_files/hybrid_cnn/
mv src/model_hybrid.py archive_files/hybrid_cnn/
mv src/env_wrapper.py archive_files/hybrid_cnn/
mv test_hybrid_model.py archive_files/hybrid_cnn/
mv train_progressive_improved.py archive_files/hybrid_cnn/
mv visualize_features.py archive_files/hybrid_cnn/
mv config.py archive_files/hybrid_cnn/
mv tests/test_feature_heatmaps.py archive_files/hybrid_cnn/
mv tests/test_feature_channels_training.py archive_files/hybrid_cnn/
mv tests/test_4channel_wrapper.py archive_files/hybrid_cnn/

# 3. Archive old/duplicate implementations
mv src/feature_extraction.py archive_files/old_training/
mv src/progressive_reward.py archive_files/old_training/
mv src/model_simple.py archive_files/old_training/ 2>/dev/null || true
mv src/reward_simple.py archive_files/old_training/ 2>/dev/null || true

# 4. Archive old training/analysis scripts
mv train_baseline_simple.py archive_files/old_training/
mv compare_models.py archive_files/old_training/
mv ablation_configs.py archive_files/old_training/
mv run_ablation_study.py archive_files/old_training/
mv monitor_training.py archive_files/old_training/ 2>/dev/null || true

# 5. Archive old test files
mv tests/diagnose_model.py archive_files/old_tests/
mv tests/diagnose_training.py archive_files/old_tests/
mv tests/test_actual_line_clear.py archive_files/old_tests/
mv tests/test_agent_exploration_mix.py archive_files/old_tests/
mv tests/test_board_extraction_fix.py archive_files/old_tests/
mv tests/test_reward_system_complete.py archive_files/old_tests/

# 6. Update .gitignore to exclude archived files
echo "# Archived files" >> .gitignore
echo "archive_files/" >> .gitignore

# 7. Commit the cleanup
git add -A
git commit -m "Archive obsolete hybrid CNN and old training files

Moved to archive_files/:
- Hybrid CNN implementation (feature heatmaps, models, wrappers)
- Old training scripts (baseline, comparisons, ablation studies)
- Duplicate implementations (old feature extraction, old progressive reward)
- Obsolete test files

Kept only feature vector approach files and essential utilities.
"
```

## Impact Analysis

**Files Removed**: ~23 files
**Lines of Code Removed**: ~5000+ lines

**Benefits:**
- Cleaner codebase focused on proven approach
- Less confusion about which files to use
- Easier maintenance
- Faster navigation

**Safety:**
- All files archived, not deleted
- Can be recovered if needed
- Git history preserved

## Alternative: Delete Instead of Archive

If you're confident and want a cleaner repo:

```bash
# Delete instead of archiving
git rm src/feature_heatmaps.py
git rm src/model_hybrid.py
# ... etc for all files
git commit -m "Remove obsolete hybrid CNN implementation"
```

Files will still be available in git history if needed.

## Recommendation

**Archive first** (safer, reversible), then **delete later** if truly not needed after 1-2 months of using feature vector approach successfully.
