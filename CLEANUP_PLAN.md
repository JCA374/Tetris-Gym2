# Repository Cleanup Plan

## Current State
18 MD files in root directory - too many!

## Categorization

### KEEP IN ROOT (Essential)
1. **README.md** - Main project documentation
2. **CLAUDE.md** - Instructions for Claude Code
3. **HYBRID_DQN_GUIDE.md** - Current implementation (active)
4. **DQN_RESEARCH_ANALYSIS.md** - Main research findings (keep per user request)
5. **IMPLEMENTATION_PLAN.md** - Implementation plan (keep per user request)
6. **PROJECT_HISTORY.md** - NEW: Track all experiments

### MOVE TO reports/ (Historical Documentation)
1. **14H_TRAINING_PLAN.md** - Old 75K episode training plan
2. **AGENT_INFORMATION_ANALYSIS.md** - Early analysis of agent info access
3. **CENTER_STACKING_FIXES.md** - Historical fix for center stacking
4. **CRITICAL_FIXES_APPLIED.md** - Dropout and train/eval mode fixes
5. **DEBUG_LOG.md** - Old debug log
6. **HOLE_MEASUREMENT_FIX.md** - Fix for measuring holes during play vs end
7. **TRAINING_ANALYSIS_10K.md** - Analysis of visual-only 10K run
8. **RECOMMENDATION.md** - Analysis that led to hybrid implementation
9. **FINAL_SUMMARY.md** - Phase 1 summary (feature channels)
10. **PROGRESS_SUMMARY.md** - Old progress tracking
11. **IMPLEMENTATION_COMPLETE.md** - Hybrid implementation completion summary
12. **dqn_architecture_analysis.md** - Early architecture analysis
13. **AGENTS.md** - Repository guidelines (move to reports)

## Actions
1. Create PROJECT_HISTORY.md summarizing all experiments
2. Move 13 files to reports/
3. Keep 6 essential files in root
4. Clean up old debug scripts
5. Update CLAUDE.md with new organization

## Result
Root directory: 6 essential MD files (down from 18)
Reports: 14 historical documents (organized)
