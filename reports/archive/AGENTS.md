# Repository Guidelines

## Project Structure & Module Organization
- Core training code lives in `src/` (`agent.py`, `model.py`, `reward_shaping.py`, `utils.py`). Wrap new logic in this package rather than the top-level scripts when possible.
- Entry points sit in the repo root (`train.py`, `train_progressive_improved.py`, `evaluate.py`, `monitor_training.py`). Keep these thin—delegate heavy lifting to `src/`.
- Configuration defaults are centralized in `config.py`; use it instead of scattering constants. Generated artifacts land in `models/` and `logs/`.
- Diagnostic and regression scripts reside under `tests/`. Each file is standalone; most inject the project root onto `sys.path` so they can be called via `python tests/<file>.py`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: create and activate an isolated environment.
- `pip install -r requirements.txt`: install runtime and training dependencies.
- `python train.py --episodes 500`: run the canonical training loop; accepts the CLI flags defined in `parse_args()`.
- `python train_progressive_improved.py --episodes 75000 --resume`: launch the staged curriculum run recommended for long training.
- `python evaluate.py --model_path models/best_model.pth --render`: evaluate a saved model with optional rendering.
- `python tests/test_actions_simple.py` (or any script in `tests/`): execute targeted diagnostics.

## Coding Style & Naming Conventions
- Python code follows PEP 8: four-space indentation, snake_case functions, CamelCase classes. Keep modules under 500 lines and prefer helper functions when a block exceeds ~40 lines.
- Use descriptive experiment and log names (`logs/improved_<timestamp>`). When adding configs, mirror existing constant names to align with `argparse` flags.
- Employ concise docstrings for new public functions; include inline comments only for non-obvious logic.

## Testing Guidelines
- Prefer extending an existing `tests/test_*.py` script before writing a new one; each test should include a short docstring explaining its intent.
- Name new diagnostics `test_<behavior>.py` (e.g., `test_reward_regression.py`). Keep assertions explicit and deterministic—seed RNG via `numpy.random.seed` when randomness is unavoidable.
- Run relevant scripts locally before submitting. For broad changes, execute a smoke sweep: `python tests/test_actions_simple.py`, `python tests/test_reward_helpers.py`, and a representative environment check.

## Commit & Pull Request Guidelines
- Write imperative, present-tense commit messages (`Fix reward clamp bug`, `Add curriculum stage metrics`). Group related edits; avoid multi-purpose commits.
- Open PRs with a brief summary, reproduction/testing notes, and links to any generated logs or checkpoints (e.g., `logs/improved_YYYYMMDD_HHMM`). Attach plots when behavior changes.
- Highlight risky areas (reward shaping, agent epsilon logic) and call out follow-up work so reviewers can plan next steps.
