"""
Ablation Study Configurations

This module defines configurations for systematic ablation studies to determine
which components of our Tetris DQN contribute to performance.

Ablation studies test:
1. Model architecture (simple features vs. CNN vs. hybrid)
2. Reward function (simple vs. curriculum)
3. Hyperparameters (learning rate, gamma, epsilon schedule)
4. Model components (dropout, dueling architecture, etc.)

Run ablations with:
    python run_ablation_study.py --study architecture
    python run_ablation_study.py --study reward
    python run_ablation_study.py --study all
"""

# ============================================================================
# ARCHITECTURE ABLATION STUDY
# ============================================================================

ARCHITECTURE_ABLATION = {
    "name": "architecture_comparison",
    "description": "Compare different model architectures",
    "base_episodes": 5000,

    "configurations": [
        {
            "name": "simple_feature_4d",
            "description": "Simple 4-feature feedforward (baseline from literature)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "basic",  # 4 features
                "--model_type": "simple_dqn",
                "--hidden_dims": [64, 64],
                "--reward_variant": "quadratic",
                "--lr": 0.001,
                "--experiment_name": "abl_arch_simple4d",
            }
        },
        {
            "name": "simple_feature_6d",
            "description": "Simple 6-feature feedforward",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "standard",  # 6 features
                "--model_type": "simple_dqn",
                "--hidden_dims": [64, 64],
                "--reward_variant": "quadratic",
                "--experiment_name": "abl_arch_simple6d",
            }
        },
        {
            "name": "simple_feature_8d",
            "description": "Simple 8-feature feedforward",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "extended",  # 8 features
                "--model_type": "simple_dqn",
                "--hidden_dims": [64, 64],
                "--reward_variant": "quadratic",
                "--experiment_name": "abl_arch_simple8d",
            }
        },
        {
            "name": "standard_cnn_8ch",
            "description": "Standard CNN with 8-channel input",
            "script": "train_progressive_improved.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "dqn",
                "--force_fresh": True,
                "--experiment_name": "abl_arch_cnn8ch",
            }
        },
        {
            "name": "hybrid_dual_branch",
            "description": "Hybrid dual-branch CNN (our innovation)",
            "script": "train_progressive_improved.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "hybrid_dqn",
                "--force_fresh": True,
                "--experiment_name": "abl_arch_hybrid",
            }
        },
    ]
}

# ============================================================================
# REWARD FUNCTION ABLATION STUDY
# ============================================================================

REWARD_ABLATION = {
    "name": "reward_comparison",
    "description": "Compare different reward functions",
    "base_episodes": 5000,

    "configurations": [
        {
            "name": "reward_quadratic",
            "description": "Simple quadratic reward (baseline from literature)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "basic",
                "--reward_variant": "quadratic",  # (lines²)×10
                "--experiment_name": "abl_reward_quadratic",
            }
        },
        {
            "name": "reward_exponential",
            "description": "Exponential reward (2^lines)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "basic",
                "--reward_variant": "exponential",  # 10×2^lines
                "--experiment_name": "abl_reward_exponential",
            }
        },
        {
            "name": "reward_sparse",
            "description": "Truly sparse (no survival bonus)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "basic",
                "--reward_variant": "sparse",
                "--experiment_name": "abl_reward_sparse",
            }
        },
        {
            "name": "reward_light_penalty",
            "description": "Quadratic + light structure penalties",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--feature_set": "basic",
                "--reward_variant": "light_penalty",
                "--experiment_name": "abl_reward_light_penalty",
            }
        },
        {
            "name": "reward_progressive",
            "description": "5-stage progressive curriculum (our approach)",
            "script": "train_progressive_improved.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "hybrid_dqn",
                "--force_fresh": True,
                "--experiment_name": "abl_reward_progressive",
            }
        },
    ]
}

# ============================================================================
# HYPERPARAMETER ABLATION STUDY
# ============================================================================

HYPERPARAMETER_ABLATION = {
    "name": "hyperparameter_tuning",
    "description": "Test different hyperparameter settings",
    "base_episodes": 5000,

    "configurations": [
        {
            "name": "lr_0001",
            "description": "Learning rate 0.0001 (conservative)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--lr": 0.0001,
                "--experiment_name": "abl_hyper_lr0001",
            }
        },
        {
            "name": "lr_001",
            "description": "Learning rate 0.001 (baseline)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--lr": 0.001,
                "--experiment_name": "abl_hyper_lr001",
            }
        },
        {
            "name": "lr_005",
            "description": "Learning rate 0.005 (aggressive)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--lr": 0.005,
                "--experiment_name": "abl_hyper_lr005",
            }
        },
        {
            "name": "gamma_095",
            "description": "Gamma 0.95 (from literature)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--gamma": 0.95,
                "--experiment_name": "abl_hyper_gamma095",
            }
        },
        {
            "name": "gamma_099",
            "description": "Gamma 0.99 (our default)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--gamma": 0.99,
                "--experiment_name": "abl_hyper_gamma099",
            }
        },
    ]
}

# ============================================================================
# MODEL COMPONENT ABLATION STUDY
# ============================================================================

COMPONENT_ABLATION = {
    "name": "model_components",
    "description": "Test which model components help",
    "base_episodes": 5000,

    "configurations": [
        {
            "name": "simple_standard",
            "description": "Simple DQN (baseline)",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "simple_dqn",
                "--experiment_name": "abl_comp_simple_dqn",
            }
        },
        {
            "name": "simple_dueling",
            "description": "Simple Dueling DQN",
            "script": "train_baseline_simple.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "simple_dueling_dqn",
                "--experiment_name": "abl_comp_simple_dueling",
            }
        },
        {
            "name": "cnn_standard",
            "description": "Standard CNN DQN",
            "script": "train_progressive_improved.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "dqn",
                "--force_fresh": True,
                "--experiment_name": "abl_comp_cnn_dqn",
            }
        },
        {
            "name": "cnn_dueling",
            "description": "CNN Dueling DQN",
            "script": "train_progressive_improved.py",
            "args": {
                "--episodes": 5000,
                "--model_type": "dueling_dqn",
                "--force_fresh": True,
                "--experiment_name": "abl_comp_cnn_dueling",
            }
        },
    ]
}

# ============================================================================
# ALL ABLATION STUDIES
# ============================================================================

ALL_ABLATIONS = {
    "architecture": ARCHITECTURE_ABLATION,
    "reward": REWARD_ABLATION,
    "hyperparameter": HYPERPARAMETER_ABLATION,
    "component": COMPONENT_ABLATION,
}


def get_ablation_study(study_name):
    """
    Get ablation study configuration by name.

    Args:
        study_name: Name of ablation study

    Returns:
        Configuration dictionary
    """
    if study_name not in ALL_ABLATIONS:
        available = ", ".join(ALL_ABLATIONS.keys())
        raise ValueError(f"Unknown ablation study: {study_name}. Available: {available}")

    return ALL_ABLATIONS[study_name]


def list_ablation_studies():
    """List all available ablation studies."""
    print("\nAvailable Ablation Studies:")
    print("=" * 80)

    for name, config in ALL_ABLATIONS.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Episodes: {config['base_episodes']}")
        print(f"  Configurations: {len(config['configurations'])}")

        for i, exp in enumerate(config['configurations'], 1):
            print(f"    {i}. {exp['name']}: {exp['description']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    list_ablation_studies()
