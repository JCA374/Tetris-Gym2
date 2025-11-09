"""
Ablation Study Runner

This script runs systematic ablation studies to determine which components
of our Tetris DQN contribute to performance.

Usage:
    python run_ablation_study.py --study architecture
    python run_ablation_study.py --study reward
    python run_ablation_study.py --study all --parallel 2
    python run_ablation_study.py --list
"""

import argparse
import subprocess
import time
import json
import os
from pathlib import Path
from ablation_configs import ALL_ABLATIONS, list_ablation_studies


def run_experiment(config, experiment_dir="ablation_results"):
    """
    Run a single experiment configuration.

    Args:
        config: Experiment configuration dictionary
        experiment_dir: Base directory for ablation results

    Returns:
        Exit code (0 for success)
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {config['name']}")
    print("=" * 80)
    print(f"Description: {config['description']}")
    print(f"Script: {config['script']}")

    # Build command
    cmd = ["python", config['script']]

    for arg, value in config['args'].items():
        if isinstance(value, bool):
            # Boolean flags (like --force_fresh)
            if value:
                # Flag is True: add it without a value
                cmd.append(arg)
            # If False: don't add the flag at all (skip it)
        elif isinstance(value, list):
            # List argument (like --hidden_dims 64 64)
            cmd.append(arg)
            cmd.extend(map(str, value))
        else:
            # Regular argument with value
            cmd.append(arg)
            cmd.append(str(value))

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run experiment
    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"✅ COMPLETED: {config['name']}")
        print(f"Time: {elapsed/3600:.2f} hours")
        print("=" * 80)

        return 0

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"❌ FAILED: {config['name']}")
        print(f"Exit code: {e.returncode}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("=" * 80)

        return e.returncode

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(f"⚠️  INTERRUPTED: {config['name']}")
        print("=" * 80)
        raise


def run_ablation_study(study_name, dry_run=False):
    """
    Run a complete ablation study.

    Args:
        study_name: Name of ablation study to run
        dry_run: If True, print commands without running

    Returns:
        Summary of results
    """
    if study_name not in ALL_ABLATIONS:
        available = ", ".join(ALL_ABLATIONS.keys())
        raise ValueError(f"Unknown study: {study_name}. Available: {available}")

    study = ALL_ABLATIONS[study_name]

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY: {study_name.upper()}")
    print("=" * 80)
    print(f"Description: {study['description']}")
    print(f"Experiments: {len(study['configurations'])}")
    print(f"Episodes per experiment: {study['base_episodes']}")
    print("=" * 80)

    if dry_run:
        print("\n🔍 DRY RUN - Commands will be printed but not executed")

    results = []
    start_time = time.time()

    for i, config in enumerate(study['configurations'], 1):
        print(f"\n\nExperiment {i}/{len(study['configurations'])}")

        if dry_run:
            print(f"\nWould run: {config['name']}")
            print(f"Script: {config['script']}")
            print(f"Args: {config['args']}")
            results.append({"name": config['name'], "status": "dry_run"})
        else:
            try:
                exit_code = run_experiment(config)
                results.append({
                    "name": config['name'],
                    "status": "success" if exit_code == 0 else "failed",
                    "exit_code": exit_code
                })
            except KeyboardInterrupt:
                print("\n\n⚠️  Ablation study interrupted by user")
                results.append({"name": config['name'], "status": "interrupted"})
                break

    total_time = time.time() - start_time

    # Print summary
    print("\n\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"Study: {study_name}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Experiments run: {len(results)}/{len(study['configurations'])}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    interrupted_count = sum(1 for r in results if r['status'] == 'interrupted')

    print(f"\nResults:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  ⚠️  Interrupted: {interrupted_count}")

    if not dry_run:
        print("\nTo compare results:")
        log_dirs = [f"logs/{cfg['args']['--experiment_name']}"
                   for cfg in study['configurations']]
        print(f"  python compare_models.py --log_dirs {' '.join(log_dirs)}")

    print("=" * 80)

    return results


def run_all_ablations(dry_run=False):
    """Run all ablation studies sequentially."""
    print("\n" + "=" * 80)
    print("RUNNING ALL ABLATION STUDIES")
    print("=" * 80)

    all_results = {}

    for study_name in ALL_ABLATIONS.keys():
        try:
            results = run_ablation_study(study_name, dry_run=dry_run)
            all_results[study_name] = results
        except KeyboardInterrupt:
            print("\n\n⚠️  All ablations interrupted by user")
            break

    # Print overall summary
    print("\n\n" + "=" * 80)
    print("ALL ABLATIONS SUMMARY")
    print("=" * 80)

    for study_name, results in all_results.items():
        success = sum(1 for r in results if r['status'] == 'success')
        total = len(results)
        print(f"{study_name:20s}: {success}/{total} completed")

    print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Tetris DQN ablation studies")

    parser.add_argument("--study", type=str,
                       choices=list(ALL_ABLATIONS.keys()) + ["all"],
                       help="Which ablation study to run")
    parser.add_argument("--list", action="store_true",
                       help="List all available ablation studies")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list:
        list_ablation_studies()
        return

    if args.study is None:
        print("\nError: Must specify --study or --list")
        print("\nAvailable studies:", ", ".join(ALL_ABLATIONS.keys()), "all")
        print("\nUse --list to see details of each study")
        return

    if args.study == "all":
        run_all_ablations(dry_run=args.dry_run)
    else:
        run_ablation_study(args.study, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
