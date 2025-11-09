"""
Master Test Runner - Run all debugging tests in sequence
=========================================================

This script runs all debugging tests and provides a comprehensive
diagnosis of what's preventing the agent from clearing lines.
"""

import subprocess
import sys
from pathlib import Path

# Test descriptions
TESTS = [
    {
        'script': 'test_1_basic_env.py',
        'name': 'Basic Environment Test',
        'description': 'Verify that line clearing is possible with random actions',
        'critical': True
    },
    {
        'script': 'test_2_wrapper.py',
        'name': 'Environment Wrapper Test',
        'description': 'Verify that the wrapper preserves info dict correctly',
        'critical': True
    },
    {
        'script': 'test_3_features.py',
        'name': 'Feature Extraction Test',
        'description': 'Verify that feature extraction works correctly',
        'critical': False
    },
    {
        'script': 'test_4_reward.py',
        'name': 'Reward Function Test',
        'description': 'Verify that reward function receives and processes line clears',
        'critical': True
    },
    {
        'script': 'test_5_actions.py',
        'name': 'Action Distribution Test',
        'description': 'Analyze which actions the trained agent uses',
        'critical': True
    },
]


def print_header(text, char='='):
    """Print a formatted header"""
    line = char * 70
    print(f"\n{line}")
    print(text)
    print(line)


def run_test(test_info):
    """Run a single test script"""
    script_path = Path(__file__).parent / test_info['script']

    print_header(f"🧪 {test_info['name']}", '=')
    print(f"Description: {test_info['description']}")
    print(f"Critical: {'Yes' if test_info['critical'] else 'No'}")
    print()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            print(f"\n✅ {test_info['name']} completed successfully")
            return 'pass'
        else:
            print(f"\n⚠️  {test_info['name']} completed with errors")
            return 'warning'

    except Exception as e:
        print(f"\n❌ {test_info['name']} failed to run: {e}")
        return 'fail'


def main():
    """Run all tests and provide summary"""
    print_header("🔬 COMPREHENSIVE DEBUGGING TEST SUITE", '=')
    print("This will run a series of tests to diagnose why the agent")
    print("has not cleared any lines after 18,500 episodes.")
    print("\nPress Enter to begin...")
    input()

    results = {}

    for test in TESTS:
        result = run_test(test)
        results[test['script']] = result

        print("\n" + "─" * 70)
        print("Press Enter to continue to next test...")
        input()

    # Print summary
    print_header("📊 TEST SUMMARY", '=')

    passed = sum(1 for r in results.values() if r == 'pass')
    warned = sum(1 for r in results.values() if r == 'warning')
    failed = sum(1 for r in results.values() if r == 'fail')

    print(f"\nResults:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ⚠️  Warnings: {warned}")
    print(f"  ❌ Failed: {failed}")

    print("\nDetailed results:")
    for test in TESTS:
        result = results[test['script']]
        symbol = {'pass': '✅', 'warning': '⚠️ ', 'fail': '❌'}[result]
        critical = '(CRITICAL)' if test['critical'] else ''
        print(f"  {symbol} {test['name']} {critical}")

    # Recommendations
    print_header("💡 RECOMMENDATIONS", '=')

    critical_failures = [
        test for test in TESTS
        if test['critical'] and results[test['script']] != 'pass'
    ]

    if critical_failures:
        print("\n⚠️  Critical issues found:")
        for test in critical_failures:
            print(f"  - {test['name']}")
        print("\nYou should fix these issues before continuing training.")
    else:
        print("\n✅ All critical tests passed!")
        print("\nPotential issues to investigate:")
        print("  1. Check test_5 results - is the agent using HARD_DROP?")
        print("  2. Review Q-values - are they meaningful?")
        print("  3. Consider increasing exploration (epsilon)")
        print("  4. Try curriculum learning or shaped rewards")

    print_header("📝 NEXT STEPS", '=')
    print("\n1. Review the test outputs above to identify the root cause")
    print("2. Focus on any failed critical tests first")
    print("3. Check action distribution - if agent doesn't use HARD_DROP,")
    print("   it won't clear lines efficiently")
    print("4. Consider these solutions:")
    print("   - Reset epsilon to encourage more exploration")
    print("   - Add shaped rewards (reward for filling rows)")
    print("   - Use curriculum learning (start with easier scenarios)")
    print("   - Add expert demonstrations to replay buffer")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
