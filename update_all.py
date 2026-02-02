"""
NBA Model Daily Update Script
Runs all necessary updates to keep the model current.

Usage:
    python update_all.py           # Update data only (fast, ~5-10 min)
    python update_all.py --retrain # Update data + regenerate training + retrain (30-60 min)

The --retrain flag:
- Collects fresh training data from ALL player logs
- Retrains the model with updated data
- This is required for the model to learn from new games
"""

import subprocess
import sys
import time
from datetime import datetime


def run_command(description: str, command: list) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(command)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            cwd=sys.path[0] or '.',
            check=False
        )
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        if result.returncode == 0:
            print("-" * 60)
            print(f"  [OK] Completed in {minutes}m {seconds}s")
            return True
        else:
            print("-" * 60)
            print(f"  [ERROR] Failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           NBA MODEL UPDATE SCRIPT                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Parse arguments
    retrain = "--retrain" in sys.argv or "--full" in sys.argv
    
    if retrain:
        print("  Mode: UPDATE + REGENERATE TRAINING DATA + RETRAIN")
    else:
        print("  Mode: DATA UPDATE ONLY (fast)")
    
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    steps_completed = 0
    steps_failed = 0
    
    # Step 1: Update period stats (1Q/1H data)
    if run_command(
        "Step 1/3: Updating period stats (1Q/1H player data)...",
        [sys.executable, "collect_all_players.py", "--update"]
    ):
        steps_completed += 1
    else:
        steps_failed += 1
        print("  Warning: Period stats update had issues, continuing...")
    
    # Step 2: Update player game logs + team stats (all players)
    if run_command(
        "Step 2/3: Updating player game logs & team stats...",
        [sys.executable, "train_model.py", "--update"]
    ):
        steps_completed += 1
    else:
        steps_failed += 1
        print("  Warning: Game logs update had issues, continuing...")
    
    # Step 3: Retrain model (optional)
    if retrain:
        if run_command(
            "Step 3/3: Regenerating training data + Retraining model (ALL players)...",
            [sys.executable, "train_model.py", "--collect", "--train"]
        ):
            steps_completed += 1
        else:
            steps_failed += 1
    else:
        print(f"\n{'='*60}")
        print("  Step 3/3: Skipping model retrain (use --retrain to include)")
        print(f"{'='*60}")
    
    # Summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    UPDATE COMPLETE                       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Steps completed: {steps_completed:<3}                                     ║
    ║  Steps failed:    {steps_failed:<3}                                     ║
    ║  Total time:      {minutes}m {seconds}s{' '*(32-len(f'{minutes}m {seconds}s'))}║
    ╠══════════════════════════════════════════════════════════╣
    ║  Your model is now up to date!                           ║
    ║  Run predictions with: python app.py                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    return 0 if steps_failed == 0 else 1


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    sys.exit(main())
