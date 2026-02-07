"""
NBA Model Daily Update Script
Runs all necessary updates to keep the model current.

Usage:
    python update_all.py           # Update data only (fast, ~5-10 min)
    python update_all.py --retrain # Update data + retrain + auto-push to GitHub (30-60 min)
    python update_all.py --no-push # Update + retrain but skip git push

The --retrain flag:
- Collects fresh training data from ALL player logs
- Retrains the model with updated data
- Auto-pushes ALL files (code + models + data) to GitHub
- This ensures Streamlit Cloud always has the latest model
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
    auto_push = "--no-push" not in sys.argv  # Push by default
    
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
    
    # Step 4: Auto-push to GitHub (ensures Streamlit Cloud gets new models)
    if auto_push:
        print(f"\n{'='*60}")
        print("  Step 4: Pushing ALL changes to GitHub...")
        print(f"{'='*60}")
        
        try:
            # Stage everything: code, models, data, period stats
            subprocess.run(["git", "add", "-A"], cwd=sys.path[0] or '.', check=True)
            
            # Check if there's anything to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=sys.path[0] or '.', capture_output=True, text=True
            )
            
            if status.stdout.strip():
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                msg = f"Auto-update {timestamp}: data"
                if retrain:
                    msg += " + retrained models"
                
                subprocess.run(
                    ["git", "commit", "-m", msg], 
                    cwd=sys.path[0] or '.', check=True
                )
                subprocess.run(
                    ["git", "push"], 
                    cwd=sys.path[0] or '.', check=True
                )
                print("  [OK] Pushed to GitHub - Streamlit Cloud will auto-deploy")
                steps_completed += 1
            else:
                print("  [OK] No changes to push")
                steps_completed += 1
        except Exception as e:
            print(f"  [ERROR] Git push failed: {e}")
            print("  WARNING: Changes are NOT live on Streamlit Cloud!")
            print("  Run manually: git add -A && git commit -m 'update' && git push")
            steps_failed += 1
    else:
        print(f"\n{'='*60}")
        print("  Step 4: Skipping git push (--no-push flag)")
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
    ║  Model is up to date + pushed to GitHub!                 ║
    ║  Streamlit Cloud will auto-deploy in ~2 min              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    return 0 if steps_failed == 0 else 1


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    sys.exit(main())
