"""
Cleanup duplicate player period stats files.
Keeps the file with more games, deletes the other.
"""

import os
import json
from typing import Dict, List, Tuple

PERIOD_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'period_stats')

# Known duplicate pairs (normalized_name: [variant1, variant2])
DUPLICATE_PAIRS = [
    ("A.J._Lawson", "AJ_Lawson"),
    ("Andre_Jackson_Jr.", "Andre_Jackson_Jr"),
    ("Derrick_Jones_Jr.", "Derrick_Jones_Jr"),
    ("Dorian_Finney-Smith", "Dorian_FinneySmith"),
    ("Gary_Trent_Jr.", "Gary_Trent_Jr"),
    ("Jabari_Smith_Jr.", "Jabari_Smith_Jr"),
    ("Jaime_Jaquez_Jr.", "Jaime_Jaquez_Jr"),
    ("Jaren_Jackson_Jr.", "Jaren_Jackson_Jr"),
    ("Jeremiah_Robinson-Earl", "Jeremiah_RobinsonEarl"),
    ("Karl-Anthony_Towns", "KarlAnthony_Towns"),
    ("Kelly_Oubre_Jr.", "Kelly_Oubre_Jr"),
    ("Kentavious_Caldwell-Pope", "Kentavious_CaldwellPope"),
    ("Kevin_Porter_Jr.", "Kevin_Porter_Jr"),
    ("Larry_Nance_Jr.", "Larry_Nance_Jr"),
    ("Michael_Porter_Jr.", "Michael_Porter_Jr"),
    ("Nick_Smith_Jr.", "Nick_Smith_Jr"),
    ("Nickeil_Alexander-Walker", "Nickeil_AlexanderWalker"),
    ("Olivier-Maxence_Prosper", "OlivierMaxence_Prosper"),
    ("P.J._Washington", "PJ_Washington"),
    ("Ron_Harper_Jr.", "Ron_Harper_Jr"),
    ("Shai_Gilgeous-Alexander", "Shai_GilgeousAlexander"),
    ("T.J._McConnell", "TJ_McConnell"),
    ("Tim_Hardaway_Jr.", "Tim_Hardaway_Jr"),
    ("Trayce_Jackson-Davis", "Trayce_JacksonDavis"),
    ("Vince_Williams_Jr.", "Vince_Williams_Jr"),
    ("Wendell_Carter_Jr.", "Wendell_Carter_Jr"),
]


def get_games_count(filename: str) -> int:
    """Get the number of games in a player's stats file."""
    filepath = os.path.join(PERIOD_DATA_DIR, filename)
    if not os.path.exists(filepath):
        return -1
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('games_collected', 0)
    except:
        return 0


def cleanup_duplicates(dry_run: bool = True):
    """
    Clean up duplicate player files, keeping the one with more data.
    
    Args:
        dry_run: If True, only show what would be deleted. If False, actually delete.
    """
    print("=" * 60)
    print("DUPLICATE FILE CLEANUP")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will delete files)'}")
    print()
    
    deleted = 0
    kept = 0
    
    for name1, name2 in DUPLICATE_PAIRS:
        file1 = f"{name1}_period_stats.json"
        file2 = f"{name2}_period_stats.json"
        
        path1 = os.path.join(PERIOD_DATA_DIR, file1)
        path2 = os.path.join(PERIOD_DATA_DIR, file2)
        
        exists1 = os.path.exists(path1)
        exists2 = os.path.exists(path2)
        
        if not exists1 and not exists2:
            continue
        
        if exists1 and not exists2:
            print(f"[OK] {name1}: Only one file exists (keeping)")
            kept += 1
            continue
            
        if exists2 and not exists1:
            print(f"[OK] {name2}: Only one file exists (keeping)")
            kept += 1
            continue
        
        # Both exist - compare games count
        games1 = get_games_count(file1)
        games2 = get_games_count(file2)
        
        if games1 >= games2:
            keep_file, delete_file = file1, file2
            keep_games, delete_games = games1, games2
        else:
            keep_file, delete_file = file2, file1
            keep_games, delete_games = games2, games1
        
        print(f"[DUPLICATE] {name1.replace('_', ' ')}")
        print(f"   Keep:   {keep_file} ({keep_games} games)")
        print(f"   Delete: {delete_file} ({delete_games} games)")
        
        if not dry_run:
            try:
                os.remove(os.path.join(PERIOD_DATA_DIR, delete_file))
                print(f"   [DELETED]")
                deleted += 1
            except Exception as e:
                print(f"   [ERROR] Could not delete: {e}")
        else:
            deleted += 1
        
        kept += 1
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files to keep: {kept}")
    print(f"Files to delete: {deleted}")
    
    if dry_run:
        print()
        print("This was a DRY RUN. To actually delete, run:")
        print("  python cleanup_duplicates.py --delete")


if __name__ == "__main__":
    import sys
    
    dry_run = "--delete" not in sys.argv
    cleanup_duplicates(dry_run=dry_run)
