"""
Collect 1H/1Q Period Stats for ALL Active NBA Players (2025-26 Season)

This script builds a complete database of real 1st Half and 1st Quarter
statistics for every active player in the current season.

Due to API rate limits (~0.7s per call, 3 calls per game), this takes time:
- ~500 active players
- ~15 games each = 7,500 games
- ~3 API calls per game = 22,500 calls
- At 0.7s each = ~4-5 hours for full collection

Run this overnight or in batches.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict

# Force unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonallplayers, leaguedashplayerstats

from period_boxscore_collector import (
    collect_player_period_stats,
    save_player_period_stats,
    load_player_period_stats,
    PERIOD_DATA_DIR
)

# ============================================================================
# Configuration
# ============================================================================

CURRENT_SEASON = "2025-26"
GAMES_PER_PLAYER = 10  # Number of recent games to collect per player
PROGRESS_FILE = os.path.join(PERIOD_DATA_DIR, '_collection_progress.json')

# ============================================================================
# Get Active Players
# ============================================================================

def get_all_active_players() -> List[Dict]:
    """
    Get all active NBA players for the current season.
    Returns list of dicts with player_id, full_name, team.
    """
    print("[INFO] Fetching all active players...")
    time.sleep(0.7)
    
    try:
        all_players = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00",
            season=CURRENT_SEASON
        )
        df = all_players.get_data_frames()[0]
        
        # Filter to only active players (those with a team)
        active_df = df[df['TEAM_ID'] != 0]
        
        players_list = []
        for _, row in active_df.iterrows():
            players_list.append({
                'player_id': row['PERSON_ID'],
                'full_name': row['DISPLAY_FIRST_LAST'],
                'team_id': row['TEAM_ID'],
                'team_abbrev': row.get('TEAM_ABBREVIATION', 'UNK')
            })
        
        print(f"[INFO] Found {len(players_list)} active players")
        return players_list
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch players: {e}")
        return []


def get_players_with_minutes() -> List[Dict]:
    """
    Get players who have actually played minutes this season.
    More efficient than checking every rostered player.
    """
    print("[INFO] Fetching players with actual playing time...")
    
    # Retry logic for API timeouts
    df = None
    for attempt in range(3):
        try:
            time.sleep(2)  # Longer delay before API call
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=CURRENT_SEASON,
                season_type_all_star='Regular Season',
                per_mode_detailed='Totals',
                timeout=60  # Longer timeout
            )
            df = stats.get_data_frames()[0]
            print(f"[OK] Got player list")
            break
        except Exception as e:
            print(f"[RETRY {attempt+1}/3] API error: {str(e)[:50]}")
            if attempt == 2:
                print("[ERROR] Failed after 3 attempts")
                return []
            time.sleep(5)
    
    if df is None:
        return []
    
    try:
        # Filter to players with at least 5 games and some minutes
        active_df = df[(df['GP'] >= 5) & (df['MIN'] > 0)]
        
        players_list = []
        for _, row in active_df.iterrows():
            players_list.append({
                'player_id': row['PLAYER_ID'],
                'full_name': row['PLAYER_NAME'],
                'team_id': row['TEAM_ID'],
                'team_abbrev': row.get('TEAM_ABBREVIATION', 'UNK'),
                'games_played': row['GP'],
                'ppg': row['PTS'] / row['GP'] if row['GP'] > 0 else 0
            })
        
        # Sort by games played (most active first)
        players_list.sort(key=lambda x: x['games_played'], reverse=True)
        
        print(f"[INFO] Found {len(players_list)} players with playing time")
        return players_list
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch player stats: {e}")
        return []


# ============================================================================
# Progress Tracking
# ============================================================================

def load_progress() -> Dict:
    """Load collection progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed': [],
        'failed': [],
        'last_update': None
    }


def save_progress(progress: Dict):
    """Save collection progress to file"""
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Batch Collection
# ============================================================================

def collect_all_players_period_stats(
    max_players: int = None,
    resume: bool = True
):
    """
    Collect period stats for all active players.
    
    Args:
        max_players: Limit number of players (for testing). None = all players.
        resume: If True, skip already collected players.
    """
    # Get players with actual playing time
    players_list = get_players_with_minutes()
    
    if not players_list:
        print("[ERROR] No players found!")
        return
    
    if max_players:
        players_list = players_list[:max_players]
    
    # Load progress
    progress = load_progress() if resume else {'completed': [], 'failed': []}
    completed_names = set(progress.get('completed', []))
    
    print(f"\n{'='*60}")
    print(f"COLLECTING 1H/1Q STATS FOR {len(players_list)} PLAYERS")
    print(f"Already completed: {len(completed_names)}")
    print(f"Games per player: {GAMES_PER_PLAYER}")
    print(f"{'='*60}\n")
    
    # Estimate time
    remaining = len([p for p in players_list if p['full_name'] not in completed_names])
    est_minutes = (remaining * GAMES_PER_PLAYER * 3 * 0.7) / 60
    print(f"Estimated time for remaining {remaining} players: ~{est_minutes:.0f} minutes")
    print(f"{'='*60}\n")
    
    collected_count = 0
    failed_count = 0
    
    for i, player in enumerate(players_list):
        player_name = player['full_name']
        
        # Skip if already collected
        if player_name in completed_names:
            safe_name = player_name.encode('ascii', 'replace').decode('ascii')
            print(f"[{i+1}/{len(players_list)}] SKIP {safe_name} (already collected)")
            continue
        
        safe_name = player_name.encode('ascii', 'replace').decode('ascii')
        team = player.get('team_abbrev', '???')
        print(f"\n[{i+1}/{len(players_list)}] Collecting: {safe_name} ({team})")
        gp = player.get('games_played', 'N/A')
        ppg = player.get('ppg', 0)
        print(f"    GP: {gp}, PPG: {ppg:.1f}")
        
        try:
            stats = collect_player_period_stats(
                player_name,
                season=CURRENT_SEASON,
                last_n_games=GAMES_PER_PLAYER
            )
            
            if stats and stats.get('games_collected', 0) > 0:
                save_player_period_stats(player_name, stats)
                progress['completed'].append(player_name)
                collected_count += 1
                
                print(f"    [OK] Collected {stats['games_collected']} games")
                print(f"         1H PTS: {stats.get('1H_PTS_avg', 0):.1f}, 1Q PTS: {stats.get('1Q_PTS_avg', 0):.1f}")
            else:
                progress['failed'].append(player_name)
                failed_count += 1
                print(f"    [SKIP] No period data available")
                
        except Exception as e:
            progress['failed'].append(player_name)
            failed_count += 1
            print(f"    [ERROR] {str(e)[:50]}")
        
        # Save progress periodically
        if (i + 1) % 10 == 0:
            save_progress(progress)
            print(f"\n    --- Progress saved: {collected_count} collected, {failed_count} failed ---\n")
    
    # Final save
    save_progress(progress)
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total collected: {collected_count}")
    print(f"Total failed: {failed_count}")
    print(f"Data saved to: {PERIOD_DATA_DIR}")


def collect_top_players(n: int = 100):
    """Collect period stats for top N players by PPG (most relevant for props)"""
    players_list = get_players_with_minutes()
    
    # Sort by PPG
    players_list.sort(key=lambda x: x.get('ppg', 0), reverse=True)
    
    print(f"\nCollecting top {n} scorers...")
    collect_all_players_period_stats(max_players=n, resume=True)


# ============================================================================
# Quick Stats
# ============================================================================

def show_collection_stats():
    """Show stats about collected data"""
    if not os.path.exists(PERIOD_DATA_DIR):
        print("No data collected yet.")
        return
    
    files = [f for f in os.listdir(PERIOD_DATA_DIR) if f.endswith('_period_stats.json')]
    
    print(f"\n{'='*60}")
    print(f"COLLECTION STATISTICS")
    print(f"{'='*60}")
    print(f"Players collected: {len(files)}")
    print(f"Data directory: {PERIOD_DATA_DIR}")
    
    # Show sample
    if files:
        print(f"\nSample players collected:")
        for f in files[:10]:
            name = f.replace('_period_stats.json', '').replace('_', ' ')
            print(f"  - {name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    # Load progress
    progress = load_progress()
    print(f"\nProgress tracking:")
    print(f"  Completed: {len(progress.get('completed', []))}")
    print(f"  Failed: {len(progress.get('failed', []))}")
    print(f"  Last update: {progress.get('last_update', 'Never')}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect 1H/1Q stats for NBA players')
    parser.add_argument('--all', action='store_true', help='Collect all players')
    parser.add_argument('--top', type=int, default=None, help='Collect top N scorers')
    parser.add_argument('--test', type=int, default=None, help='Test with N players')
    parser.add_argument('--stats', action='store_true', help='Show collection stats')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from last progress')
    
    args = parser.parse_args()
    
    if args.stats:
        show_collection_stats()
    elif args.all:
        collect_all_players_period_stats(resume=args.resume)
    elif args.top:
        collect_top_players(n=args.top)
    elif args.test:
        print(f"Testing with {args.test} players...")
        collect_all_players_period_stats(max_players=args.test, resume=False)
    else:
        # Default: show stats and prompt
        show_collection_stats()
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print("  python collect_all_players.py --test 5    # Test with 5 players")
        print("  python collect_all_players.py --top 50    # Collect top 50 scorers")
        print("  python collect_all_players.py --all       # Collect ALL players")
        print("  python collect_all_players.py --stats     # Show collection stats")
