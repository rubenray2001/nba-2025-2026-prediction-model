"""
Quick Refresh Script - Updates recent game data for all players
Run this weekly to keep predictions fresh with latest game stats.

Usage:
    python quick_refresh.py           # Default: last 3 games
    python quick_refresh.py --games 5 # Custom: last 5 games
"""

import sys
import json
import time
import argparse
from datetime import datetime

# Unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)

from nba_api.stats.endpoints import leaguedashplayerstats
from period_boxscore_collector import collect_player_period_stats, load_player_period_stats, save_player_period_stats

# Files
PERIOD_STATS_FILE = '_all_players_period_stats.json'
REFRESH_LOG_FILE = '_refresh_log.json'


def get_active_players():
    """Get all players who have played this season."""
    print("\n[INFO] Fetching active players from NBA API...")
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        df = stats.get_data_frames()[0]
        
        # Filter to players with at least 1 game played
        active = df[df['GP'] >= 1][['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP']].copy()
        active = active.sort_values('GP', ascending=False)
        
        print(f"[INFO] Found {len(active)} active players")
        return active.to_dict('records')
        
    except Exception as e:
        print(f"[ERROR] Failed to get active players: {e}")
        return []


def load_refresh_log():
    """Load the refresh log to track last update times."""
    try:
        with open(REFRESH_LOG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


def save_refresh_log(log):
    """Save the refresh log."""
    with open(REFRESH_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def quick_refresh(last_n_games=3, force=False):
    """
    Quickly refresh player stats with only recent games.
    
    Args:
        last_n_games: Number of recent games to collect (default 3)
        force: If True, refresh all players. If False, skip recently updated.
    """
    print("=" * 70)
    print("QUICK REFRESH - Updating Recent Game Data")
    print("=" * 70)
    print(f"Games to collect per player: {last_n_games}")
    print(f"Force refresh all: {force}")
    print()
    
    # Get active players
    players = get_active_players()
    if not players:
        print("[ERROR] No players found. Exiting.")
        return
    
    # Load existing data
    existing_stats = load_player_period_stats()
    refresh_log = load_refresh_log()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Track progress
    updated = 0
    skipped = 0
    failed = 0
    
    total = len(players)
    
    print(f"\n[INFO] Processing {total} players...")
    print("-" * 70)
    
    for i, player in enumerate(players):
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER_NAME']
        team = player['TEAM_ABBREVIATION']
        
        # Safe name for printing
        safe_name = player_name.encode('ascii', 'replace').decode('ascii')
        
        # Check if recently updated (within last 2 days)
        last_refresh = refresh_log.get(str(player_id), {}).get('date', '')
        if not force and last_refresh == today:
            skipped += 1
            continue
        
        print(f"[{i+1}/{total}] {safe_name} ({team})")
        
        try:
            # Collect recent games only
            stats = collect_player_period_stats(
                player_id=player_id,
                player_name=player_name,
                season='2025-26',
                last_n_games=last_n_games
            )
            
            if stats and stats.get('games_collected', 0) > 0:
                # Merge with existing data (keep more games if we have them)
                if player_name in existing_stats:
                    old_stats = existing_stats[player_name]
                    # Update averages with fresh data
                    existing_stats[player_name] = stats
                else:
                    existing_stats[player_name] = stats
                
                # Update refresh log
                refresh_log[str(player_id)] = {
                    'name': player_name,
                    'date': today,
                    'games': stats.get('games_collected', 0)
                }
                
                updated += 1
                print(f"    [OK] Updated with {stats.get('games_collected', 0)} games")
            else:
                failed += 1
                print(f"    [SKIP] No recent games")
                
        except Exception as e:
            failed += 1
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"    [ERROR] {error_msg}")
        
        # Rate limiting
        time.sleep(0.6)
        
        # Save periodically
        if (i + 1) % 50 == 0:
            save_player_period_stats(existing_stats)
            save_refresh_log(refresh_log)
            print(f"\n[CHECKPOINT] Saved progress at {i+1}/{total}\n")
    
    # Final save
    save_player_period_stats(existing_stats)
    save_refresh_log(refresh_log)
    
    # Summary
    print("\n" + "=" * 70)
    print("REFRESH COMPLETE")
    print("=" * 70)
    print(f"Updated: {updated} players")
    print(f"Skipped (already fresh): {skipped} players")
    print(f"Failed/No data: {failed} players")
    print(f"Total in database: {len(existing_stats)} players")
    print()


def main():
    parser = argparse.ArgumentParser(description='Quick refresh of player period stats')
    parser.add_argument('--games', type=int, default=3, help='Number of recent games to collect (default: 3)')
    parser.add_argument('--force', action='store_true', help='Force refresh all players, even if recently updated')
    
    args = parser.parse_args()
    
    quick_refresh(last_n_games=args.games, force=args.force)


if __name__ == '__main__':
    main()
