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
import shutil
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
    get_recent_game_ids,
    get_player_id,
    get_1h_boxscore,
    get_1q_boxscore,
    get_full_game_boxscore,
    PERIOD_DATA_DIR
)
import pandas as pd

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

_player_id_cache: Dict[str, int] = {}


def _normalize_player_name(name: str) -> str:
    """Normalize player names for matching (remove punctuation, lowercase)."""
    return ''.join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).strip()


def resolve_player_id(player_name: str) -> int:
    """
    Resolve player ID with robust name matching.
    Tries direct lookup, then normalized lookup to handle dots/hyphens.
    """
    if player_name in _player_id_cache:
        return _player_id_cache[player_name]

    # First try the standard lookup
    player_id = get_player_id(player_name)
    if player_id:
        _player_id_cache[player_name] = player_id
        return player_id

    # Fallback: normalized matching against static player list
    target = _normalize_player_name(player_name)
    for player in players.get_players():
        if _normalize_player_name(player['full_name']) == target:
            _player_id_cache[player_name] = player['id']
            return player['id']

    # Final fallback: contains match (normalized)
    for player in players.get_players():
        if target in _normalize_player_name(player['full_name']):
            _player_id_cache[player_name] = player['id']
            return player['id']

    _player_id_cache[player_name] = 0
    return 0


def backup_player_stats(player_name: str) -> bool:
    """Create a backup of player's stats before modifying. Returns True if backup created."""
    safe_name = player_name.replace(' ', '_').replace("'", "")
    filepath = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.json')
    backup_path = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.backup.json')
    
    if os.path.exists(filepath):
        try:
            shutil.copy2(filepath, backup_path)
            return True
        except Exception as e:
            print(f"[WARNING] Could not create backup for {player_name}: {e}")
            return False
    return False


def restore_player_stats(player_name: str) -> bool:
    """Restore player stats from backup. Returns True if restored."""
    safe_name = player_name.replace(' ', '_').replace("'", "")
    filepath = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.json')
    backup_path = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.backup.json')
    
    if os.path.exists(backup_path):
        try:
            shutil.copy2(backup_path, filepath)
            return True
        except Exception as e:
            print(f"[ERROR] Could not restore backup for {player_name}: {e}")
            return False
    return False


def cleanup_backup(player_name: str):
    """Remove backup file after successful update."""
    safe_name = player_name.replace(' ', '_').replace("'", "")
    backup_path = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.backup.json')
    
    if os.path.exists(backup_path):
        try:
            os.remove(backup_path)
        except:
            pass  # Not critical if cleanup fails


def collect_new_games_for_player(player_name: str, player_id: int, season: str = CURRENT_SEASON) -> Dict:
    """
    Collect ONLY new games for a player (games not already in their data).
    
    SAFETY FEATURES:
    - Creates backup before any modification
    - Validates new data before saving
    - Restores from backup if anything goes wrong
    - Never loses existing games
    
    Returns:
        Dict with update stats: {'new_games': N, 'total_games': M, 'updated': bool}
    """
    # Load existing data
    existing_stats = load_player_period_stats(player_name)
    
    if not existing_stats:
        # No existing data - this shouldn't happen in update mode
        return {'new_games': 0, 'total_games': 0, 'updated': False, 'reason': 'no_existing_data'}
    
    # Get existing game IDs - PRESERVE THESE
    existing_games = existing_stats.get('games_data', [])
    original_game_count = len(existing_games)
    existing_game_ids = set(g.get('game_id') for g in existing_games if g.get('game_id'))
    
    # Get current game IDs from API
    try:
        current_game_ids = get_recent_game_ids(player_id=player_id, season=season, last_n_games=30)
    except Exception as e:
        # API error - don't touch existing data
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': f'api_error: {str(e)[:30]}'}
    
    if not current_game_ids:
        # No games returned - don't touch existing data
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': 'api_returned_empty'}
    
    # Find NEW games (in current but not in existing)
    new_game_ids = [gid for gid in current_game_ids if gid not in existing_game_ids]
    
    if not new_game_ids:
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': 'no_new_games'}
    
    # CREATE BACKUP before making any changes
    backup_created = backup_player_stats(player_name)
    
    # Collect data for new games only
    new_games_data = []
    
    try:
        for game_id in new_game_ids:
            game_data = {
                'game_id': game_id,
                'player_id': player_id,
                'player_name': player_name
            }
            
            # Get 1st Quarter stats
            df_1q = get_1q_boxscore(game_id)
            if df_1q is not None and not df_1q.empty:
                player_row = df_1q[df_1q['PLAYER_ID'] == player_id]
                if not player_row.empty:
                    row = player_row.iloc[0]
                    game_data['1Q_PTS'] = row.get('PTS', 0)
                    game_data['1Q_REB'] = row.get('REB', 0)
                    game_data['1Q_AST'] = row.get('AST', 0)
                    game_data['1Q_MIN'] = row.get('MIN', '0:00')
            
            # Get 1st Half stats
            df_1h = get_1h_boxscore(game_id)
            if df_1h is not None and not df_1h.empty:
                player_row = df_1h[df_1h['PLAYER_ID'] == player_id]
                if not player_row.empty:
                    row = player_row.iloc[0]
                    game_data['1H_PTS'] = row.get('PTS', 0)
                    game_data['1H_REB'] = row.get('REB', 0)
                    game_data['1H_AST'] = row.get('AST', 0)
                    game_data['1H_MIN'] = row.get('MIN', '0:00')
            
            # Get Full Game stats
            df_full = get_full_game_boxscore(game_id)
            if df_full is not None and not df_full.empty:
                player_row = df_full[df_full['PLAYER_ID'] == player_id]
                if not player_row.empty:
                    row = player_row.iloc[0]
                    game_data['FULL_PTS'] = row.get('PTS', 0)
                    game_data['FULL_REB'] = row.get('REB', 0)
                    game_data['FULL_AST'] = row.get('AST', 0)
                    game_data['FULL_MIN'] = row.get('MIN', '0:00')
            
            # Only add if we got data
            if any(k.startswith('1Q_') or k.startswith('1H_') for k in game_data.keys()):
                new_games_data.append(game_data)
    
    except Exception as e:
        # Error during collection - restore backup and return
        if backup_created:
            restore_player_stats(player_name)
            cleanup_backup(player_name)
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': f'collection_error: {str(e)[:30]}'}
    
    if not new_games_data:
        # No new period data found - cleanup backup and return (no changes made)
        cleanup_backup(player_name)
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': 'no_period_data'}
    
    # Merge new games with existing (new games go at the front - most recent)
    all_games = new_games_data + existing_games
    
    # SAFETY CHECK: Verify we haven't lost any games
    if len(all_games) < original_game_count:
        # Something went wrong - restore backup
        if backup_created:
            restore_player_stats(player_name)
        cleanup_backup(player_name)
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': 'data_validation_failed'}
    
    # Recalculate averages and save
    try:
        df = pd.DataFrame(all_games)
        
        # Ensure numeric columns are properly converted
        numeric_cols = ['1Q_PTS', '1Q_REB', '1Q_AST', '1H_PTS', '1H_REB', '1H_AST', 
                        'FULL_PTS', 'FULL_REB', 'FULL_AST']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        updated_stats = {
            'player_name': player_name,
            'player_id': player_id,
            'games_collected': len(all_games),
            'games_data': all_games,
            'last_updated': datetime.now().isoformat(),
            
            # 1Q Averages
            '1Q_PTS_avg': float(df['1Q_PTS'].mean()) if '1Q_PTS' in df.columns and df['1Q_PTS'].notna().any() else None,
            '1Q_REB_avg': float(df['1Q_REB'].mean()) if '1Q_REB' in df.columns and df['1Q_REB'].notna().any() else None,
            '1Q_AST_avg': float(df['1Q_AST'].mean()) if '1Q_AST' in df.columns and df['1Q_AST'].notna().any() else None,
            
            # 1H Averages  
            '1H_PTS_avg': float(df['1H_PTS'].mean()) if '1H_PTS' in df.columns and df['1H_PTS'].notna().any() else None,
            '1H_REB_avg': float(df['1H_REB'].mean()) if '1H_REB' in df.columns and df['1H_REB'].notna().any() else None,
            '1H_AST_avg': float(df['1H_AST'].mean()) if '1H_AST' in df.columns and df['1H_AST'].notna().any() else None,
            
            # Full Game Averages
            'FULL_PTS_avg': float(df['FULL_PTS'].mean()) if 'FULL_PTS' in df.columns and df['FULL_PTS'].notna().any() else None,
            'FULL_REB_avg': float(df['FULL_REB'].mean()) if 'FULL_REB' in df.columns and df['FULL_REB'].notna().any() else None,
            'FULL_AST_avg': float(df['FULL_AST'].mean()) if 'FULL_AST' in df.columns and df['FULL_AST'].notna().any() else None,
        }
        
        # Calculate ratios
        if updated_stats['FULL_PTS_avg'] and updated_stats['FULL_PTS_avg'] > 0:
            updated_stats['1H_PTS_ratio'] = updated_stats['1H_PTS_avg'] / updated_stats['FULL_PTS_avg']
            updated_stats['1Q_PTS_ratio'] = updated_stats['1Q_PTS_avg'] / updated_stats['FULL_PTS_avg'] if updated_stats['1Q_PTS_avg'] else None
        
        if updated_stats['FULL_REB_avg'] and updated_stats['FULL_REB_avg'] > 0:
            updated_stats['1H_REB_ratio'] = updated_stats['1H_REB_avg'] / updated_stats['FULL_REB_avg']
            updated_stats['1Q_REB_ratio'] = updated_stats['1Q_REB_avg'] / updated_stats['FULL_REB_avg'] if updated_stats['1Q_REB_avg'] else None
        
        if updated_stats['FULL_AST_avg'] and updated_stats['FULL_AST_avg'] > 0:
            updated_stats['1H_AST_ratio'] = updated_stats['1H_AST_avg'] / updated_stats['FULL_AST_avg']
            updated_stats['1Q_AST_ratio'] = updated_stats['1Q_AST_avg'] / updated_stats['FULL_AST_avg'] if updated_stats['1Q_AST_avg'] else None
        
        # FINAL SAFETY CHECK before saving
        if updated_stats['games_collected'] < original_game_count:
            raise ValueError(f"Game count dropped from {original_game_count} to {updated_stats['games_collected']}")
        
        # Save updated stats
        save_player_period_stats(player_name, updated_stats)
        
        # Success - cleanup backup
        cleanup_backup(player_name)
        
    except Exception as e:
        # Error during save - restore backup
        if backup_created:
            restore_player_stats(player_name)
        cleanup_backup(player_name)
        return {'new_games': 0, 'total_games': original_game_count, 'updated': False, 'reason': f'save_error: {str(e)[:30]}'}
    
    return {
        'new_games': len(new_games_data),
        'total_games': len(all_games),
        'updated': True,
        'reason': 'success'
    }


def smart_update_all_players():
    """
    Smart incremental update - only fetches NEW games for players who have played.
    
    This is MUCH faster than full collection because:
    1. Only checks players who already have data
    2. Only downloads games not already in the database
    3. Skips players with no new games
    """
    print(f"\n{'='*70}")
    print("SMART INCREMENTAL UPDATE")
    print("Only fetching NEW games since last update")
    print(f"{'='*70}\n")
    
    # Get all players with existing data
    if not os.path.exists(PERIOD_DATA_DIR):
        print("[ERROR] No period stats directory found. Run full collection first.")
        return
    
    player_files = [f for f in os.listdir(PERIOD_DATA_DIR) 
                    if f.endswith('_period_stats.json') and not f.startswith('_')]
    
    if not player_files:
        print("[ERROR] No player data found. Run full collection first.")
        return
    
    print(f"Found {len(player_files)} players with existing data")
    
    # Stats tracking
    stats = {
        'players_checked': 0,
        'players_updated': 0,
        'players_skipped': 0,
        'new_games_total': 0,
        'errors': 0
    }
    
    # Process each player
    for i, filename in enumerate(player_files):
        player_name = filename.replace('_period_stats.json', '').replace('_', ' ')
        safe_name = player_name.encode('ascii', 'replace').decode('ascii')
        
        stats['players_checked'] += 1
        
        # Get player ID
        # Prefer stored player_id from existing data to avoid name mismatches
        existing_stats = load_player_period_stats(player_name)
        player_id = existing_stats.get('player_id') if existing_stats else 0
        if not player_id:
            player_id = resolve_player_id(player_name)
        if not player_id:
            print(f"[{i+1}/{len(player_files)}] {safe_name}: Could not find player ID")
            stats['errors'] += 1
            continue
        
        print(f"[{i+1}/{len(player_files)}] {safe_name}...", end=" ", flush=True)
        
        try:
            result = collect_new_games_for_player(player_name, player_id)
            
            if result['updated']:
                stats['players_updated'] += 1
                stats['new_games_total'] += result['new_games']
                print(f"+{result['new_games']} new games (total: {result['total_games']})")
            else:
                stats['players_skipped'] += 1
                reason = result.get('reason', 'unknown')
                if reason == 'no_new_games':
                    print("up to date")
                else:
                    print(f"skipped ({reason})")
                    
        except Exception as e:
            stats['errors'] += 1
            print(f"ERROR: {str(e)[:40]}")
        
        # Progress save every 50 players
        if (i + 1) % 50 == 0:
            print(f"\n--- Progress: {stats['players_updated']} updated, {stats['new_games_total']} new games ---\n")
    
    # Summary
    print(f"\n{'='*70}")
    print("UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"Players checked:  {stats['players_checked']}")
    print(f"Players updated:  {stats['players_updated']}")
    print(f"Players skipped:  {stats['players_skipped']} (already up to date)")
    print(f"New games added:  {stats['new_games_total']}")
    print(f"Errors:           {stats['errors']}")
    print()
    
    return stats


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
    parser.add_argument('--all', action='store_true', help='Collect all players (full collection)')
    parser.add_argument('--update', action='store_true', help='Smart update - only fetch NEW games (fast!)')
    parser.add_argument('--top', type=int, default=None, help='Collect top N scorers')
    parser.add_argument('--test', type=int, default=None, help='Test with N players')
    parser.add_argument('--stats', action='store_true', help='Show collection stats')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from last progress')
    
    args = parser.parse_args()
    
    if args.stats:
        show_collection_stats()
    elif args.update:
        # Smart incremental update - only new games!
        smart_update_all_players()
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
        print("\n" + "="*70)
        print("USAGE:")
        print("="*70)
        print("  python collect_all_players.py --update    # FAST: Only fetch new games")
        print("  python collect_all_players.py --test 5    # Test with 5 players")
        print("  python collect_all_players.py --top 50    # Collect top 50 scorers")
        print("  python collect_all_players.py --all       # Collect ALL players (slow)")
        print("  python collect_all_players.py --stats     # Show collection stats")
        print()
        print("RECOMMENDED: Use --update for daily refreshes (only downloads new games)")