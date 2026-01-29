"""Retry failed players with better name matching"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import unicodedata
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguedashplayerstats
from period_boxscore_collector import (
    collect_player_period_stats, 
    save_player_period_stats,
    get_recent_game_ids,
    get_1q_boxscore,
    get_1h_boxscore,
    PERIOD_DATA_DIR
)
import os

def normalize_name(name):
    """Remove accents and normalize name"""
    # Remove accents
    normalized = unicodedata.normalize('NFKD', name)
    normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Lowercase for comparison
    return normalized.lower().strip()

def find_player_id_fuzzy(name):
    """Find player ID with fuzzy matching"""
    all_players = players.get_players()
    name_normalized = normalize_name(name)
    
    # Try exact match first
    for p in all_players:
        if normalize_name(p['full_name']) == name_normalized:
            return p['id'], p['full_name']
    
    # Try contains match
    for p in all_players:
        p_normalized = normalize_name(p['full_name'])
        if name_normalized in p_normalized or p_normalized in name_normalized:
            return p['id'], p['full_name']
    
    # Try last name match
    name_parts = name_normalized.split()
    if name_parts:
        last_name = name_parts[-1]
        for p in all_players:
            p_parts = normalize_name(p['full_name']).split()
            if p_parts and p_parts[-1] == last_name:
                # Check first initial too
                if len(name_parts) > 1 and len(p_parts) > 1:
                    if name_parts[0][0] == p_parts[0][0]:
                        return p['id'], p['full_name']
    
    return None, None

def collect_by_id(player_id, player_name, last_n_games=10):
    """Collect period stats using player ID directly"""
    game_ids = get_recent_game_ids(player_id=player_id, season='2025-26', last_n_games=last_n_games)
    
    if not game_ids:
        return None
    
    games_data = []
    
    for game_id in game_ids:
        game_data = {
            'game_id': game_id,
            'player_id': player_id,
            'player_name': player_name
        }
        
        # Get 1Q stats
        df_1q = get_1q_boxscore(game_id)
        if df_1q is not None and not df_1q.empty:
            player_row = df_1q[df_1q['PLAYER_ID'] == player_id]
            if not player_row.empty:
                row = player_row.iloc[0]
                game_data['1Q_PTS'] = row.get('PTS', 0)
                game_data['1Q_REB'] = row.get('REB', 0)
                game_data['1Q_AST'] = row.get('AST', 0)
        
        # Get 1H stats
        df_1h = get_1h_boxscore(game_id)
        if df_1h is not None and not df_1h.empty:
            player_row = df_1h[df_1h['PLAYER_ID'] == player_id]
            if not player_row.empty:
                row = player_row.iloc[0]
                game_data['1H_PTS'] = row.get('PTS', 0)
                game_data['1H_REB'] = row.get('REB', 0)
                game_data['1H_AST'] = row.get('AST', 0)
        
        if '1H_PTS' in game_data or '1Q_PTS' in game_data:
            games_data.append(game_data)
            safe_name = player_name.encode('ascii', 'replace').decode('ascii')
            print(f"  [OK] Game {game_id}: 1Q={game_data.get('1Q_PTS', 'N/A')} pts, 1H={game_data.get('1H_PTS', 'N/A')} pts")
        
        time.sleep(0.6)
    
    if not games_data:
        return None
    
    # Calculate averages
    stats = {
        'player_name': player_name,
        'player_id': player_id,
        'games_collected': len(games_data),
        'games_data': games_data
    }
    
    for period in ['1Q', '1H']:
        for stat in ['PTS', 'REB', 'AST']:
            key = f'{period}_{stat}'
            values = [g.get(key, 0) for g in games_data if key in g]
            if values:
                stats[f'{key}_avg'] = sum(values) / len(values)
    
    return stats

def main():
    print("=" * 60)
    print("RETRYING FAILED PLAYERS WITH FUZZY MATCHING")
    print("=" * 60)
    
    # Load failed players
    progress_file = os.path.join(PERIOD_DATA_DIR, '_collection_progress.json')
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    failed = list(set(progress.get('failed', [])))  # Remove duplicates
    print(f"\nFailed players to retry: {len(failed)}")
    
    # Get current season players with IDs
    print("\n[INFO] Fetching current player list...")
    time.sleep(2)
    
    collected = 0
    still_failed = []
    
    for i, name in enumerate(failed, 1):
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        print(f"\n[{i}/{len(failed)}] {safe_name}")
        
        # Try fuzzy match
        player_id, matched_name = find_player_id_fuzzy(name)
        
        if player_id:
            print(f"    Found: {matched_name} (ID: {player_id})")
            
            try:
                stats = collect_by_id(player_id, matched_name, last_n_games=10)
                
                if stats and stats.get('games_collected', 0) > 0:
                    save_player_period_stats(matched_name, stats)
                    print(f"    [OK] Collected {stats['games_collected']} games")
                    collected += 1
                else:
                    print(f"    [SKIP] No period data")
                    still_failed.append(name)
            except Exception as e:
                print(f"    [ERROR] {str(e)[:40]}")
                still_failed.append(name)
        else:
            print(f"    [SKIP] Could not find player")
            still_failed.append(name)
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("RETRY COMPLETE")
    print("=" * 60)
    print(f"Newly collected: {collected}")
    print(f"Still failed: {len(still_failed)}")
    
    if still_failed:
        print(f"\nStill missing ({len(still_failed)}):")
        for name in still_failed[:20]:
            safe = name.encode('ascii', 'replace').decode('ascii')
            print(f"  - {safe}")
        if len(still_failed) > 20:
            print(f"  ... and {len(still_failed) - 20} more")

if __name__ == "__main__":
    main()
