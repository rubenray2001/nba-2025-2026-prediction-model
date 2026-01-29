"""
Collect ALL players using player IDs directly from current season stats.
This bypasses name matching issues with special characters.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import time
import json
from datetime import datetime
from nba_api.stats.endpoints import leaguedashplayerstats
from period_boxscore_collector import (
    get_recent_game_ids,
    get_1q_boxscore,
    get_1h_boxscore,
    PERIOD_DATA_DIR
)

PROGRESS_FILE = os.path.join(PERIOD_DATA_DIR, '_collection_progress_v2.json')

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed_ids': [], 'failed_ids': []}

def save_progress(progress):
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def save_player_stats(player_name, stats):
    """Save with safe filename"""
    # Remove special chars for filename
    safe_name = ''.join(c if c.isalnum() or c == ' ' else '' for c in player_name)
    safe_name = safe_name.replace(' ', '_')
    filepath = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.json')
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    return filepath

def collect_player_by_id(player_id, player_name, last_n_games=10):
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
        
        has_data = False
        
        # Get 1Q stats
        df_1q = get_1q_boxscore(game_id)
        if df_1q is not None and not df_1q.empty:
            player_row = df_1q[df_1q['PLAYER_ID'] == player_id]
            if not player_row.empty:
                row = player_row.iloc[0]
                game_data['1Q_PTS'] = int(row.get('PTS', 0) or 0)
                game_data['1Q_REB'] = int(row.get('REB', 0) or 0)
                game_data['1Q_AST'] = int(row.get('AST', 0) or 0)
                game_data['1Q_MIN'] = str(row.get('MIN', '0'))
                has_data = True
        
        # Get 1H stats
        df_1h = get_1h_boxscore(game_id)
        if df_1h is not None and not df_1h.empty:
            player_row = df_1h[df_1h['PLAYER_ID'] == player_id]
            if not player_row.empty:
                row = player_row.iloc[0]
                game_data['1H_PTS'] = int(row.get('PTS', 0) or 0)
                game_data['1H_REB'] = int(row.get('REB', 0) or 0)
                game_data['1H_AST'] = int(row.get('AST', 0) or 0)
                game_data['1H_MIN'] = str(row.get('MIN', '0'))
                has_data = True
        
        if has_data:
            games_data.append(game_data)
            print(f"  [OK] Game {game_id}: 1Q={game_data.get('1Q_PTS', 'N/A')} pts, 1H={game_data.get('1H_PTS', 'N/A')} pts")
        else:
            print(f"  [SKIP] Game {game_id}: No period data")
        
        time.sleep(0.6)
    
    if not games_data:
        return None
    
    # Calculate averages and ratios
    stats = {
        'player_name': player_name,
        'player_id': player_id,
        'games_collected': len(games_data),
        'games_data': games_data
    }
    
    # Calculate averages
    for period in ['1Q', '1H']:
        for stat in ['PTS', 'REB', 'AST']:
            key = f'{period}_{stat}'
            values = [g.get(key, 0) for g in games_data if key in g]
            if values:
                stats[f'{key}_avg'] = round(sum(values) / len(values), 2)
    
    # Calculate ratios (need full game stats)
    # For now, just save the period averages
    
    return stats

def main():
    print("=" * 70)
    print("COLLECTING ALL PLAYERS BY ID (BYPASSES NAME MATCHING)")
    print("=" * 70)
    
    # Get ALL players from current season with their IDs
    print("\n[INFO] Fetching all players from 2025-26 season...")
    time.sleep(2)
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals',
            timeout=120
        )
        df = stats.get_data_frames()[0]
        print(f"[OK] Got {len(df)} players from API")
    except Exception as e:
        print(f"[ERROR] Failed to get players: {e}")
        return
    
    # Filter to players with games
    df = df[df['GP'] >= 1].copy()
    df = df.sort_values('GP', ascending=False)
    
    players_list = []
    for _, row in df.iterrows():
        players_list.append({
            'player_id': row['PLAYER_ID'],
            'player_name': row['PLAYER_NAME'],
            'team': row.get('TEAM_ABBREVIATION', '???'),
            'gp': row['GP'],
            'ppg': row['PTS'] / row['GP'] if row['GP'] > 0 else 0
        })
    
    print(f"[INFO] {len(players_list)} players to process")
    
    # Load progress
    progress = load_progress()
    completed_ids = set(progress.get('completed_ids', []))
    
    print(f"[INFO] Already completed: {len(completed_ids)}")
    
    # Count how many need collection
    need_collection = [p for p in players_list if p['player_id'] not in completed_ids]
    print(f"[INFO] Need to collect: {len(need_collection)}")
    
    if not need_collection:
        print("\n[OK] All players already collected!")
        return
    
    est_minutes = (len(need_collection) * 10 * 2 * 0.6) / 60
    print(f"[INFO] Estimated time: ~{est_minutes:.0f} minutes")
    print("=" * 70)
    
    collected = 0
    failed = 0
    
    for i, player in enumerate(players_list, 1):
        player_id = player['player_id']
        player_name = player['player_name']
        
        if player_id in completed_ids:
            continue
        
        safe_name = player_name.encode('ascii', 'replace').decode('ascii')
        print(f"\n[{i}/{len(players_list)}] {safe_name} ({player['team']}) - GP: {player['gp']}")
        
        try:
            stats = collect_player_by_id(player_id, player_name, last_n_games=10)
            
            if stats and stats.get('games_collected', 0) > 0:
                filepath = save_player_stats(player_name, stats)
                progress['completed_ids'].append(player_id)
                collected += 1
                print(f"    [OK] Saved {stats['games_collected']} games")
            else:
                progress['failed_ids'].append(player_id)
                failed += 1
                print(f"    [SKIP] No period data available")
        
        except Exception as e:
            progress['failed_ids'].append(player_id)
            failed += 1
            print(f"    [ERROR] {str(e)[:50]}")
        
        # Save progress every 10 players
        if i % 10 == 0:
            save_progress(progress)
            print(f"\n--- Progress saved: {collected} collected, {failed} failed ---")
    
    save_progress(progress)
    
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Newly collected: {collected}")
    print(f"Failed (no period data): {failed}")
    print(f"Total with data: {len([f for f in os.listdir(PERIOD_DATA_DIR) if f.endswith('_period_stats.json') and not f.startswith('_')])}")

if __name__ == "__main__":
    main()
