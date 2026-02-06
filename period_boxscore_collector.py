"""
Period-Specific Box Score Collector

Fetches ACTUAL 1st Half and 1st Quarter box scores from NBA API.
This provides REAL data instead of estimated ratios.

Key endpoints:
- BoxScoreTraditionalV2 with start_period/end_period parameters

Period values:
- Q1 = 1
- Q2 = 2  
- Q3 = 3
- Q4 = 4
- 1st Half = start_period=1, end_period=2
- 1st Quarter = start_period=1, end_period=1
"""

import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from nba_api.stats.endpoints import (
    boxscoretraditionalv2,
    leaguegamefinder,
    playergamelog
)
from nba_api.stats.static import players, teams

# ============================================================================
# Configuration
# ============================================================================

PERIOD_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'period_stats')
os.makedirs(PERIOD_DATA_DIR, exist_ok=True)

# Rate limiting
API_DELAY = 0.7  # seconds between calls

# Current season
CURRENT_SEASON = "2025-26"

# ============================================================================
# Period Box Score Fetching
# ============================================================================

def get_boxscore_by_time_range(
    game_id: str, 
    end_range: int = 28800
) -> Optional[pd.DataFrame]:
    """
    Fetch box score for a specific time range of a game.
    
    IMPORTANT: Uses range_type=2 with time-based filtering.
    Range values are in tenths of a second:
    - 7200 = 1Q (12 minutes)
    - 14400 = 1H (24 minutes)  
    - 28800 = Full game (48 minutes)
    
    Args:
        game_id: NBA game ID (e.g., "0022400123")
        end_range: End time in tenths of seconds
    
    Returns:
        DataFrame with player stats for specified time range, or None if failed
    """
    try:
        time.sleep(API_DELAY)
        
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=game_id,
            start_period=0,
            end_period=10,
            start_range=0,
            end_range=end_range,
            range_type=2  # Time-based range filtering
        )
        
        # Get player stats dataframe (index 0)
        dfs = boxscore.get_data_frames()
        if dfs and len(dfs) > 0:
            player_stats = dfs[0]
            if not player_stats.empty:
                return player_stats
        
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch boxscore for game {game_id}: {e}")
        return None


# Time range constants (in tenths of seconds)
RANGE_1Q = 7200    # 12 minutes
RANGE_1H = 14400   # 24 minutes (1st Half)
RANGE_FULL = 28800 # 48 minutes (Full Game)


def get_1h_boxscore(game_id: str) -> Optional[pd.DataFrame]:
    """Get 1st Half box score (first 24 minutes)"""
    return get_boxscore_by_time_range(game_id, end_range=RANGE_1H)


def get_1q_boxscore(game_id: str) -> Optional[pd.DataFrame]:
    """Get 1st Quarter box score (first 12 minutes)"""
    return get_boxscore_by_time_range(game_id, end_range=RANGE_1Q)


def get_full_game_boxscore(game_id: str) -> Optional[pd.DataFrame]:
    """Get full game box score"""
    return get_boxscore_by_time_range(game_id, end_range=RANGE_FULL)


# ============================================================================
# Game ID Lookup
# ============================================================================

def get_recent_game_ids(
    team_abbrev: str = None,
    player_id: int = None,
    season: str = CURRENT_SEASON,
    last_n_games: int = 20
) -> List[str]:
    """
    Get recent game IDs for a team or player.
    
    Args:
        team_abbrev: Team abbreviation (e.g., "LAL")
        player_id: Player ID (alternative to team)
        season: Season string
        last_n_games: Number of recent games to fetch
    
    Returns:
        List of game IDs
    """
    try:
        time.sleep(API_DELAY)
        
        if player_id:
            # Get games from player's game log
            game_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = game_log.get_data_frames()[0]
            if not df.empty:
                return df['Game_ID'].head(last_n_games).tolist()
        
        elif team_abbrev:
            # Get games from league game finder
            team_id = get_team_id(team_abbrev)
            if team_id:
                game_finder = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=team_id,
                    season_nullable=season,
                    season_type_nullable='Regular Season'
                )
                df = game_finder.get_data_frames()[0]
                if not df.empty:
                    return df['GAME_ID'].head(last_n_games).tolist()
        
        return []
        
    except Exception as e:
        print(f"[ERROR] Failed to get game IDs: {e}")
        return []


@lru_cache(maxsize=50)
def get_team_id(team_abbrev: str) -> Optional[int]:
    """Get team ID from abbreviation"""
    team_list = teams.get_teams()
    for team in team_list:
        if team['abbreviation'] == team_abbrev:
            return team['id']
    return None


@lru_cache(maxsize=500)
def get_player_id(player_name: str) -> Optional[int]:
    """Get player ID from name"""
    player_list = players.get_players()
    for player in player_list:
        if player['full_name'].lower() == player_name.lower():
            return player['id']
    # Try partial match
    for player in player_list:
        if player_name.lower() in player['full_name'].lower():
            return player['id']
    return None


# ============================================================================
# Period Stats Collection & Storage
# ============================================================================

def collect_player_period_stats(
    player_name: str,
    season: str = CURRENT_SEASON,
    last_n_games: int = 20
) -> Dict:
    """
    Collect actual 1H and 1Q stats for a player from recent games.
    
    Returns:
        Dict with period-specific averages and game-by-game data
    """
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"[ERROR] Player not found: {player_name}")
        return {}
    
    # Get recent game IDs
    game_ids = get_recent_game_ids(player_id=player_id, season=season, last_n_games=last_n_games)
    
    if not game_ids:
        print(f"[WARNING] No games found for {player_name}")
        return {}
    
    print(f"[INFO] Collecting period stats for {player_name} ({len(game_ids)} games)...")
    
    # Collect stats for each game
    games_data = []
    
    for game_id in game_ids:
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
        
        # Get Full Game stats for comparison
        df_full = get_full_game_boxscore(game_id)
        if df_full is not None and not df_full.empty:
            player_row = df_full[df_full['PLAYER_ID'] == player_id]
            if not player_row.empty:
                row = player_row.iloc[0]
                game_data['FULL_PTS'] = row.get('PTS', 0)
                game_data['FULL_REB'] = row.get('REB', 0)
                game_data['FULL_AST'] = row.get('AST', 0)
                game_data['FULL_MIN'] = row.get('MIN', '0:00')
        
        # Only add if we got at least some data
        if any(k.startswith('1Q_') or k.startswith('1H_') for k in game_data.keys()):
            games_data.append(game_data)
            print(f"  [OK] Game {game_id}: 1Q={game_data.get('1Q_PTS', 'N/A')} pts, 1H={game_data.get('1H_PTS', 'N/A')} pts")
        else:
            print(f"  [SKIP] Game {game_id}: No period data available")
    
    if not games_data:
        return {}
    
    # Calculate averages
    df = pd.DataFrame(games_data)
    
    result = {
        'player_name': player_name,
        'player_id': player_id,
        'games_collected': len(games_data),
        'games_data': games_data,
        
        # 1Q Averages
        '1Q_PTS_avg': df['1Q_PTS'].mean() if '1Q_PTS' in df.columns else None,
        '1Q_REB_avg': df['1Q_REB'].mean() if '1Q_REB' in df.columns else None,
        '1Q_AST_avg': df['1Q_AST'].mean() if '1Q_AST' in df.columns else None,
        
        # 1H Averages
        '1H_PTS_avg': df['1H_PTS'].mean() if '1H_PTS' in df.columns else None,
        '1H_REB_avg': df['1H_REB'].mean() if '1H_REB' in df.columns else None,
        '1H_AST_avg': df['1H_AST'].mean() if '1H_AST' in df.columns else None,
        
        # Full Game Averages (for ratio calculation)
        'FULL_PTS_avg': df['FULL_PTS'].mean() if 'FULL_PTS' in df.columns else None,
        'FULL_REB_avg': df['FULL_REB'].mean() if 'FULL_REB' in df.columns else None,
        'FULL_AST_avg': df['FULL_AST'].mean() if 'FULL_AST' in df.columns else None,
    }
    
    # Calculate actual ratios - guard against None numerators and 0.0 falsy issue
    if result['FULL_PTS_avg'] is not None and result['FULL_PTS_avg'] > 0:
        result['1H_PTS_ratio'] = result['1H_PTS_avg'] / result['FULL_PTS_avg'] if result['1H_PTS_avg'] is not None else None
        result['1Q_PTS_ratio'] = result['1Q_PTS_avg'] / result['FULL_PTS_avg'] if result['1Q_PTS_avg'] is not None else None
    
    if result['FULL_REB_avg'] is not None and result['FULL_REB_avg'] > 0:
        result['1H_REB_ratio'] = result['1H_REB_avg'] / result['FULL_REB_avg'] if result['1H_REB_avg'] is not None else None
        result['1Q_REB_ratio'] = result['1Q_REB_avg'] / result['FULL_REB_avg'] if result['1Q_REB_avg'] is not None else None
    
    if result['FULL_AST_avg'] is not None and result['FULL_AST_avg'] > 0:
        result['1H_AST_ratio'] = result['1H_AST_avg'] / result['FULL_AST_avg'] if result['1H_AST_avg'] is not None else None
        result['1Q_AST_ratio'] = result['1Q_AST_avg'] / result['FULL_AST_avg'] if result['1Q_AST_avg'] is not None else None
    
    return result


def save_player_period_stats(player_name: str, stats: Dict):
    """Save player's period stats to file"""
    safe_name = player_name.replace(' ', '_').replace("'", "")
    filepath = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.json')
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"[INFO] Saved period stats to {filepath}")


def load_player_period_stats(player_name: str) -> Optional[Dict]:
    """Load player's period stats from file"""
    safe_name = player_name.replace(' ', '_').replace("'", "")
    filepath = os.path.join(PERIOD_DATA_DIR, f'{safe_name}_period_stats.json')
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


# ============================================================================
# Quick Access Functions
# ============================================================================

def get_player_1h_average(player_name: str, stat: str = 'PTS') -> Optional[float]:
    """
    Get player's actual 1st Half average for a stat.
    
    Args:
        player_name: Player's full name
        stat: 'PTS', 'REB', or 'AST'
    
    Returns:
        Average value or None if not available
    """
    stats = load_player_period_stats(player_name)
    if stats:
        return stats.get(f'1H_{stat}_avg')
    return None


def get_player_1q_average(player_name: str, stat: str = 'PTS') -> Optional[float]:
    """
    Get player's actual 1st Quarter average for a stat.
    """
    stats = load_player_period_stats(player_name)
    if stats:
        return stats.get(f'1Q_{stat}_avg')
    return None


def get_player_period_ratio(player_name: str, period: str = '1h', stat: str = 'PTS') -> Optional[float]:
    """
    Get player's actual ratio (1H or 1Q stats / Full game stats).
    
    This is the REAL ratio instead of the generic 0.48/0.24 estimates.
    """
    stats = load_player_period_stats(player_name)
    if stats:
        period_key = '1H' if period.lower() == '1h' else '1Q'
        return stats.get(f'{period_key}_{stat}_ratio')
    return None


# ============================================================================
# Batch Collection
# ============================================================================

def collect_multiple_players(
    player_names: List[str],
    season: str = CURRENT_SEASON,
    last_n_games: int = 15
):
    """
    Collect period stats for multiple players.
    Useful for building initial database.
    """
    print(f"\n{'='*60}")
    print(f"Collecting period stats for {len(player_names)} players")
    print(f"{'='*60}\n")
    
    results = {}
    
    for i, player_name in enumerate(player_names):
        print(f"\n[{i+1}/{len(player_names)}] {player_name}")
        print("-" * 40)
        
        try:
            stats = collect_player_period_stats(
                player_name, 
                season=season, 
                last_n_games=last_n_games
            )
            
            if stats:
                save_player_period_stats(player_name, stats)
                results[player_name] = stats
                
                # Print summary
                print(f"\n  Summary for {player_name}:")
                _1h_pts = stats.get('1H_PTS_avg')
                _1q_pts = stats.get('1Q_PTS_avg')
                _1h_ratio = stats.get('1H_PTS_ratio')
                _1q_ratio = stats.get('1Q_PTS_ratio')
                _1h_reb = stats.get('1H_REB_avg')
                _1q_reb = stats.get('1Q_REB_avg')
                print(f"  1H Points: {f'{_1h_pts:.1f}' if _1h_pts is not None else 'N/A'} avg (ratio: {f'{_1h_ratio:.2%}' if _1h_ratio is not None else 'N/A'})")
                print(f"  1Q Points: {f'{_1q_pts:.1f}' if _1q_pts is not None else 'N/A'} avg (ratio: {f'{_1q_ratio:.2%}' if _1q_ratio is not None else 'N/A'})")
                print(f"  1H Rebounds: {f'{_1h_reb:.1f}' if _1h_reb is not None else 'N/A'} avg")
                print(f"  1Q Rebounds: {f'{_1q_reb:.1f}' if _1q_reb is not None else 'N/A'} avg")
            else:
                print(f"  [WARNING] No data collected for {player_name}")
                
        except Exception as e:
            print(f"  [ERROR] Failed to collect stats for {player_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Collection complete! {len(results)}/{len(player_names)} players collected")
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Period Box Score Collector - Test Run")
    print("=" * 60)
    
    # Test with a single player
    test_player = "LeBron James"
    
    print(f"\nTesting with: {test_player}")
    print("-" * 40)
    
    # Collect stats
    stats = collect_player_period_stats(test_player, last_n_games=5)
    
    if stats:
        print(f"\n[SUCCESS] Collected data!")
        print(f"\nResults for {test_player}:")
        print(f"  Games collected: {stats['games_collected']}")
        print(f"\n  1st Half Averages:")
        if stats.get('1H_PTS_avg') is not None:
            print(f"    Points: {stats.get('1H_PTS_avg'):.1f}")
        if stats.get('1H_REB_avg') is not None:
            print(f"    Rebounds: {stats.get('1H_REB_avg'):.1f}")
        if stats.get('1H_AST_avg') is not None:
            print(f"    Assists: {stats.get('1H_AST_avg'):.1f}")
        print(f"\n  1st Quarter Averages:")
        if stats.get('1Q_PTS_avg') is not None:
            print(f"    Points: {stats.get('1Q_PTS_avg'):.1f}")
        if stats.get('1Q_REB_avg') is not None:
            print(f"    Rebounds: {stats.get('1Q_REB_avg'):.1f}")
        if stats.get('1Q_AST_avg') is not None:
            print(f"    Assists: {stats.get('1Q_AST_avg'):.1f}")
        print(f"\n  Actual Ratios (vs Full Game):")
        if stats.get('1H_PTS_ratio') is not None:
            print(f"    1H Points ratio: {stats.get('1H_PTS_ratio'):.2%}")
        if stats.get('1Q_PTS_ratio') is not None:
            print(f"    1Q Points ratio: {stats.get('1Q_PTS_ratio'):.2%}")
        
        # Save it
        save_player_period_stats(test_player, stats)
    else:
        print(f"\n[FAILED] Could not collect data for {test_player}")
