"""
NBA Data Collection Module using nba_api
Fetches player game logs, team stats, and defensive ratings

Performance Optimized:
- In-memory LRU caching for frequent lookups
- Reduced API calls with smarter caching
- Lazy loading and batch operations
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from functools import lru_cache
import hashlib

from nba_api.stats.endpoints import (
    playergamelog,
    commonallplayers,
    leaguedashteamstats,
    playercareerstats,
    commonteamroster
)
from nba_api.stats.static import players, teams

from config import CURRENT_SEASON, SEASONS_TO_FETCH, DATA_DIR, CACHE_DIR, TEAM_ABBREVIATIONS

# ============================================================================
# In-Memory Caching Layer
# ============================================================================

# Cache for player lookups (very frequent)
_player_cache: Dict[str, Optional[Dict]] = {}
_player_list_cache: Optional[List[Dict]] = None
_player_list_cache_time: Optional[datetime] = None

# Cache for team data (rarely changes)
_team_cache: Dict[str, Dict] = {}
_all_teams_cache: Optional[List[Dict]] = None

# Cache for defensive stats
_defensive_stats_cache: Dict[str, pd.DataFrame] = {}
_pace_stats_cache: Dict[str, pd.DataFrame] = {}


def ensure_dirs():
    """Create necessary directories - cached check"""
    if not hasattr(ensure_dirs, '_initialized'):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "player_logs"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "team_stats"), exist_ok=True)
        ensure_dirs._initialized = True


def _get_active_players_list() -> List[Dict]:
    """Get cached list of active players - uses live NBA data for completeness"""
    global _player_list_cache, _player_list_cache_time
    
    # Check if cache is valid (refresh every 24 hours)
    if _player_list_cache is not None and _player_list_cache_time is not None:
        if datetime.now() - _player_list_cache_time < timedelta(hours=24):
            return _player_list_cache
    
    # Use get_all_active_players() which fetches from live NBA API
    # This includes rookies and recent additions that static list misses
    try:
        df = get_all_active_players()
        if not df.empty:
            _player_list_cache = [
                {
                    'id': int(row['PERSON_ID']),
                    'full_name': row['DISPLAY_FIRST_LAST'],
                    'first_name': row.get('DISPLAY_FIRST_LAST', '').split()[0] if row.get('DISPLAY_FIRST_LAST') else '',
                    'last_name': ' '.join(row.get('DISPLAY_FIRST_LAST', '').split()[1:]) if row.get('DISPLAY_FIRST_LAST') else '',
                    'is_active': True
                }
                for _, row in df.iterrows()
            ]
            _player_list_cache_time = datetime.now()
            return _player_list_cache
    except Exception as e:
        print(f"Error loading players from live API: {e}")
    
    # Fallback to static list if live API fails
    _player_list_cache = players.get_active_players()
    _player_list_cache_time = datetime.now()
    return _player_list_cache


def get_all_active_players() -> pd.DataFrame:
    """Get all active NBA players for current season"""
    ensure_dirs()
    cache_file = os.path.join(CACHE_DIR, "active_players.csv")
    
    # Check file cache (valid for 24 hours)
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=24):
            return pd.read_csv(cache_file)
    
    try:
        all_players = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            season=CURRENT_SEASON
        )
        time.sleep(0.6)  # Rate limiting
        
        df = all_players.get_data_frames()[0]
        df = df[df['ROSTERSTATUS'] == 1]  # Only rostered players
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"Error fetching active players: {e}")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        return pd.DataFrame()


def search_player(player_name: str) -> Optional[Dict]:
    """
    Search for a player by name (fuzzy match)
    Performance: Uses in-memory cache for repeated lookups
    """
    # Normalize name for cache key
    cache_key = player_name.lower().strip()
    
    # Check in-memory cache first
    if cache_key in _player_cache:
        return _player_cache[cache_key]
    
    # Get players list (cached)
    all_players = _get_active_players_list()
    
    # Try exact match first
    for player in all_players:
        full_name = player['full_name'].lower()
        if full_name == cache_key:
            _player_cache[cache_key] = player
            return player
    
    # Try partial match
    matches = []
    for player in all_players:
        full_name = player['full_name'].lower()
        if cache_key in full_name or full_name in cache_key:
            matches.append(player)
    
    # Try last name match
    if not matches:
        for player in all_players:
            last_name = player['last_name'].lower()
            if cache_key == last_name or cache_key in last_name:
                matches.append(player)
    
    result = matches[0] if matches else None
    _player_cache[cache_key] = result  # Cache even None results
    return result


def get_player_game_logs(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Fetch player game logs for a season
    Performance: File-based caching with 6-hour TTL
    """
    ensure_dirs()
    cache_file = os.path.join(DATA_DIR, "player_logs", f"{player_id}_{season.replace('-', '_')}.csv")
    
    # Check file cache
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=6):
            return pd.read_csv(cache_file)
    
    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        time.sleep(0.6)  # Rate limiting
        
        df = game_log.get_data_frames()[0]
        if not df.empty:
            df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"Error fetching game logs for player {player_id}: {e}")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        return pd.DataFrame()


def get_player_multi_season_logs(player_id: int, seasons: List[str] = None) -> pd.DataFrame:
    """
    Fetch player game logs across multiple seasons
    Performance: Parallel-ready structure, uses cached single-season calls
    """
    if seasons is None:
        seasons = SEASONS_TO_FETCH
    
    all_logs = []
    for season in seasons:
        logs = get_player_game_logs(player_id, season)
        if not logs.empty:
            # Use assign instead of direct assignment to avoid copy warning
            logs = logs.assign(SEASON=season)
            all_logs.append(logs)
        time.sleep(0.3)  # Additional rate limiting
    
    if all_logs:
        return pd.concat(all_logs, ignore_index=True)
    return pd.DataFrame()


def get_team_defensive_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Get team defensive statistics (opponent stats)
    Performance: In-memory + file caching
    """
    # Check in-memory cache first
    if season in _defensive_stats_cache:
        return _defensive_stats_cache[season]
    
    ensure_dirs()
    cache_file = os.path.join(DATA_DIR, "team_stats", f"defensive_{season.replace('-', '_')}.csv")
    
    # Check file cache
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=12):
            df = pd.read_csv(cache_file)
            _defensive_stats_cache[season] = df
            return df
    
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Opponent'
        )
        time.sleep(0.6)
        
        df = team_stats.get_data_frames()[0]
        if not df.empty:
            df.to_csv(cache_file, index=False)
            _defensive_stats_cache[season] = df
        return df
    except Exception as e:
        print(f"Error fetching team defensive stats: {e}")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            _defensive_stats_cache[season] = df
            return df
        return pd.DataFrame()


def get_team_pace_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Get team pace and advanced stats
    Performance: In-memory + file caching
    """
    # Check in-memory cache first
    if season in _pace_stats_cache:
        return _pace_stats_cache[season]
    
    ensure_dirs()
    cache_file = os.path.join(DATA_DIR, "team_stats", f"pace_{season.replace('-', '_')}.csv")
    
    # Check file cache
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=12):
            df = pd.read_csv(cache_file)
            _pace_stats_cache[season] = df
            return df
    
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Advanced'
        )
        time.sleep(0.6)
        
        df = team_stats.get_data_frames()[0]
        if not df.empty:
            df.to_csv(cache_file, index=False)
            _pace_stats_cache[season] = df
        return df
    except Exception as e:
        print(f"Error fetching team pace stats: {e}")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            _pace_stats_cache[season] = df
            return df
        return pd.DataFrame()


def get_all_teams() -> List[Dict]:
    """
    Get all NBA teams
    Performance: Cached static data
    """
    global _all_teams_cache
    if _all_teams_cache is None:
        _all_teams_cache = teams.get_teams()
    return _all_teams_cache


def get_team_by_abbreviation(abbrev: str) -> Optional[Dict]:
    """
    Get team info by abbreviation
    Performance: In-memory caching
    """
    abbrev = abbrev.upper()
    
    # Check cache
    if abbrev in _team_cache:
        return _team_cache[abbrev]
    
    all_teams = get_all_teams()
    for team in all_teams:
        if team['abbreviation'] == abbrev:
            _team_cache[abbrev] = team
            return team
    
    _team_cache[abbrev] = None  # Cache miss
    return None


def get_team_roster(team_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Get team roster for a season"""
    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        time.sleep(0.6)
        return roster.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return pd.DataFrame()


def get_matchup_history(player_id: int, opponent_team_id: int, num_games: int = 10) -> pd.DataFrame:
    """
    Get player's history against specific opponent
    Performance: Reuses cached multi-season logs
    """
    # Get all game logs (uses cache)
    all_logs = get_player_multi_season_logs(player_id)
    
    if all_logs.empty:
        return pd.DataFrame()
    
    # Get team abbreviation (cached)
    team_info = None
    for team in get_all_teams():
        if team['id'] == opponent_team_id:
            team_info = team
            break
    
    if not team_info:
        return pd.DataFrame()
    
    abbrev = team_info['abbreviation']
    
    # Filter for games against opponent
    matchup_games = all_logs[all_logs['MATCHUP'].str.contains(abbrev, na=False)]
    
    return matchup_games.head(num_games)


def collect_training_data(player_ids: List[int] = None, seasons: List[str] = None) -> pd.DataFrame:
    """
    Collect comprehensive training data for multiple players.
    Performance: Batch processing with progress saves
    """
    ensure_dirs()
    
    if seasons is None:
        seasons = SEASONS_TO_FETCH
    
    if player_ids is None:
        active_players = get_all_active_players()
        if active_players.empty:
            return pd.DataFrame()
        player_ids = active_players['PERSON_ID'].tolist()[:100]
    
    all_data = []
    total = len(player_ids)
    
    for i, player_id in enumerate(player_ids):
        print(f"Collecting data for player {i+1}/{total} (ID: {player_id})")
        
        logs = get_player_multi_season_logs(player_id, seasons)
        if not logs.empty:
            logs = logs.assign(PLAYER_ID=player_id)
            all_data.append(logs)
        
        # Progress save every 50 players
        if (i + 1) % 50 == 0 and all_data:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df.to_csv(os.path.join(DATA_DIR, f"training_data_checkpoint_{i+1}.csv"), index=False)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(os.path.join(DATA_DIR, "training_data_full.csv"), index=False)
        return final_df
    
    return pd.DataFrame()


def get_player_current_team(player_id: int) -> Optional[str]:
    """Get player's current team abbreviation"""
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        time.sleep(0.6)
        df = career.get_data_frames()[0]
        if not df.empty:
            # Get most recent season
            latest = df[df['SEASON_ID'] == df['SEASON_ID'].max()]
            if not latest.empty:
                return latest.iloc[0]['TEAM_ABBREVIATION']
    except Exception as e:
        print(f"Error getting current team: {e}")
    return None


def clear_memory_cache():
    """Clear all in-memory caches (useful for testing)"""
    global _player_cache, _player_list_cache, _player_list_cache_time
    global _team_cache, _all_teams_cache
    global _defensive_stats_cache, _pace_stats_cache
    
    _player_cache = {}
    _player_list_cache = None
    _player_list_cache_time = None
    _team_cache = {}
    _all_teams_cache = None
    _defensive_stats_cache = {}
    _pace_stats_cache = {}


def get_player_names_for_autocomplete() -> List[str]:
    """
    Get list of all active player names for autocomplete.
    Performance: Uses cached player list
    """
    all_players = _get_active_players_list()
    return sorted([p['full_name'] for p in all_players])


def search_players_fuzzy(query: str, limit: int = 10) -> List[Dict]:
    """
    Fuzzy search for players by name.
    Returns list of matching players for autocomplete.
    """
    if not query or len(query) < 2:
        return []
    
    query = query.lower().strip()
    all_players = _get_active_players_list()
    
    matches = []
    
    # Exact start match (highest priority)
    for player in all_players:
        full_name = player['full_name'].lower()
        if full_name.startswith(query):
            matches.append((0, player))
    
    # Contains match
    for player in all_players:
        full_name = player['full_name'].lower()
        if query in full_name and not full_name.startswith(query):
            matches.append((1, player))
    
    # Last name match
    for player in all_players:
        last_name = player['last_name'].lower()
        if query in last_name and query not in player['full_name'].lower():
            matches.append((2, player))
    
    # Sort by priority and return
    matches.sort(key=lambda x: x[0])
    return [m[1] for m in matches[:limit]]


def get_todays_games() -> List[Dict]:
    """
    Get today's NBA games.
    Returns list of game dictionaries with teams.
    Uses multiple methods for reliability.
    """
    from datetime import date, datetime, timedelta
    
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    
    # Method 1: Try ScoreboardV2
    try:
        from nba_api.stats.endpoints import scoreboardv2
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today_str)
        time.sleep(0.6)
        
        # ScoreboardV2 returns multiple dataframes, game header is usually index 0 or 1
        dfs = scoreboard.get_data_frames()
        games_df = None
        
        # Find the dataframe with game data
        for df in dfs:
            if not df.empty and 'GAME_ID' in df.columns:
                games_df = df
                break
        
        if games_df is not None and not games_df.empty:
            games = _parse_games_dataframe(games_df)
            if games:
                return games
    except Exception:
        pass  # ScoreboardV2 not available or no games
    
    # Method 2: Try LeagueGameFinder
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        time.sleep(0.6)
        
        # Search for games on today's date
        game_finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=today_str,
            date_to_nullable=today_str,
            league_id_nullable='00'  # NBA
        )
        time.sleep(0.6)
        
        games_df = game_finder.get_data_frames()[0]
        
        if not games_df.empty:
            # LeagueGameFinder returns one row per team, so group by game
            games = _parse_game_finder_results(games_df)
            if games:
                return games
    except Exception:
        pass  # LeagueGameFinder not available or no games
    
    # Method 3: Try Scoreboard (live)
    try:
        from nba_api.live.nba.endpoints import scoreboard
        time.sleep(0.6)
        
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            live_games = games_data['scoreboard']['games']
            if live_games:
                games = []
                for g in live_games:
                    game = {
                        'game_id': g.get('gameId'),
                        'home_team': g.get('homeTeam', {}).get('teamTricode', ''),
                        'home_team_name': g.get('homeTeam', {}).get('teamName', ''),
                        'home_team_id': g.get('homeTeam', {}).get('teamId'),
                        'away_team': g.get('awayTeam', {}).get('teamTricode', ''),
                        'away_team_name': g.get('awayTeam', {}).get('teamName', ''),
                        'away_team_id': g.get('awayTeam', {}).get('teamId'),
                        'game_status': g.get('gameStatusText', ''),
                    }
                    games.append(game)
                return games
    except Exception:
        pass  # Live scoreboard not available or no games
    
    return []


def _parse_games_dataframe(games_df: pd.DataFrame) -> List[Dict]:
    """Parse games from ScoreboardV2 dataframe"""
    games = []
    all_teams = get_all_teams()
    team_lookup = {t['id']: t for t in all_teams}
    
    for _, row in games_df.iterrows():
        home_id = row.get('HOME_TEAM_ID')
        away_id = row.get('VISITOR_TEAM_ID')
        
        game = {
            'game_id': row.get('GAME_ID'),
            'home_team_id': home_id,
            'away_team_id': away_id,
            'game_status': row.get('GAME_STATUS_TEXT', ''),
        }
        
        # Lookup team info
        if home_id in team_lookup:
            game['home_team'] = team_lookup[home_id]['abbreviation']
            game['home_team_name'] = team_lookup[home_id]['full_name']
        if away_id in team_lookup:
            game['away_team'] = team_lookup[away_id]['abbreviation']
            game['away_team_name'] = team_lookup[away_id]['full_name']
        
        games.append(game)
    
    return games


def _parse_game_finder_results(games_df: pd.DataFrame) -> List[Dict]:
    """Parse games from LeagueGameFinder results"""
    games = []
    all_teams = get_all_teams()
    team_lookup = {t['id']: t for t in all_teams}
    
    # Group by GAME_ID since each team appears as separate row
    seen_games = set()
    
    for _, row in games_df.iterrows():
        game_id = row.get('GAME_ID')
        if game_id in seen_games:
            continue
        
        matchup = row.get('MATCHUP', '')
        team_id = row.get('TEAM_ID')
        
        # Determine home/away from matchup string (e.g., "LAL vs. GSW" or "LAL @ GSW")
        is_home = ' vs. ' in matchup
        
        game = {
            'game_id': game_id,
            'game_status': '',
        }
        
        if is_home:
            game['home_team_id'] = team_id
            if team_id in team_lookup:
                game['home_team'] = team_lookup[team_id]['abbreviation']
                game['home_team_name'] = team_lookup[team_id]['full_name']
            # Try to extract away team from matchup
            parts = matchup.split(' vs. ')
            if len(parts) == 2:
                away_abbrev = parts[1].strip()
                for t in all_teams:
                    if t['abbreviation'] == away_abbrev:
                        game['away_team'] = t['abbreviation']
                        game['away_team_name'] = t['full_name']
                        game['away_team_id'] = t['id']
                        break
        else:
            game['away_team_id'] = team_id
            if team_id in team_lookup:
                game['away_team'] = team_lookup[team_id]['abbreviation']
                game['away_team_name'] = team_lookup[team_id]['full_name']
            # Try to extract home team from matchup
            parts = matchup.split(' @ ')
            if len(parts) == 2:
                home_abbrev = parts[1].strip()
                for t in all_teams:
                    if t['abbreviation'] == home_abbrev:
                        game['home_team'] = t['abbreviation']
                        game['home_team_name'] = t['full_name']
                        game['home_team_id'] = t['id']
                        break
        
        if 'home_team' in game and 'away_team' in game:
            seen_games.add(game_id)
            games.append(game)
    
    return games


def get_team_players(team_abbrev: str) -> List[Dict]:
    """Get all players on a specific team"""
    team = get_team_by_abbreviation(team_abbrev)
    if not team:
        return []
    
    try:
        roster = get_team_roster(team['id'])
        if roster.empty:
            return []
        
        players = []
        for _, row in roster.iterrows():
            players.append({
                'id': row.get('PLAYER_ID'),
                'name': row.get('PLAYER'),
                'position': row.get('POSITION'),
                'number': row.get('NUM')
            })
        return players
    except Exception as e:
        print(f"Error getting team players: {e}")
        return []


if __name__ == "__main__":
    # Test data collection
    print("Testing data collection...")
    
    # Test player search
    player = search_player("LeBron James")
    if player:
        print(f"Found player: {player['full_name']} (ID: {player['id']})")
        
        # Get game logs
        logs = get_player_game_logs(player['id'])
        print(f"Game logs: {len(logs)} games")
        
        if not logs.empty:
            print(logs[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']].head())
    
    # Test team stats
    print("\nFetching team defensive stats...")
    def_stats = get_team_defensive_stats()
    if not def_stats.empty:
        print(f"Got stats for {len(def_stats)} teams")
    
    # Test cache effectiveness
    print("\nTesting cache (second lookup should be instant)...")
    import time as t
    start = t.time()
    player2 = search_player("LeBron James")
    print(f"Cached lookup took: {(t.time() - start)*1000:.2f}ms")
