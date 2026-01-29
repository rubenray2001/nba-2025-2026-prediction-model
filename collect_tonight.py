"""Collect period stats for all players in tonight's games"""

import sys
import time
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2, commonteamroster
from nba_api.stats.static import teams
from period_boxscore_collector import collect_player_period_stats, load_player_period_stats, save_player_period_stats
from advanced_stats_collector import get_player_advanced_stats

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

def get_tonights_games():
    """Get all games scheduled for today"""
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n[INFO] Fetching games for {today}...")
    
    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        games = scoreboard.game_header.get_data_frame()
        
        if games.empty:
            print("[WARNING] No games found for today")
            return []
        
        game_list = []
        for _, game in games.iterrows():
            game_info = {
                'game_id': game['GAME_ID'],
                'home_team_id': game['HOME_TEAM_ID'],
                'away_team_id': game['VISITOR_TEAM_ID'],
                'status': game.get('GAME_STATUS_TEXT', 'Unknown')
            }
            game_list.append(game_info)
        
        print(f"[OK] Found {len(game_list)} games tonight")
        return game_list
    except Exception as e:
        print(f"[ERROR] Failed to get games: {e}")
        return []

def get_team_roster(team_id):
    """Get active roster for a team"""
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2025-26')
        players = roster.common_team_roster.get_data_frame()
        
        player_list = []
        for _, player in players.iterrows():
            player_list.append({
                'id': player['PLAYER_ID'],
                'name': player['PLAYER'],
                'team_id': team_id
            })
        return player_list
    except Exception as e:
        print(f"[ERROR] Failed to get roster for team {team_id}: {e}")
        return []

def get_team_name(team_id):
    """Get team abbreviation from ID"""
    all_teams = teams.get_teams()
    for team in all_teams:
        if team['id'] == team_id:
            return team['abbreviation']
    return str(team_id)

def main():
    print("=" * 70)
    print("COLLECTING DATA FOR TONIGHT'S PLAYERS")
    print("=" * 70)
    
    # Get tonight's games
    games = get_tonights_games()
    
    if not games:
        print("\nNo games tonight or unable to fetch schedule.")
        return
    
    # Get all team IDs playing tonight
    team_ids = set()
    for game in games:
        team_ids.add(game['home_team_id'])
        team_ids.add(game['away_team_id'])
    
    print(f"\n[INFO] {len(team_ids)} teams playing tonight")
    
    # Get all players from those teams
    all_players = []
    for team_id in team_ids:
        team_name = get_team_name(team_id)
        print(f"  Getting roster: {team_name}...", end=" ")
        players = get_team_roster(team_id)
        print(f"{len(players)} players")
        for p in players:
            p['team'] = team_name
        all_players.extend(players)
        time.sleep(0.5)  # Rate limit
    
    print(f"\n[INFO] Total players to collect: {len(all_players)}")
    
    # Check which players already have data
    need_collection = []
    already_have = []
    
    for player in all_players:
        existing = load_player_period_stats(player['name'])
        if existing and existing.get('games_collected', 0) >= 5:
            already_have.append(player['name'])
        else:
            need_collection.append(player)
    
    print(f"[INFO] Already have data: {len(already_have)} players")
    print(f"[INFO] Need to collect: {len(need_collection)} players")
    
    if not need_collection:
        print("\n[OK] All players already have data!")
        return
    
    # Collect data for players who need it
    print(f"\n{'='*70}")
    print(f"COLLECTING PERIOD STATS FOR {len(need_collection)} PLAYERS")
    print(f"{'='*70}\n")
    
    collected = 0
    failed = 0
    
    for i, player in enumerate(need_collection, 1):
        player_name = player['name'].encode('ascii', 'replace').decode('ascii')
        print(f"[{i}/{len(need_collection)}] {player_name} ({player['team']})")
        
        try:
            stats = collect_player_period_stats(player['name'], last_n_games=10)
            if stats and stats.get('games_collected', 0) > 0:
                # SAVE THE DATA!
                save_player_period_stats(player['name'], stats)
                print(f"    [OK] Collected {stats['games_collected']} games - SAVED")
                collected += 1
            else:
                print(f"    [SKIP] No games found")
                failed += 1
        except Exception as e:
            print(f"    [ERROR] {str(e)[:50]}")
            failed += 1
        
        # Rate limit (faster)
        time.sleep(0.5)
    
    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Collected: {collected}")
    print(f"Failed/Skipped: {failed}")
    print(f"Already had: {len(already_have)}")
    print(f"Total coverage: {collected + len(already_have)} players")

if __name__ == "__main__":
    main()
