"""
Test to understand the range parameters better
"""

import time
from nba_api.stats.endpoints import boxscoretraditionalv2, playergamelog

player_id = 2544  # LeBron James

# Get a recent game
game_log = playergamelog.PlayerGameLog(
    player_id=player_id,
    season="2025-26",
    season_type_all_star='Regular Season'
)
time.sleep(0.7)

df = game_log.get_data_frames()[0]
game_id = df['Game_ID'].iloc[0]
full_game_pts = df['PTS'].iloc[0]
print(f"Game ID: {game_id}")
print(f"LeBron full game PTS from game log: {full_game_pts}")

# Understanding range:
# - NBA quarters are 12 minutes each
# - Range appears to be in 10ths of a second
# - 1 minute = 600 (10ths of second)
# - Q1 end = 12 min = 7200
# - Q2 end (1H) = 24 min = 14400
# - Q3 end = 36 min = 21600
# - Q4 end = 48 min = 28800

print("\n--- Testing different range values ---")

# Test: Full game with range
print("\nFull Game (range_type=2, end_range=28800):")
time.sleep(0.7)
try:
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=0,
        end_period=10,
        start_range=0,
        end_range=28800,
        range_type=2
    )
    dfs = bs.get_data_frames()
    if dfs and len(dfs[0]) > 0:
        df_players = dfs[0]
        lebron = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron.empty:
            print(f"  PTS: {lebron['PTS'].values[0]}, MIN: {lebron['MIN'].values[0]}")
except Exception as e:
    print(f"  Error: {e}")

# Test: 1st Half (0-14400 range = first 24 minutes)
print("\n1st Half (range_type=2, end_range=14400):")
time.sleep(0.7)
try:
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=0,
        end_period=10,
        start_range=0,
        end_range=14400,  # 24 minutes in 10ths of second
        range_type=2
    )
    dfs = bs.get_data_frames()
    if dfs and len(dfs[0]) > 0:
        df_players = dfs[0]
        lebron = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron.empty:
            print(f"  PTS: {lebron['PTS'].values[0]}, MIN: {lebron['MIN'].values[0]}")
    else:
        print("  No data returned")
except Exception as e:
    print(f"  Error: {e}")

# Test: 1st Quarter (0-7200 range = first 12 minutes)
print("\n1st Quarter (range_type=2, end_range=7200):")
time.sleep(0.7)
try:
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=0,
        end_period=10,
        start_range=0,
        end_range=7200,  # 12 minutes in 10ths of second
        range_type=2
    )
    dfs = bs.get_data_frames()
    if dfs and len(dfs[0]) > 0:
        df_players = dfs[0]
        lebron = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron.empty:
            print(f"  PTS: {lebron['PTS'].values[0]}, MIN: {lebron['MIN'].values[0]}")
    else:
        print("  No data returned")
except Exception as e:
    print(f"  Error: {e}")

# Test: 2nd Quarter only (7200-14400 range)
print("\n2nd Quarter Only (range_type=2, start=7200, end=14400):")
time.sleep(0.7)
try:
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=0,
        end_period=10,
        start_range=7200,
        end_range=14400,
        range_type=2
    )
    dfs = bs.get_data_frames()
    if dfs and len(dfs[0]) > 0:
        df_players = dfs[0]
        lebron = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron.empty:
            print(f"  PTS: {lebron['PTS'].values[0]}, MIN: {lebron['MIN'].values[0]}")
    else:
        print("  No data returned")
except Exception as e:
    print(f"  Error: {e}")

print("\n--- DONE ---")
