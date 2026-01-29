"""
Quick test to see what the BoxScoreTraditionalV2 endpoint returns
"""

import time
from nba_api.stats.endpoints import boxscoretraditionalv2, playergamelog
from nba_api.stats.static import players

# Get a recent LeBron game
print("Getting LeBron's recent game...")
player_id = 2544  # LeBron James

game_log = playergamelog.PlayerGameLog(
    player_id=player_id,
    season="2025-26",
    season_type_all_star='Regular Season'
)
time.sleep(0.7)

df = game_log.get_data_frames()[0]
game_id = df['Game_ID'].iloc[0]
print(f"Testing with game_id: {game_id}")

# Test 1: Full game boxscore
print("\n--- TEST 1: Full Game (start=0, end=14) ---")
time.sleep(0.7)
try:
    bs_full = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=0,
        end_period=14
    )
    dfs = bs_full.get_data_frames()
    print(f"Number of dataframes returned: {len(dfs)}")
    if dfs:
        df_players = dfs[0]
        print(f"Player stats shape: {df_players.shape}")
        print(f"Columns: {list(df_players.columns)}")
        lebron_row = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron_row.empty:
            print(f"LeBron full game: {lebron_row[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'MIN']].values}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: 1st Half (periods 1-2)
print("\n--- TEST 2: 1st Half (start=1, end=2) ---")
time.sleep(0.7)
try:
    bs_1h = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=1,
        end_period=2
    )
    dfs = bs_1h.get_data_frames()
    print(f"Number of dataframes returned: {len(dfs)}")
    if dfs:
        df_players = dfs[0]
        print(f"Player stats shape: {df_players.shape}")
        lebron_row = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron_row.empty:
            print(f"LeBron 1H: {lebron_row[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'MIN']].values}")
        else:
            print("LeBron not found in 1H data")
            print(f"Available players: {df_players['PLAYER_NAME'].tolist()[:5]}...")
except Exception as e:
    print(f"Error: {e}")

# Test 3: 1st Quarter only (period 1)
print("\n--- TEST 3: 1st Quarter (start=1, end=1) ---")
time.sleep(0.7)
try:
    bs_1q = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=1,
        end_period=1
    )
    dfs = bs_1q.get_data_frames()
    print(f"Number of dataframes returned: {len(dfs)}")
    if dfs:
        df_players = dfs[0]
        print(f"Player stats shape: {df_players.shape}")
        lebron_row = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron_row.empty:
            print(f"LeBron 1Q: {lebron_row[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'MIN']].values}")
        else:
            print("LeBron not found in 1Q data")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Try with range parameters too
print("\n--- TEST 4: With range parameters ---")
time.sleep(0.7)
try:
    # Range type 1 might be needed for period filtering
    bs_range = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=1,
        end_period=2,
        start_range=0,
        end_range=28800,  # 28800 = end of 2nd quarter in 10ths of seconds
        range_type=2  # Try range type 2
    )
    dfs = bs_range.get_data_frames()
    print(f"Number of dataframes returned: {len(dfs)}")
    if dfs:
        df_players = dfs[0]
        print(f"Player stats shape: {df_players.shape}")
        lebron_row = df_players[df_players['PLAYER_ID'] == player_id]
        if not lebron_row.empty:
            print(f"LeBron (range): {lebron_row[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'MIN']].values}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- DONE ---")
