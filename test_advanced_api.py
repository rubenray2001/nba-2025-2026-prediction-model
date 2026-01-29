"""Test advanced box score API"""

from nba_api.stats.endpoints import boxscoreadvancedv2
import time

game_id = "0022500661"

print(f"Testing advanced boxscore for game {game_id}")
print("="*60)

time.sleep(0.7)

# Try full game first (no range params)
try:
    boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(
        game_id=game_id
    )
    
    dfs = boxscore.get_data_frames()
    print(f"Number of DataFrames: {len(dfs)}")
    
    for i, df in enumerate(dfs):
        print(f"\nDataFrame {i}: {len(df)} rows")
        if not df.empty:
            print(f"Columns: {list(df.columns)[:10]}...")
            print(df.head(2).to_string())
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Try with range parameters
print("\n" + "="*60)
print("Trying with range_type=2, end_range=7200 (1Q)")
print("="*60)

time.sleep(0.7)

try:
    boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(
        game_id=game_id,
        start_period=0,
        end_period=14,
        start_range=0,
        end_range=7200,
        range_type=2
    )
    
    dfs = boxscore.get_data_frames()
    print(f"Number of DataFrames: {len(dfs)}")
    
    for i, df in enumerate(dfs):
        print(f"\nDataFrame {i}: {len(df)} rows")
        if not df.empty:
            print(f"Columns: {list(df.columns)[:10]}...")
            print(df.head(2).to_string())
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
