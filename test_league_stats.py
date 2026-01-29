"""Test league-wide player stats for usage%"""

from nba_api.stats.endpoints import leaguedashplayerstats
import time

print("Fetching league-wide advanced stats...")
print("="*60)

time.sleep(0.7)

try:
    # Get advanced stats for all players
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"  # This gets usage%, ts%, etc.
    )
    
    df = stats.get_data_frames()[0]
    
    print(f"Got {len(df)} players")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for LeBron
    lebron = df[df['PLAYER_NAME'] == 'LeBron James']
    if not lebron.empty:
        print(f"\n{'='*60}")
        print("LeBron James Advanced Stats:")
        print(f"{'='*60}")
        for col in lebron.columns:
            print(f"  {col}: {lebron[col].values[0]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
