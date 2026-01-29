"""
CHECK ANY PLAYER'S REAL DATA
Usage: python check_player.py "Player Name"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json

if len(sys.argv) < 2:
    print("Usage: python check_player.py \"Player Name\"")
    print("Example: python check_player.py \"Myles Turner\"")
    sys.exit(1)

player_name = sys.argv[1]
safe_name = player_name.replace(' ', '_')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'period_stats')
filepath = os.path.join(DATA_DIR, f'{safe_name}_period_stats.json')

print(f"\n{'='*50}")
print(f"  CHECKING: {player_name}")
print(f"{'='*50}")

if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\n[FILE FOUND] {filepath}")
    print(f"\nREAL DATA FROM {data.get('games_collected', 0)} GAMES:")
    print(f"  1Q Points Avg: {data.get('1Q_PTS_avg', 'N/A')}")
    print(f"  1H Points Avg: {data.get('1H_PTS_avg', 'N/A')}")
    print(f"  1Q Rebounds Avg: {data.get('1Q_REB_avg', 'N/A')}")
    print(f"  1H Rebounds Avg: {data.get('1H_REB_avg', 'N/A')}")
    print(f"  1Q Assists Avg: {data.get('1Q_AST_avg', 'N/A')}")
    print(f"  1H Assists Avg: {data.get('1H_AST_avg', 'N/A')}")
    
    # Show last 3 games
    games = data.get('games_data', [])[:3]
    if games:
        print(f"\nLAST 3 GAMES (1Q POINTS):")
        for g in games:
            print(f"  Game {g.get('game_id', '?')}: {g.get('1Q_PTS', '?')} pts")
    
    # Now check prediction
    print(f"\n{'='*50}")
    print("  MODEL PREDICTION CHECK")
    print(f"{'='*50}")
    
    from model import make_prediction
    for period in ['1q', '1h']:
        result = make_prediction(player_name, 'LAL', 10.0, 'points', True, period)
        pred = result.get('predicted_1h', 0)
        has_real = result.get('has_real_period_data', False)
        source = result.get('data_source', 'UNKNOWN')
        
        real_avg = data.get(f'{period.upper()}_PTS_avg', 0)
        match = "YES" if abs(pred - real_avg) < 0.1 else "NO"
        
        print(f"\n{period.upper()} Prediction:")
        print(f"  Real Avg: {real_avg}")
        print(f"  Model Predicts: {pred}")
        print(f"  Match: {match}")
        print(f"  Using Real Data: {has_real} ({source})")

else:
    print(f"\n[NOT FOUND] No data file for {player_name}")
    print(f"Looked for: {filepath}")
    
    # Try to find similar names
    all_files = os.listdir(DATA_DIR)
    search = player_name.lower().replace(' ', '_')
    matches = [f for f in all_files if search in f.lower()]
    if matches:
        print(f"\nDid you mean one of these?")
        for m in matches[:5]:
            print(f"  - {m.replace('_period_stats.json', '').replace('_', ' ')}")
