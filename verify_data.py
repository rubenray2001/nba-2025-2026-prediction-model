"""Verify real data is working correctly"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'period_stats')

# Check how many players have real data
files = [f for f in os.listdir(DATA_DIR) if f.endswith('_period_stats.json') and not f.startswith('_')]
print(f"Total players with real data: {len(files)}")

# Test a few players
test_players = ['Myles_Turner', 'Bobby_Portis', 'Khris_Middleton']

print("\n" + "="*60)
print("REAL DATA CHECK")
print("="*60)

for player_file in test_players:
    filepath = os.path.join(DATA_DIR, f'{player_file}_period_stats.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"\n{data.get('player_name', player_file)}:")
        print(f"  Games collected: {data.get('games_collected', 0)}")
        print(f"  1Q PTS avg: {data.get('1Q_PTS_avg', 'N/A')}")
        print(f"  1H PTS avg: {data.get('1H_PTS_avg', 'N/A')}")
    else:
        print(f"\n{player_file}: NO DATA FILE")

# Now test predictions
print("\n" + "="*60)
print("PREDICTION TEST")
print("="*60)

from model import make_prediction

test_cases = [
    ('Myles Turner', 4.5, '1q'),
    ('Myles Turner', 4.5, '1h'),
    ('Bobby Portis', 3.5, '1q'),
    ('Khris Middleton', 2.5, '1q'),
]

for player, line, period in test_cases:
    result = make_prediction(player, 'WAS', line, 'points', True, period)
    pred = result.get('predicted_1h', 0)
    score = result.get('lock_score', 0)
    pick = result.get('pick', 'N/A')
    has_real = result.get('has_real_period_data', False)
    edge = pred - line
    
    status = "REAL" if has_real else "EST"
    print(f"\n{player} | {period.upper()} | Line: {line}")
    print(f"  Predicted: {pred:.1f} | Edge: {edge:+.1f}")
    print(f"  Lock Score: {score} | Data: {status}")
    print(f"  Pick: {pick}")
