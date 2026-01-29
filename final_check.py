"""
FINAL VERIFICATION - Real Data Check
=====================================
This shows exactly what the model predicts vs real data
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'period_stats')
from model import make_prediction

print("="*70)
print("  NBA PROP PREDICTOR - REAL DATA VERIFICATION")
print("="*70)

# Count total players
files = [f for f in os.listdir(DATA_DIR) if f.endswith('_period_stats.json') and not f.startswith('_')]
print(f"\n[OK] Total players with REAL period data: {len(files)}")

# Test players from user's list
test_players = [
    ('Khris Middleton', 2.5, 'Khris_Middleton'),
    ('Myles Turner', 4.5, 'Myles_Turner'),
    ('Bobby Portis', 3.5, 'Bobby_Portis'),
    ('Alex Sarr', 5.5, 'Alex_Sarr'),
]

print("\n" + "="*70)
print("  1Q PREDICTIONS vs REAL DATA")
print("="*70)
print(f"{'Player':<20} {'Line':>6} {'Real 1Q':>8} {'Predict':>8} {'Match?':>8} {'Score':>6}")
print("-"*70)

all_match = True
for name, line, filename in test_players:
    # Get real data
    filepath = os.path.join(DATA_DIR, f'{filename}_period_stats.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        real_1q = data.get('1Q_PTS_avg', 0)
        
        # Get prediction
        result = make_prediction(name, 'WAS', line, 'points', True, '1q')
        predicted = result.get('predicted_1h', 0)
        score = result.get('lock_score', 0)
        has_real = result.get('has_real_period_data', False)
        
        # Check if prediction matches real data
        match = abs(predicted - real_1q) < 0.1
        match_str = "YES" if match else "NO"
        if not match:
            all_match = False
        
        print(f"{name:<20} {line:>6.1f} {real_1q:>8.1f} {predicted:>8.1f} {match_str:>8} {score:>6}")
    else:
        print(f"{name:<20} - NO DATA FILE -")
        all_match = False

print("-"*70)

if all_match:
    print("\n[OK] ALL PREDICTIONS MATCH REAL DATA!")
else:
    print("\n[!!] SOME PREDICTIONS DON'T MATCH - CHECK ABOVE")

print("\n" + "="*70)
print("  1H PREDICTIONS vs REAL DATA")
print("="*70)
print(f"{'Player':<20} {'Line':>6} {'Real 1H':>8} {'Predict':>8} {'Match?':>8} {'Score':>6}")
print("-"*70)

all_match_1h = True
for name, line, filename in test_players:
    filepath = os.path.join(DATA_DIR, f'{filename}_period_stats.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        real_1h = data.get('1H_PTS_avg', 0)
        
        result = make_prediction(name, 'WAS', line, 'points', True, '1h')
        predicted = result.get('predicted_1h', 0)
        score = result.get('lock_score', 0)
        
        match = abs(predicted - real_1h) < 0.1
        match_str = "YES" if match else "NO"
        if not match:
            all_match_1h = False
        
        print(f"{name:<20} {line:>6.1f} {real_1h:>8.1f} {predicted:>8.1f} {match_str:>8} {score:>6}")

print("-"*70)

if all_match_1h:
    print("\n[OK] ALL 1H PREDICTIONS MATCH REAL DATA!")

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"  Players with real data: {len(files)}")
print(f"  1Q predictions accurate: {'YES' if all_match else 'NO'}")
print(f"  1H predictions accurate: {'YES' if all_match_1h else 'NO'}")
print("="*70)
