"""Quick test to verify real data integration"""

from model import make_prediction
from period_boxscore_collector import load_player_period_stats
from advanced_stats_collector import get_player_advanced_stats

# Test players with collected data
test_players = [
    ("LeBron James", "CHI", 8.5, "points", "1h"),
    ("LeBron James", "CHI", 4.0, "points", "1q"),
    ("Julius Randle", "GSW", 10.0, "points", "1h"),
    ("Anfernee Simons", "BOS", 3.5, "rebounds", "1q"),
]

print("="*70)
print("TESTING REAL DATA INTEGRATION")
print("="*70)

for player, opp, line, prop, period in test_players:
    print(f"\n{'='*70}")
    print(f"Player: {player} | {prop.upper()} {period.upper()} | Line: {line}")
    print("="*70)
    
    # Check if real period data exists
    period_stats = load_player_period_stats(player)
    if period_stats:
        print(f"[OK] REAL PERIOD DATA FOUND!")
        print(f"     Games: {period_stats.get('games_collected', 0)}")
        if period == '1h':
            print(f"     Real 1H avg: {period_stats.get('1H_PTS_avg', 'N/A')}")
        else:
            print(f"     Real 1Q avg: {period_stats.get('1Q_PTS_avg', 'N/A')}")
    else:
        print(f"[--] No real period data (using estimates)")
    
    # Check advanced stats
    adv_stats = get_player_advanced_stats(player)
    if adv_stats:
        print(f"[OK] ADVANCED STATS FOUND!")
        print(f"     Usage%: {adv_stats.get('usage_pct', 0):.1%}")
        print(f"     Pace: {adv_stats.get('pace', 0):.1f}")
    
    # Make prediction
    result = make_prediction(player, opp, line, prop, True, period)
    
    if 'error' in result:
        print(f"[ERROR] {result['error']}")
    else:
        print(f"\nPREDICTION RESULT:")
        print(f"  Data Source: {result.get('data_source', 'UNKNOWN')}")
        print(f"  Has Real Data: {result.get('has_real_period_data', False)}")
        print(f"  Predicted: {result.get('predicted_1h', 0):.1f}")
        pick = result.get('pick', 'N/A')
        # Remove emoji for console
        pick_clean = pick.encode('ascii', 'replace').decode('ascii') if pick else 'N/A'
        print(f"  Pick: {pick_clean}")
        print(f"  Lock Score: {result.get('lock_score', 0)}")
        print(f"  Edge: {result.get('difference', 0):+.1f}")
        
        if result.get('advanced_boost'):
            print(f"  Advanced Boost: {result.get('advanced_boost'):.3f}x")
            boost_info = result.get('advanced_boost_info', {})
            for reason in boost_info.get('reasoning', []):
                print(f"    - {reason}")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
