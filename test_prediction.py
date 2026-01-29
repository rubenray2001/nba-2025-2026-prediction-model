"""Quick test of prediction with real data"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from model import make_prediction

# Test both 1H and 1Q
for period in ['1h', '1q']:
    print(f"\n{'='*50}")
    print(f"TESTING {period.upper()} - Myles Turner vs WAS, Line 4.5")
    print(f"{'='*50}")
    
    result = make_prediction('Myles Turner', 'WAS', 4.5, 'points', True, period)
    
    print(f"Predicted: {result.get('predicted_1h', 0):.1f}")
    print(f"Lock Score: {result.get('lock_score', 0)}")
    print(f"Has Real Data: {result.get('has_real_period_data', False)}")
    print(f"Data Source: {result.get('data_source', 'UNKNOWN')}")
    print(f"Pick: {result.get('pick', 'N/A')}")
    
    # Show key factors only
    print("\nKey Lock Factors:")
    for f in result.get('lock_factors', [])[:10]:
        name = ''.join(c for c in f.get('name', '') if ord(c) < 128)
        print(f"  {name}: {f.get('score')}")
