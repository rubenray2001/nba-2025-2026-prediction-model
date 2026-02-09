import sys
sys.path.insert(0, '.')
from model import get_real_hit_rate_and_variance

# Test Bam Adebayo @ 18 PRA 1H
data = get_real_hit_rate_and_variance('Bam Adebayo', 'pra', 18, '1h')
print('Bam Adebayo @ 18 PRA 1H:')
print(f"  Under hit rate: {data['under_hit_rate']*100:.1f}%")
print(f"  Over hit rate: {data['over_hit_rate']*100:.1f}%")
print(f"  Games counted: {data['games_counted']}")
print(f"  Range: {data['min_value']:.0f} - {data['max_value']:.0f}")
print(f"  Volatile: {data['is_volatile']}")

# Test Rudy Gobert @ 12 PRA 1H
data2 = get_real_hit_rate_and_variance('Rudy Gobert', 'pra', 12, '1h')
print('\nRudy Gobert @ 12 PRA 1H:')
print(f"  Under hit rate: {data2['under_hit_rate']*100:.1f}%")
print(f"  Over hit rate: {data2['over_hit_rate']*100:.1f}%")
print(f"  Games counted: {data2['games_counted']}")
print(f"  Range: {data2['min_value']:.0f} - {data2['max_value']:.0f}")
print(f"  Volatile: {data2['is_volatile']}")
