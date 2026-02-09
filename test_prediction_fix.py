import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from model import make_prediction

# Test Bam Adebayo @ 18 PRA 1H
print("=" * 60)
print("BAM ADEBAYO @ 18 PRA 1H")
print("=" * 60)
result = make_prediction(
    player_name="Bam Adebayo",
    opponent="WAS",
    prop_line=18,
    prop_type="pra",
    is_home=False,
    period="1h"
)
pick = result.get('pick', 'N/A')
# Strip emojis for console
pick_clean = ''.join(c for c in pick if ord(c) < 128) if pick else 'N/A'
print(f"Pick: {pick_clean}")
print(f"Lock Score: {result.get('lock_score')}")
print(f"Predicted 1H: {result.get('predicted_1h')}")
print(f"Data Source: {result.get('data_source')}")
print("\nLock Factors:")
for f in result.get('lock_factors', [])[:10]:
    name = ''.join(c for c in f['name'] if ord(c) < 128)
    print(f"  {name}: {f['score']} - {f['desc'][:60]}")

# Test Rudy Gobert @ 12 PRA 1H
print("\n" + "=" * 60)
print("RUDY GOBERT @ 12 PRA 1H")
print("=" * 60)
result2 = make_prediction(
    player_name="Rudy Gobert",
    opponent="LAC",
    prop_line=12,
    prop_type="pra",
    is_home=False,
    period="1h"
)
pick2 = result2.get('pick', 'N/A')
pick2_clean = ''.join(c for c in pick2 if ord(c) < 128) if pick2 else 'N/A'
print(f"Pick: {pick2_clean}")
print(f"Lock Score: {result2.get('lock_score')}")
print(f"Predicted 1H: {result2.get('predicted_1h')}")
print(f"Data Source: {result2.get('data_source')}")
print("\nLock Factors:")
for f in result2.get('lock_factors', [])[:10]:
    name = ''.join(c for c in f['name'] if ord(c) < 128)
    print(f"  {name}: {f['score']} - {f['desc'][:60]}")
