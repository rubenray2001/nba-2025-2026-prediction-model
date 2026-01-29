"""Fast collection of star players for tonight"""
import sys
sys.stdout.reconfigure(line_buffering=True)

from period_boxscore_collector import collect_player_period_stats, save_player_period_stats

stars = [
    'Stephen Curry', 'Anthony Edwards', 'Luka Doncic', 'Kevin Durant',
    'Donovan Mitchell', 'Paolo Banchero', 'Alperen Sengun',
    'Julius Randle', 'Coby White', 'Nikola Vucevic', 'Evan Mobley',
    'Jaren Jackson Jr.', 'Jalen Suggs', 'Jarrett Allen', 'Draymond Green',
    'Amen Thompson', 'Jabari Smith Jr.', 'Jaden McDaniels', 'Naz Reid',
    'Brandon Miller', 'LaMelo Ball', 'Tyrese Maxey', 'Miles Bridges',
    'Wendell Carter Jr.', 'Matas Buzelis', 'Josh Giddey', 'Deandre Ayton'
]

print(f"Collecting {len(stars)} star players...")
print("=" * 50)

for i, name in enumerate(stars, 1):
    print(f"[{i}/{len(stars)}] {name}...", end=" ", flush=True)
    try:
        stats = collect_player_period_stats(name, last_n_games=5)
        if stats and stats.get('games_collected', 0) > 0:
            save_player_period_stats(name, stats)
            print(f"OK ({stats['games_collected']} games)")
        else:
            print("SKIP")
    except Exception as e:
        print("ERR")

print("=" * 50)
print("DONE!")
