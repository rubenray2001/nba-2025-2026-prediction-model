"""
Advanced Stats Collector for Enhanced Prop Predictions

Collects league-wide advanced metrics for all players:
- Usage Percentage (key for points props)
- Pace (game speed affects all props)
- Rebound Percentage (for rebound props)
- Assist Percentage (for assist props)
- True Shooting, PIE, etc.

These metrics help predict which players are likely to hit OVER/UNDER.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, List

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players

# ============================================================================
# Configuration
# ============================================================================

CURRENT_SEASON = "2025-26"
ADVANCED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'advanced_stats')
ALL_PLAYERS_FILE = os.path.join(ADVANCED_DATA_DIR, '_all_players_advanced.json')

# Ensure directory exists
os.makedirs(ADVANCED_DATA_DIR, exist_ok=True)


# ============================================================================
# Fetch All Players Advanced Stats
# ============================================================================

def fetch_all_players_advanced_stats(season: str = CURRENT_SEASON) -> Optional[Dict]:
    """
    Fetch advanced stats for ALL players in one API call.
    Much more efficient than per-player calls.
    
    Returns dict keyed by player name with their advanced stats.
    """
    print(f"[INFO] Fetching advanced stats for all players ({season})...")
    
    try:
        time.sleep(0.7)
        
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced"
        )
        
        df = stats.get_data_frames()[0]
        
        if df.empty:
            print("[ERROR] No data returned")
            return None
        
        print(f"[OK] Got advanced stats for {len(df)} players")
        
        # Convert to dict keyed by player name
        result = {
            'season': season,
            'collected_at': datetime.now().isoformat(),
            'player_count': len(df),
            'players': {}
        }
        
        for _, row in df.iterrows():
            player_name = row.get('PLAYER_NAME', '')
            if player_name:
                result['players'][player_name] = {
                    'player_id': int(row.get('PLAYER_ID', 0)),
                    'team': row.get('TEAM_ABBREVIATION', ''),
                    'games_played': int(row.get('GP', 0)),
                    'minutes_per_game': float(row.get('MIN', 0)),
                    
                    # Key advanced stats
                    'usage_pct': float(row.get('USG_PCT', 0)),
                    'ast_pct': float(row.get('AST_PCT', 0)),
                    'ast_ratio': float(row.get('AST_RATIO', 0)),
                    'oreb_pct': float(row.get('OREB_PCT', 0)),
                    'dreb_pct': float(row.get('DREB_PCT', 0)),
                    'reb_pct': float(row.get('REB_PCT', 0)),
                    'ts_pct': float(row.get('TS_PCT', 0)),
                    'efg_pct': float(row.get('EFG_PCT', 0)),
                    'pace': float(row.get('PACE', 0)),
                    'pie': float(row.get('PIE', 0)),
                    'off_rating': float(row.get('OFF_RATING', 0)),
                    'def_rating': float(row.get('DEF_RATING', 0)),
                    'net_rating': float(row.get('NET_RATING', 0)),
                    
                    # Rankings (lower = better)
                    'usage_rank': int(row.get('USG_PCT_RANK', 999)),
                    'ast_rank': int(row.get('AST_PCT_RANK', 999)),
                    'reb_rank': int(row.get('REB_PCT_RANK', 999)),
                    'pie_rank': int(row.get('PIE_RANK', 999)),
                }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch advanced stats: {e}")
        return None


def save_all_players_advanced(data: Dict):
    """Save all players advanced stats to JSON"""
    with open(ALL_PLAYERS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved advanced stats to {ALL_PLAYERS_FILE}")


def load_all_players_advanced() -> Optional[Dict]:
    """Load all players advanced stats from JSON"""
    if os.path.exists(ALL_PLAYERS_FILE):
        with open(ALL_PLAYERS_FILE, 'r') as f:
            return json.load(f)
    return None


# ============================================================================
# Helper Functions for Model Integration
# ============================================================================

def get_player_advanced_stats(player_name: str) -> Optional[Dict]:
    """Get advanced stats for a specific player"""
    data = load_all_players_advanced()
    if data and 'players' in data:
        return data['players'].get(player_name)
    return None


def get_player_usage_pct(player_name: str) -> Optional[float]:
    """
    Get player's usage percentage.
    
    Usage% = how many team plays end with this player.
    High usage (>25%) = primary scoring option = more likely to hit points OVER
    """
    stats = get_player_advanced_stats(player_name)
    if stats:
        return stats.get('usage_pct')
    return None


def get_player_ast_pct(player_name: str) -> Optional[float]:
    """
    Get player's assist percentage.
    
    AST% = % of teammate FGs assisted by this player.
    High assist% (>25%) = primary playmaker = good for assist props
    """
    stats = get_player_advanced_stats(player_name)
    if stats:
        return stats.get('ast_pct')
    return None


def get_player_reb_pct(player_name: str) -> Optional[float]:
    """
    Get player's rebound percentage.
    
    REB% = % of available rebounds grabbed.
    High reb% (>15%) = dominant rebounder = good for rebound props
    """
    stats = get_player_advanced_stats(player_name)
    if stats:
        return stats.get('reb_pct')
    return None


def get_player_pace(player_name: str) -> Optional[float]:
    """
    Get player's team pace.
    
    PACE = possessions per 48 minutes.
    High pace (>102) = fast game = more scoring opportunities
    """
    stats = get_player_advanced_stats(player_name)
    if stats:
        return stats.get('pace')
    return None


# ============================================================================
# Usage Boost Calculation for Model
# ============================================================================

def calculate_usage_boost(player_name: str, prop_type: str = 'points') -> float:
    """
    Calculate a boost/penalty multiplier based on usage stats.
    
    Returns:
        Multiplier (0.9 - 1.15) to apply to predictions
    """
    stats = get_player_advanced_stats(player_name)
    if not stats:
        return 1.0
    
    boost = 1.0
    
    if prop_type == 'points':
        # High usage = more likely to hit points OVER
        usage = stats.get('usage_pct', 0)
        if usage >= 0.30:  # Elite usage (top 10)
            boost = 1.10
        elif usage >= 0.27:  # High usage
            boost = 1.05
        elif usage >= 0.24:  # Above average
            boost = 1.02
        elif usage < 0.18:  # Low usage
            boost = 0.95
            
    elif prop_type == 'assists':
        # High assist% = more likely to hit assist OVER
        ast_pct = stats.get('ast_pct', 0)
        if ast_pct >= 0.35:  # Elite playmaker
            boost = 1.12
        elif ast_pct >= 0.28:  # Good playmaker
            boost = 1.06
        elif ast_pct >= 0.20:  # Above average
            boost = 1.02
        elif ast_pct < 0.12:  # Non-playmaker
            boost = 0.92
            
    elif prop_type == 'rebounds':
        # High rebound% = more likely to hit rebound OVER
        reb_pct = stats.get('reb_pct', 0)
        if reb_pct >= 0.18:  # Elite rebounder
            boost = 1.10
        elif reb_pct >= 0.14:  # Good rebounder
            boost = 1.05
        elif reb_pct >= 0.10:  # Average
            boost = 1.01
        elif reb_pct < 0.06:  # Poor rebounder
            boost = 0.92
    
    return boost


def calculate_pace_adjustment(player_name: str) -> float:
    """
    Calculate pace-based adjustment.
    
    Fast-paced teams have more possessions = more stat opportunities.
    
    Returns:
        Multiplier (0.95 - 1.08)
    """
    pace = get_player_pace(player_name)
    if pace is None:
        return 1.0
    
    # League average pace is ~100
    if pace >= 104:  # Very fast
        return 1.06
    elif pace >= 102:  # Fast
        return 1.03
    elif pace >= 100:  # Average
        return 1.0
    elif pace >= 98:  # Slow
        return 0.98
    else:  # Very slow
        return 0.95


def get_comprehensive_boost(player_name: str, prop_type: str) -> Dict:
    """
    Get all boost factors for a player/prop combination.
    
    Returns dict with:
        - usage_boost: Based on usage%
        - pace_boost: Based on team pace
        - combined_boost: Product of all boosts
        - reasoning: Explanation string
    """
    usage_boost = calculate_usage_boost(player_name, prop_type)
    pace_boost = calculate_pace_adjustment(player_name)
    combined = usage_boost * pace_boost
    
    stats = get_player_advanced_stats(player_name)
    reasoning = []
    
    if stats:
        if prop_type == 'points':
            usage = stats.get('usage_pct', 0)
            if usage >= 0.27:
                reasoning.append(f"High usage ({usage:.1%}) - primary scorer")
            elif usage < 0.18:
                reasoning.append(f"Low usage ({usage:.1%}) - limited touches")
                
        elif prop_type == 'assists':
            ast = stats.get('ast_pct', 0)
            if ast >= 0.28:
                reasoning.append(f"Elite playmaker ({ast:.1%} AST%)")
            elif ast < 0.12:
                reasoning.append(f"Non-playmaker ({ast:.1%} AST%)")
                
        elif prop_type == 'rebounds':
            reb = stats.get('reb_pct', 0)
            if reb >= 0.14:
                reasoning.append(f"Strong rebounder ({reb:.1%} REB%)")
            elif reb < 0.06:
                reasoning.append(f"Weak rebounder ({reb:.1%} REB%)")
        
        pace = stats.get('pace', 0)
        if pace >= 104:
            reasoning.append(f"Fast pace ({pace:.1f}) - more opportunities")
        elif pace < 98:
            reasoning.append(f"Slow pace ({pace:.1f}) - fewer opportunities")
    
    return {
        'usage_boost': round(usage_boost, 3),
        'pace_boost': round(pace_boost, 3),
        'combined_boost': round(combined, 3),
        'reasoning': reasoning
    }


# ============================================================================
# Show Stats Summary
# ============================================================================

def show_top_usage_players(n: int = 20):
    """Show top N players by usage%"""
    data = load_all_players_advanced()
    if not data:
        print("No data. Run fetch first.")
        return
    
    players = [(name, p) for name, p in data['players'].items()]
    players.sort(key=lambda x: x[1].get('usage_pct', 0), reverse=True)
    
    print(f"\nTop {n} Players by Usage%:")
    print("="*60)
    for i, (name, stats) in enumerate(players[:n], 1):
        usage = stats.get('usage_pct', 0)
        team = stats.get('team', '???')
        # Handle unicode names
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        print(f"{i:2}. {safe_name:25} ({team}) - {usage:.1%}")


def show_player_summary(player_name: str):
    """Show detailed stats for a player"""
    stats = get_player_advanced_stats(player_name)
    if not stats:
        print(f"No data for {player_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Advanced Stats: {player_name}")
    print(f"{'='*60}")
    print(f"Team: {stats.get('team')}")
    print(f"Games: {stats.get('games_played')}")
    print(f"Minutes: {stats.get('minutes_per_game'):.1f}")
    print(f"\nKey Metrics:")
    print(f"  Usage%: {stats.get('usage_pct', 0):.1%} (rank: #{stats.get('usage_rank', '?')})")
    print(f"  Assist%: {stats.get('ast_pct', 0):.1%} (rank: #{stats.get('ast_rank', '?')})")
    print(f"  Rebound%: {stats.get('reb_pct', 0):.1%} (rank: #{stats.get('reb_rank', '?')})")
    print(f"  Pace: {stats.get('pace', 0):.1f}")
    print(f"  PIE: {stats.get('pie', 0):.3f} (rank: #{stats.get('pie_rank', '?')})")
    print(f"  TS%: {stats.get('ts_pct', 0):.1%}")
    
    # Show boosts
    for prop in ['points', 'assists', 'rebounds']:
        boost = get_comprehensive_boost(player_name, prop)
        print(f"\n{prop.upper()} Prop Boost: {boost['combined_boost']:.3f}x")
        for r in boost['reasoning']:
            print(f"  - {r}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect advanced stats')
    parser.add_argument('--fetch', action='store_true', help='Fetch fresh data from API')
    parser.add_argument('--top', type=int, default=None, help='Show top N by usage')
    parser.add_argument('--player', type=str, default=None, help='Show player stats')
    
    args = parser.parse_args()
    
    if args.fetch:
        data = fetch_all_players_advanced_stats()
        if data:
            save_all_players_advanced(data)
            print(f"\nCollected {data['player_count']} players!")
    
    if args.top:
        show_top_usage_players(args.top)
    
    if args.player:
        show_player_summary(args.player)
    
    if not args.fetch and not args.top and not args.player:
        # Default: fetch and show top 15
        data = load_all_players_advanced()
        if not data:
            print("No cached data. Fetching...")
            data = fetch_all_players_advanced_stats()
            if data:
                save_all_players_advanced(data)
        
        if data:
            print(f"\nData from: {data.get('collected_at', 'Unknown')}")
            print(f"Total players: {data.get('player_count', 0)}")
            show_top_usage_players(15)
            
            # Show example player
            print("\n")
            show_player_summary("LeBron James")
