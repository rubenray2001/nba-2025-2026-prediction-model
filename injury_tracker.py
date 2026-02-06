"""
NBA Injury Tracker Module
Fetches current injury data and calculates impact on predictions.

Features:
- Scrapes ESPN injury report
- Identifies OUT/Doubtful/Questionable players
- Calculates teammate absence impact (usage boost)
- Calculates opponent injury impact (easier/harder matchup)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import re
import json
import os
from functools import lru_cache

# Cache file for injury data (refreshes daily)
INJURY_CACHE_FILE = "data/injury_cache.json"


def fetch_espn_injuries() -> Dict[str, List[Dict]]:
    """
    Fetch current NBA injuries from ESPN.
    Returns dict mapping team abbreviation to list of injured players.
    """
    url = "https://www.espn.com/nba/injuries"
    
    # Team name mapping
    team_mapping = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    # Initialize all teams
    injuries = {abbrev: [] for abbrev in team_mapping.values()}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Method 1: Try to find tables with injury data
        current_team_abbrev = None
        
        # Find all text and tables
        all_text = soup.get_text('\n')
        lines = all_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for team names
            for team_name, team_abbrev in team_mapping.items():
                if team_name == line or (team_name in line and len(line) < len(team_name) + 10):
                    current_team_abbrev = team_abbrev
                    break
            
            # Look for injury status keywords
            if current_team_abbrev:
                status = None
                if 'Out' in line:
                    status = 'Out'
                elif 'Day-To-Day' in line:
                    status = 'Day-To-Day'
                elif 'Questionable' in line:
                    status = 'Questionable'
                elif 'Doubtful' in line:
                    status = 'Doubtful'
                
                if status:
                    # Try to extract player name from previous lines
                    for j in range(max(0, i-5), i):
                        prev_line = lines[j].strip()
                        # Player names typically have 2-3 words, start with capital
                        if prev_line and len(prev_line) > 3 and prev_line[0].isupper():
                            # Skip if it's a team name or header
                            if prev_line not in team_mapping and 'NAME' not in prev_line and 'POS' not in prev_line:
                                # Check if it looks like a name (has letters, maybe periods/apostrophes)
                                if re.match(r'^[A-Z][a-z\'\.\-]+\s+[A-Z]', prev_line):
                                    player_name = prev_line.split('\t')[0].strip()
                                    # Don't add duplicates
                                    existing_names = [p['name'].lower() for p in injuries[current_team_abbrev]]
                                    if player_name.lower() not in existing_names:
                                        injuries[current_team_abbrev].append({
                                            'name': player_name,
                                            'position': '',
                                            'status': status,
                                            'team': current_team_abbrev
                                        })
                                    break
        
        # Method 2: Pattern-based extraction from raw HTML
        # Look for player profile links followed by position and status
        # ESPN format often has: [Player Name](url)|POS|Date|Status|
        profile_pattern = r'\[([A-Za-z\'\.\-\s]+)\]\([^)]+\)\|([CGFPGSF]{1,2})\|[^|]+\|(Out|Day-To-Day|Questionable|Doubtful)\|'
        matches = re.findall(profile_pattern, content)
        
        # Track position in content to determine team
        for match in matches:
            player_name, position, status = match[0].strip(), match[1], match[2]
            player_idx = content.find(f'[{player_name}]')
            if player_idx > 0:
                search_text = content[max(0, player_idx-3000):player_idx]
                for team_name, team_abbrev in team_mapping.items():
                    if team_name in search_text:
                        existing_names = [p['name'].lower() for p in injuries[team_abbrev]]
                        if player_name.lower() not in existing_names:
                            injuries[team_abbrev].append({
                                'name': player_name,
                                'position': position,
                                'status': status,
                                'team': team_abbrev
                            })
                        break
        
        # Method 3: Simple text pattern
        # Look for lines like "Player Name|G|Feb 5|Out|Comment"
        simple_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\'\.\-]+)+)\|([CGFPGSF]{1,2})\|[A-Za-z]+\s+\d+\|(Out|Day-To-Day|Questionable|Doubtful)'
        simple_matches = re.findall(simple_pattern, content)
        
        for match in simple_matches:
            player_name, position, status = match[0].strip(), match[1], match[2]
            player_idx = content.find(player_name)
            if player_idx > 0:
                search_text = content[max(0, player_idx-3000):player_idx]
                for team_name, team_abbrev in team_mapping.items():
                    if team_name in search_text:
                        existing_names = [p['name'].lower() for p in injuries[team_abbrev]]
                        if player_name.lower() not in existing_names:
                            injuries[team_abbrev].append({
                                'name': player_name,
                                'position': position,
                                'status': status,
                                'team': team_abbrev
                            })
                        break
        
        return injuries
        
    except Exception as e:
        print(f"[WARNING] Could not fetch ESPN injuries: {e}")
        return injuries


def parse_injury_data_from_text(text: str) -> Dict[str, List[Dict]]:
    """
    Parse injury data from raw ESPN page text.
    More robust parsing.
    """
    injuries = {team: [] for team in [
        'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]}
    
    team_mapping = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    current_team = None
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a team header
        for full_name, abbrev in team_mapping.items():
            if full_name in line and len(line) < 100:  # Likely a header
                current_team = abbrev
                break
        
        # Check if this line contains injury info
        if current_team and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                # Look for status
                status = None
                for p in parts:
                    if p in ['Out', 'Day-To-Day', 'Questionable', 'Doubtful']:
                        status = p
                        break
                
                if status:
                    # First part is usually the player name
                    player_name = parts[0]
                    # Clean up the name
                    player_name = re.sub(r'\[.*?\]', '', player_name)
                    player_name = re.sub(r'\(.*?\)', '', player_name)
                    player_name = player_name.strip()
                    
                    # Position is usually second
                    position = parts[1] if len(parts) > 1 else ''
                    
                    if player_name and len(player_name) > 2:
                        injuries[current_team].append({
                            'name': player_name,
                            'position': position,
                            'status': status,
                            'team': current_team
                        })
    
    return injuries


MANUAL_INJURIES_FILE = "data/manual_injuries.json"

def load_manual_injuries() -> Dict[str, List[Dict]]:
    """
    Load manually entered injuries from JSON file.
    Format: {"ATL": [{"name": "Player", "status": "Out"}], ...}
    """
    if os.path.exists(MANUAL_INJURIES_FILE):
        try:
            with open(MANUAL_INJURIES_FILE, 'r') as f:
                data = json.load(f)
                # Validate and add team field
                for team, players in data.items():
                    for player in players:
                        player['team'] = team
                        if 'position' not in player:
                            player['position'] = ''
                return data
        except Exception as e:
            print(f"[WARNING] Error reading manual injuries: {e}")
    return {}


def fetch_cbs_injuries() -> Dict[str, List[Dict]]:
    """
    Fetch injuries from CBS Sports (backup source).
    """
    url = "https://www.cbssports.com/nba/injuries/"
    
    team_mapping = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 
        'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    injuries = {abbrev: [] for abbrev in set(team_mapping.values())}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # CBS uses tables with team sections
        current_team = None
        
        # Get all text
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            for row in rows:
                text = row.get_text(' ', strip=True)
                
                # Check if this is a team header
                for team_name, abbrev in team_mapping.items():
                    if team_name in text:
                        current_team = abbrev
                        break
                
                # Check for injury status
                if current_team:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        player_cell = cells[0].get_text(strip=True)
                        status = None
                        
                        for cell in cells:
                            cell_text = cell.get_text(strip=True)
                            if 'Out' in cell_text:
                                status = 'Out'
                            elif 'Doubtful' in cell_text:
                                status = 'Doubtful'
                            elif 'Questionable' in cell_text:
                                status = 'Questionable'
                            elif 'Day-To-Day' in cell_text or 'Day-to-Day' in cell_text:
                                status = 'Day-To-Day'
                        
                        if status and player_cell and len(player_cell) > 2:
                            # Clean player name
                            player_name = re.sub(r'\s+(G|F|C|SG|PG|SF|PF)$', '', player_cell)
                            existing = [p['name'].lower() for p in injuries[current_team]]
                            if player_name.lower() not in existing:
                                injuries[current_team].append({
                                    'name': player_name,
                                    'position': '',
                                    'status': status,
                                    'team': current_team
                                })
        
        return injuries
        
    except Exception as e:
        print(f"[WARNING] Could not fetch CBS injuries: {e}")
        return injuries


def get_injuries(force_refresh: bool = False) -> Dict[str, List[Dict]]:
    """
    Get current injuries from multiple sources:
    1. Manual entries (highest priority - always included)
    2. Cache (if fresh)
    3. ESPN scrape
    4. CBS scrape (fallback)
    """
    # Initialize empty injuries
    all_teams = [
        'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]
    injuries = {team: [] for team in all_teams}
    
    # Always load manual injuries first (highest priority)
    manual = load_manual_injuries()
    for team, players in manual.items():
        if team in injuries:
            for player in players:
                injuries[team].append(player)
    
    # Check cache if not forcing refresh
    if not force_refresh and os.path.exists(INJURY_CACHE_FILE):
        try:
            with open(INJURY_CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                hours_old = (datetime.now() - cache_time).total_seconds() / 3600
                
                if hours_old < 4:  # Cache valid for 4 hours
                    cached_injuries = cache.get('injuries', {})
                    # Merge with manual (manual takes priority)
                    for team, players in cached_injuries.items():
                        if team in injuries:
                            existing_names = [p['name'].lower() for p in injuries[team]]
                            for player in players:
                                if player['name'].lower() not in existing_names:
                                    injuries[team].append(player)
                    return injuries
        except Exception as e:
            print(f"[WARNING] Error reading injury cache: {e}")
    
    # Try ESPN first
    print("[INFO] Fetching fresh injury data from ESPN...")
    espn_injuries = fetch_espn_injuries()
    
    # If ESPN didn't get much, try CBS
    total_espn = sum(len(v) for v in espn_injuries.values())
    if total_espn < 10:
        print("[INFO] ESPN data sparse, trying CBS Sports...")
        cbs_injuries = fetch_cbs_injuries()
        # Merge CBS into espn_injuries
        for team, players in cbs_injuries.items():
            if team in espn_injuries:
                existing = [p['name'].lower() for p in espn_injuries[team]]
                for player in players:
                    if player['name'].lower() not in existing:
                        espn_injuries[team].append(player)
    
    # Merge fetched data with manual
    for team, players in espn_injuries.items():
        if team in injuries:
            existing_names = [p['name'].lower() for p in injuries[team]]
            for player in players:
                if player['name'].lower() not in existing_names:
                    injuries[team].append(player)
    
    # Save to cache
    total = sum(len(v) for v in injuries.values())
    if total > 0:
        try:
            os.makedirs(os.path.dirname(INJURY_CACHE_FILE), exist_ok=True)
            with open(INJURY_CACHE_FILE, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'injuries': injuries
                }, f, indent=2)
            print(f"[INFO] Cached {total} injuries")
        except Exception as e:
            print(f"[WARNING] Could not save injury cache: {e}")
    
    return injuries


def get_team_injuries(team_abbrev: str) -> List[Dict]:
    """Get all injuries for a specific team."""
    injuries = get_injuries()
    return injuries.get(team_abbrev.upper(), [])


def get_out_players(team_abbrev: str) -> List[str]:
    """Get list of players who are OUT for a team."""
    team_injuries = get_team_injuries(team_abbrev)
    return [p['name'] for p in team_injuries if p['status'] == 'Out']


def get_questionable_players(team_abbrev: str) -> List[str]:
    """Get list of players who are Questionable/Doubtful/Day-To-Day."""
    team_injuries = get_team_injuries(team_abbrev)
    return [p['name'] for p in team_injuries 
            if p['status'] in ['Questionable', 'Doubtful', 'Day-To-Day']]


def is_player_out(player_name: str, team_abbrev: str = None) -> bool:
    """Check if a specific player is OUT."""
    injuries = get_injuries()
    
    player_name_lower = player_name.lower()
    
    teams_to_check = [team_abbrev] if team_abbrev else injuries.keys()
    
    for team in teams_to_check:
        if team not in injuries:
            continue
        for player in injuries[team]:
            if player_name_lower in player['name'].lower() or player['name'].lower() in player_name_lower:
                return player['status'] == 'Out'
    
    return False


def get_player_injury_status(player_name: str) -> Optional[Dict]:
    """
    Get injury status for a specific player.
    Returns dict with status info or None if not injured.
    """
    injuries = get_injuries()
    
    player_name_lower = player_name.lower()
    
    for team, team_injuries in injuries.items():
        for player in team_injuries:
            if player_name_lower in player['name'].lower() or player['name'].lower() in player_name_lower:
                return player
    
    return None


# Star player impact ratings (how much their absence affects teammates)
STAR_PLAYER_USAGE = {
    # Format: 'Player Name': (scoring_usage, assist_usage)
    # Higher = more impact when they're out
    'LeBron James': (0.28, 0.35),
    'Stephen Curry': (0.30, 0.20),
    'Kevin Durant': (0.28, 0.15),
    'Giannis Antetokounmpo': (0.32, 0.25),
    'Luka Doncic': (0.30, 0.35),
    'Jayson Tatum': (0.28, 0.20),
    'Joel Embiid': (0.32, 0.15),
    'Nikola Jokic': (0.25, 0.40),
    'Shai Gilgeous-Alexander': (0.30, 0.20),
    'Anthony Edwards': (0.28, 0.15),
    'Donovan Mitchell': (0.28, 0.15),
    'Damian Lillard': (0.30, 0.25),
    'Ja Morant': (0.28, 0.30),
    'Trae Young': (0.26, 0.40),
    'Devin Booker': (0.28, 0.20),
    'Kyrie Irving': (0.28, 0.25),
    'James Harden': (0.26, 0.35),
    'Jimmy Butler': (0.24, 0.25),
    'Paul George': (0.24, 0.18),
    'Kawhi Leonard': (0.28, 0.15),
    'Anthony Davis': (0.28, 0.12),
    'Tyrese Haliburton': (0.22, 0.40),
    'Jalen Brunson': (0.26, 0.28),
    'De\'Aaron Fox': (0.28, 0.25),
    'Tyler Herro': (0.25, 0.20),
    'Zion Williamson': (0.30, 0.15),
    'Paolo Banchero': (0.28, 0.18),
    'Victor Wembanyama': (0.26, 0.15),
    'Cade Cunningham': (0.25, 0.30),
    'Tyrese Maxey': (0.26, 0.18),
    'LaMelo Ball': (0.26, 0.30),
    'Brandon Ingram': (0.26, 0.18),
    'Desmond Bane': (0.24, 0.15),
    'Fred VanVleet': (0.22, 0.30),
    'Darius Garland': (0.24, 0.32),
    'Jalen Green': (0.28, 0.12),
    'Franz Wagner': (0.26, 0.18),
    'Scottie Barnes': (0.24, 0.25),
    'Alperen Sengun': (0.24, 0.25),
}


def calculate_teammate_out_boost(
    player_name: str,
    player_team: str,
    stat_type: str = 'points'
) -> float:
    """
    Calculate scoring/usage boost when a star teammate is OUT.
    
    Returns multiplier (e.g., 1.08 = 8% boost expected)
    """
    out_players = get_out_players(player_team)
    
    if not out_players:
        return 1.0  # No boost
    
    total_boost = 0.0
    
    for out_player in out_players:
        # Check if this is a star player
        for star_name, (scoring_usage, assist_usage) in STAR_PLAYER_USAGE.items():
            if star_name.lower() in out_player.lower() or out_player.lower() in star_name.lower():
                # Don't boost if the player being predicted is the one who's out
                if star_name.lower() in player_name.lower():
                    continue
                
                # Calculate boost based on stat type
                if stat_type in ['points', 'pra']:
                    # Scoring gets redistributed to remaining players
                    # Rough estimate: star's usage * 0.3 gets spread to other starters
                    total_boost += scoring_usage * 0.25
                elif stat_type == 'rebounds':
                    # Rebounds might increase if a big man is out
                    total_boost += 0.05
                elif stat_type == 'assists':
                    # If primary ball handler is out, another player may get more assists
                    total_boost += assist_usage * 0.15
                
                break
    
    # Cap the boost at 20%
    boost_multiplier = 1.0 + min(total_boost, 0.20)
    return boost_multiplier


def calculate_opponent_injury_boost(
    opponent_team: str,
    stat_type: str = 'points'
) -> float:
    """
    Calculate boost when opponent has key players OUT.
    
    Returns multiplier (e.g., 1.05 = 5% easier matchup)
    """
    out_players = get_out_players(opponent_team)
    
    if not out_players:
        return 1.0  # No boost
    
    total_boost = 0.0
    
    # Check for defensive anchors being out
    defensive_anchors = [
        'Rudy Gobert', 'Bam Adebayo', 'Jaren Jackson', 'Anthony Davis',
        'Giannis Antetokounmpo', 'Draymond Green', 'Herb Jones', 'Marcus Smart',
        'Mikal Bridges', 'OG Anunoby', 'Alex Caruso', 'Jrue Holiday',
        'Evan Mobley', 'Chet Holmgren', 'Victor Wembanyama', 'Brook Lopez',
        'Robert Williams', 'Myles Turner', 'Walker Kessler', 'Jakob Poeltl'
    ]
    
    for out_player in out_players:
        for defender in defensive_anchors:
            if defender.lower() in out_player.lower() or out_player.lower() in defender.lower():
                if stat_type in ['points', 'pra']:
                    total_boost += 0.06  # 6% easier to score
                break
        
        # Also check if any star is out (less disciplined defense overall)
        for star_name in STAR_PLAYER_USAGE.keys():
            if star_name.lower() in out_player.lower():
                total_boost += 0.03  # General team disarray
                break
    
    # Cap at 15%
    boost_multiplier = 1.0 + min(total_boost, 0.15)
    return boost_multiplier


def get_injury_features(
    player_name: str,
    player_team: str,
    opponent_team: str,
    stat_type: str = 'points'
) -> Dict:
    """
    Get all injury-related features for a prediction.
    
    Returns dict with:
    - teammate_out_boost: Multiplier for teammate absences
    - opponent_injury_boost: Multiplier for opponent injuries
    - combined_boost: Total multiplier
    - out_teammates: List of OUT teammates
    - out_opponents: List of OUT opponents
    - player_is_out: Whether the player themselves is OUT
    - player_status: Player's injury status if any
    """
    # Check if player is injured
    player_status = get_player_injury_status(player_name)
    player_is_out = player_status['status'] == 'Out' if player_status else False
    
    # Get boosts
    teammate_boost = calculate_teammate_out_boost(player_name, player_team, stat_type)
    opponent_boost = calculate_opponent_injury_boost(opponent_team, stat_type)
    
    # Combined boost
    combined_boost = teammate_boost * opponent_boost
    
    return {
        'teammate_out_boost': teammate_boost,
        'opponent_injury_boost': opponent_boost,
        'combined_injury_boost': combined_boost,
        'out_teammates': get_out_players(player_team),
        'questionable_teammates': get_questionable_players(player_team),
        'out_opponents': get_out_players(opponent_team),
        'player_is_out': player_is_out,
        'player_status': player_status
    }


def format_injury_report(team_abbrev: str) -> str:
    """Format injury report for display."""
    injuries = get_team_injuries(team_abbrev)
    
    if not injuries:
        return f"No injuries reported for {team_abbrev}"
    
    report = f"**{team_abbrev} Injury Report:**\n"
    
    out_players = [p for p in injuries if p['status'] == 'Out']
    questionable = [p for p in injuries if p['status'] in ['Questionable', 'Doubtful', 'Day-To-Day']]
    
    if out_players:
        report += "🔴 OUT: " + ", ".join([p['name'] for p in out_players]) + "\n"
    
    if questionable:
        report += "🟡 GTD: " + ", ".join([p['name'] for p in questionable]) + "\n"
    
    return report


# CLI test
if __name__ == "__main__":
    print("Testing injury tracker...\n")
    
    # Force refresh to test fetching
    injuries = get_injuries(force_refresh=True)
    
    # Print summary
    total_injuries = sum(len(v) for v in injuries.values())
    print(f"Total injuries found: {total_injuries}\n")
    
    # Print each team's injuries
    for team in sorted(injuries.keys()):
        if injuries[team]:
            print(f"\n{team}:")
            for player in injuries[team]:
                print(f"  - {player['name']} ({player['position']}): {player['status']}")
    
    # Test injury features
    print("\n" + "="*50)
    print("Testing injury features for Tyrese Maxey vs GSW:")
    features = get_injury_features(
        player_name="Tyrese Maxey",
        player_team="PHI",
        opponent_team="GSW",
        stat_type="points"
    )
    print(f"  Teammate boost: {features['teammate_out_boost']:.2f}x")
    print(f"  Opponent boost: {features['opponent_injury_boost']:.2f}x")
    print(f"  Combined: {features['combined_injury_boost']:.2f}x")
    print(f"  OUT teammates: {features['out_teammates']}")
    print(f"  OUT opponents: {features['out_opponents']}")
