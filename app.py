"""
NBA 1st Half Prediction Model - Enhanced Streamlit App
Features:
- Single & Batch Predictions
- Player Comparison
- Today's Games Integration
- Export to CSV
- Prediction History
- Confidence Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from functools import lru_cache
import json
import os

# Must be first Streamlit command
st.set_page_config(
    page_title="Prizepicks NBA 1H & 1Q Prop Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Lazy Imports & Caching
# ============================================================================

@st.cache_resource
def load_prediction_module():
    from model import make_prediction
    return make_prediction

@st.cache_resource
def load_data_collector():
    from data_collector import (
        search_player, get_all_teams, get_team_by_abbreviation,
        get_player_names_for_autocomplete, search_players_fuzzy,
        get_todays_games, get_team_players
    )
    return {
        'search_player': search_player,
        'get_all_teams': get_all_teams,
        'get_team_by_abbreviation': get_team_by_abbreviation,
        'get_player_names': get_player_names_for_autocomplete,
        'search_players_fuzzy': search_players_fuzzy,
        'get_todays_games': get_todays_games,
        'get_team_players': get_team_players
    }

@st.cache_data(ttl=3600)
def get_team_list():
    from config import TEAM_ABBREVIATIONS
    teams = list(TEAM_ABBREVIATIONS.values())
    teams.sort()
    return teams

@st.cache_data(ttl=3600)
def get_first_half_ratios():
    from config import FIRST_HALF_RATIOS
    return FIRST_HALF_RATIOS

@st.cache_data(ttl=3600)
def get_lock_thresholds(period: str = '1h'):
    from config import LOCK_THRESHOLDS, LOCK_THRESHOLDS_1Q
    if period == '1q':
        return LOCK_THRESHOLDS_1Q
    return LOCK_THRESHOLDS

@st.cache_data(ttl=86400)
def get_all_player_names():
    """Get all player names for autocomplete"""
    dc = load_data_collector()
    return dc['get_player_names']()

@st.cache_data(ttl=300, show_spinner=False)
def cached_prediction(player_name: str, opponent: str, prop_line: float, prop_type: str, is_home: bool, period: str = '1h'):
    make_prediction = load_prediction_module()
    return make_prediction(
        player_name=player_name,
        opponent=opponent,
        prop_line=prop_line,
        prop_type=prop_type,
        is_home=is_home,
        period=period
    )

# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []

# ============================================================================
# CSS Styles
# ============================================================================

CSS_STYLES = """
<style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .over-pick {
        background-color: #1a472a;
        border: 2px solid #2ecc71;
    }
    .under-pick {
        background-color: #4a1a1a;
        border: 2px solid #e74c3c;
    }
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f1c40f; font-weight: bold; }
    .confidence-low { color: #95a5a6; font-weight: bold; }
    .stat-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .gauge-container {
        width: 100%;
        height: 20px;
        background: linear-gradient(to right, #e74c3c, #f1c40f, #2ecc71);
        border-radius: 10px;
        position: relative;
        margin: 10px 0;
    }
    .gauge-marker {
        width: 4px;
        height: 30px;
        background: white;
        position: absolute;
        top: -5px;
        border-radius: 2px;
    }
    .player-card {
        background: #262730;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
    }
    .comparison-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e1e2e 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
"""

# ============================================================================
# Main App
# ============================================================================

def main():
    init_session_state()
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # Sidebar with injury info
    with st.sidebar:
        st.header("🏥 Injury Report")
        
        try:
            from injury_tracker import get_injuries, get_out_players
            
            injuries = get_injuries()
            total_out = sum(len([p for p in team_injuries if p['status'] == 'Out']) 
                          for team_injuries in injuries.values())
            
            st.metric("Players OUT", total_out)
            
            # Show teams with significant injuries
            with st.expander("View by Team", expanded=False):
                teams_with_injuries = [(t, i) for t, i in injuries.items() if i]
                teams_with_injuries.sort(key=lambda x: len([p for p in x[1] if p['status'] == 'Out']), reverse=True)
                
                for team, team_injuries in teams_with_injuries[:15]:
                    out_players = [p['name'] for p in team_injuries if p['status'] == 'Out']
                    gtd_players = [p['name'] for p in team_injuries if p['status'] in ['Day-To-Day', 'Questionable', 'Doubtful']]
                    
                    if out_players or gtd_players:
                        st.markdown(f"**{team}**")
                        if out_players:
                            st.markdown(f"🔴 {', '.join(out_players[:3])}")
                        if gtd_players:
                            st.markdown(f"🟡 {', '.join(gtd_players[:3])}")
                        st.divider()
            
            # Refresh button
            if st.button("🔄 Refresh Injuries"):
                from injury_tracker import get_injuries
                get_injuries(force_refresh=True)
                st.rerun()
                
        except ImportError:
            st.info("Injury tracking not available")
        except Exception as e:
            st.error(f"Could not load injuries: {str(e)}")
        
        st.divider()
        st.caption("Injuries affect predictions automatically")
    
    # Header
    st.title("🏀 Prizepicks NBA 1H & 1Q Prop Predictor")
    st.markdown("*Ensemble ML Model for 1st Half & 1st Quarter Props*")
    
    # Accuracy recommendation banner
    st.warning("""
    ⚡ **THE RULE: EDGE + CONFIDENCE TOGETHER**
    
    1. **Select the correct PERIOD** (1H = First Half, 1Q = First Quarter) that matches your prop
    2. **Look for 🎯 ELITE or ✅ SOLID picks** - these have BOTH high score AND big edge
    3. **Avoid ⚠️ THIN EDGE picks** - high score but small edge (<2 pts) can flip easily
    4. For Flex plays, only use picks with **Score 85+ AND Edge 2.5+**
    5. **⭐ NEVER take UNDER on star/volatile players** unless edge is 3.5+ pts
    6. **🔄 Watch for recently traded players** - old team stats may not apply
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Prediction",
        "📊 Batch Predictions", 
        "⚖️ Compare Players",
        "📅 Today's Games",
        "📜 History & Export"
    ])
    
    with tab1:
        single_prediction_tab()
    
    with tab2:
        batch_prediction_tab()
    
    with tab3:
        comparison_tab()
    
    with tab4:
        todays_games_tab()
    
    with tab5:
        history_tab()
    
    show_footer()


# ============================================================================
# Tab 1: Single Prediction
# ============================================================================

def single_prediction_tab():
    """Single player prediction interface"""
    teams = get_team_list()
    player_names = get_all_player_names()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Enter Prop Details")
        
        # Player autocomplete
        player_name = st.selectbox(
            "Player Name",
            options=[""] + player_names,
            index=0,
            help="Start typing to search for a player",
            key="single_player"
        )
        
        # If no selection, allow manual input
        if not player_name:
            player_name = st.text_input(
                "Or type player name",
                placeholder="e.g., LeBron James",
                key="single_player_manual"
            )
        
        # Period selection (1H or 1Q) - VERY CLEAR
        st.markdown("---")
        st.markdown("**⏱️ Which period is this prop for?**")
        period = st.radio(
            "Period",
            options=["1h", "1q"],
            format_func=lambda x: "🏀 1st HALF (1H) - Halftime stats" if x == "1h" else "⏱️ 1st QUARTER (1Q) - End of Q1 stats",
            horizontal=False,
            key="single_period",
            help="1H = First Half stats (through halftime). 1Q = First Quarter stats only (first 12 minutes)."
        )
        
        # Show clear explanation of what they selected
        if period == "1h":
            st.success("✅ **1st HALF** - Predicting stats at HALFTIME")
        else:
            st.warning("⏱️ **1st QUARTER** - Predicting stats after Q1 ONLY (higher variance)")
        
        st.markdown("---")
        
        # PrizePicks availability:
        # 1Q: Points, Rebounds, Assists only (no PRA)
        # 1H: Points, PRA only (no individual REB/AST)
        if period == "1q":
            prop_options = ["points", "rebounds", "assists"]
            prop_help = "1Q supports: Points, Rebounds, Assists (no PRA)"
        else:
            prop_options = ["points", "pra"]
            prop_help = "1H supports: Points, PRA only (no individual REB/AST)"
        
        prop_type = st.selectbox(
            "Stat Type",
            options=prop_options,
            format_func=lambda x: {
                "points": "Points",
                "rebounds": "Rebounds", 
                "assists": "Assists",
                "pra": "PRA (Pts + Reb + Ast)"
            }.get(x, x),
            key="single_prop_type",
            help=prop_help
        )
        
        # Show what they're predicting
        period_label = "1st HALF" if period == "1h" else "1st QUARTER"
        stat_label = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "pra": "PRA"}.get(prop_type, prop_type)
        st.info(f"📊 Predicting: **{period_label} {stat_label}**")
        
        # Adjust default line based on BOTH period AND prop type
        # 1Q: Points, Rebounds, Assists (no PRA)
        # 1H: Points, PRA (no individual REB/AST)
        default_lines = {
            # 1Q defaults (no PRA)
            ("1q", "points"): 6.5,
            ("1q", "rebounds"): 2.5,
            ("1q", "assists"): 2.0,
            # 1H defaults (no individual REB/AST)
            ("1h", "points"): 12.5,
            ("1h", "pra"): 20.5,
        }
        default_line = default_lines.get((period, prop_type), 6.5)
        
        # Typical ranges for help text
        typical_ranges = {
            "points": "1Q: 4-10, 1H: 10-20",
            "rebounds": "1Q: 1.5-4, 1H: 3-8",
            "assists": "1Q: 1-3.5, 1H: 2-7",
            "pra": "1Q: 8-15, 1H: 15-30"
        }
        
        prop_line = st.number_input(
            f"PrizePicks Line (for {period_label} {stat_label})",
            min_value=0.5,
            max_value=50.0,
            value=default_line,
            step=0.5,
            key="single_prop_line",
            help=f"Typical {stat_label} lines: {typical_ranges.get(prop_type, '5-15')}"
        )
        
        st.divider()
        
        opponent = st.selectbox(
            "Opponent Team",
            options=teams,
            index=teams.index("LAL") if "LAL" in teams else 0,
            key="single_opponent"
        )
        
        is_home = st.radio(
            "Location",
            options=[True, False],
            format_func=lambda x: "🏠 Home" if x else "✈️ Away",
            horizontal=True,
            key="single_location"
        )
        
        predict_btn = st.button("🎯 Get Prediction", type="primary", width="stretch", key="single_predict")
    
    with col2:
        if predict_btn and player_name:
            with st.spinner(f"Analyzing {player_name}..."):
                result = cached_prediction(
                    player_name=player_name.strip(),
                    opponent=opponent,
                    prop_line=prop_line,
                    prop_type=prop_type,
                    is_home=is_home,
                    period=period
                )
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Add to history
                    add_to_history(result, prop_type, opponent, is_home, period)
                    display_prediction(result, prop_type, prop_line, period)
        elif predict_btn:
            st.warning("⚠️ Please enter a player name")
        else:
            show_instructions()


# ============================================================================
# Tab 2: Batch Predictions
# ============================================================================

def parse_prizepicks_paste(raw_text: str) -> list:
    """
    Parse raw PrizePicks copy-paste format into structured data.
    
    PrizePicks multiline format:
    ```
    Jalen Johnson          ← Player name (may have Goblin/Demon suffix)
    ATL - F                ← Team - Position
    Jalen Johnson          ← Player name repeated (sometimes)
    vs IND 1Q Mon 10:40am  ← Matchup: vs/@ OPPONENT + time
    6                      ← Prop line (number)
    1Q Points              ← Prop type
    Less
    More
    ```
    
    Returns list of dicts: [{player, opponent, is_home, prop_line, prop_type}, ...]
    """
    import re
    
    # All NBA team abbreviations
    NBA_TEAMS = {'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
                 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
                 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'}
    
    # Words to skip entirely
    SKIP_WORDS = {'less', 'more', 'trending', 'swap', 'combos',
                  # All prop type variations
                  'points', 'rebounds', 'assists', 'pra',
                  '1q points', '1q rebounds', '1q assists', '1q pra',
                  '1h points', '1h rebounds', '1h assists', '1h pra',
                  '2q points', '2q rebounds', '2q assists',
                  '3q points', '3q rebounds', '3q assists',
                  '4q points', '4q rebounds', '4q assists'}
    
    parsed = []
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        lower_line = line.lower()
        
        # Skip noise lines
        if lower_line in SKIP_WORDS:
            i += 1
            continue
        
        # Skip lines that are just numbers with K/M suffix (like "1.2K", "565")
        if re.match(r'^[\d.,]+[kKmM]?$', line):
            i += 1
            continue
        
        # Skip "TEAM - POSITION" lines (like "ATL - F", "IND - G-F")
        if re.match(r'^[A-Z]{2,3}\s*-\s*[A-Z\-]+$', line):
            i += 1
            continue
        
        # Skip matchup lines that start with vs or @
        if lower_line.startswith('vs ') or lower_line.startswith('@ ') or line.startswith('@'):
            i += 1
            continue
        
        # Skip lines that are prop type indicators (e.g., "1Q Points", "1Q Rebounds")
        if re.match(r'^[1-4][qQhH]\s*(points|rebounds|assists|pra)$', lower_line, re.IGNORECASE):
            i += 1
            continue
        
        # Skip time-related lines (e.g., "55m 59s", "Mon 6:40pm")
        if re.match(r'^\d+[mhd]\s*\d*[smh]?$', lower_line) or re.search(r'\d+:\d+\s*(am|pm)', lower_line):
            i += 1
            continue
        
        # Check if this is a player name line
        # Player names: have letters, don't start with @/vs, aren't team-position format
        has_letters = bool(re.search(r'[a-zA-Z]{3,}', line))
        is_not_team_pos = not re.match(r'^[A-Z]{2,3}\s*-', line)
        is_not_number = not re.match(r'^[\d.]+$', line)
        
        if has_letters and is_not_team_pos and is_not_number:
            # This looks like a player name
            player_name = line
            
            # Remove Goblin/Demon suffixes
            player_name = re.sub(r'(Goblin|Demon)$', '', player_name, flags=re.IGNORECASE).strip()
            
            # Skip if too short or is just a team abbrev
            if len(player_name) < 3 or player_name.upper() in NBA_TEAMS:
                i += 1
                continue
            
            # Skip if it's a noise word
            if player_name.lower() in SKIP_WORDS or player_name.lower().startswith('trending'):
                i += 1
                continue
            
            # Look ahead to find opponent, prop line, and prop type
            opponent = None
            is_home = True
            prop_line = None
            prop_type = 'points'  # Default to points
            
            # Search next 10 lines for matchup, prop line, and prop type
            for j in range(i + 1, min(i + 12, len(lines))):
                next_line = lines[j].strip()
                next_upper = next_line.upper()
                next_lower = next_line.lower()
                
                # Check for "vs TEAM" (home game) - e.g., "vs IND 1Q Mon 10:40am"
                vs_match = re.search(r'\bVS\.?\s*([A-Z]{2,3})\b', next_upper)
                if vs_match and vs_match.group(1) in NBA_TEAMS:
                    opponent = vs_match.group(1)
                    is_home = True
                
                # Check for "@ TEAM" (away game) - e.g., "@ ATL 1Q Mon 10:40am"  
                at_match = re.search(r'@\s*([A-Z]{2,3})\b', next_upper)
                if at_match and at_match.group(1) in NBA_TEAMS:
                    opponent = at_match.group(1)
                    is_home = False
                
                # Check for prop line (just a number like "6", "5.5", "11.5")
                if re.match(r'^[\d.]+$', next_line):
                    try:
                        val = float(next_line)
                        if 0.5 <= val <= 60:  # Reasonable prop range
                            prop_line = val
                    except:
                        pass
                
                # Detect prop type from lines like "1Q Points", "1H Rebounds", "PRA", etc.
                if 'pra' in next_lower or 'pts+rebs+asts' in next_lower or 'pts + rebs + asts' in next_lower:
                    prop_type = 'pra'
                elif 'rebound' in next_lower:
                    prop_type = 'rebounds'
                elif 'assist' in next_lower:
                    prop_type = 'assists'
                elif 'point' in next_lower:
                    prop_type = 'points'
                
                # Stop if we've found prop type line (usually ends the player block)
                if any(x in next_lower for x in ['points', 'rebounds', 'assists', 'pra']):
                    if prop_line is not None:
                        break
                
                # If we hit another player name and we already have a prop line, stop
                if j > i + 3 and prop_line is not None:
                    if (re.search(r'[a-zA-Z]{3,}', next_line) and 
                        not re.match(r'^[A-Z]{2,3}\s*-', next_line) and
                        not re.match(r'^[\d.]+', next_line) and
                        next_lower not in SKIP_WORDS and
                        not next_lower.startswith('vs ') and
                        not next_line.startswith('@')):
                        break
            
            # Only add if we found a prop line
            if prop_line is not None:
                parsed.append({
                    'player': player_name,
                    'opponent': opponent if opponent else 'UNK',
                    'is_home': is_home,
                    'prop_line': prop_line,
                    'prop_type': prop_type
                })
        
        i += 1
    
    # Remove duplicates - keep first occurrence
    seen = set()
    unique_parsed = []
    for p in parsed:
        # Normalize player name for comparison
        normalized_name = re.sub(r'[^a-z\s]', '', p['player'].lower()).strip()
        normalized_name = ' '.join(normalized_name.split())
        key = (normalized_name, p['prop_line'])
        if key not in seen:
            seen.add(key)
            unique_parsed.append(p)
    
    return unique_parsed


def batch_prediction_tab():
    """Batch prediction for multiple players"""
    st.subheader("📊 Batch Predictions")
    st.markdown("*Analyze multiple players at once*")
    
    teams = get_team_list()
    
    # Period selection for batch - VERY CLEAR
    st.markdown("**⏱️ What period are these props for?**")
    batch_period = st.radio(
        "Period for ALL props below",
        options=["1h", "1q"],
        format_func=lambda x: "🏀 1st HALF (1H)" if x == "1h" else "⏱️ 1st QUARTER (1Q)",
        horizontal=True,
        key="batch_period"
    )
    
    # Show clear confirmation
    if batch_period == "1h":
        st.success("✅ All predictions below will be for **1st HALF** stats (through halftime)")
    else:
        st.warning("⏱️ All predictions below will be for **1st QUARTER** stats only (first 12 min)")
    
    # Supported prop types - DIFFERENT for 1Q vs 1H based on PrizePicks
    period_label = "1Q" if batch_period == "1q" else "1H"
    with st.expander(f"📋 Supported {period_label} Prop Types (click to expand)"):
        if batch_period == "1q":
            # 1Q supports individual props only (no PRA)
            st.success("✅ **1Q supports:** Points, Rebounds, Assists (no PRA)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **1Q Points** 
                - Typical lines: 4-10
                - Parser detects: "1Q Points"
                
                **1Q Rebounds**
                - Typical lines: 1.5-4
                - Parser detects: "1Q Rebounds"
                """)
            with col2:
                st.markdown("""
                **1Q Assists**
                - Typical lines: 1-3.5
                - Parser detects: "1Q Assists"
                """)
        else:
            # 1H only supports Points and PRA
            st.warning("⚠️ **1H supports:** Points, PRA only (no individual REB/AST on PrizePicks)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **1H Points** 
                - Typical lines: 10-20
                - Parser detects: "1H Points", "Points"
                """)
            with col2:
                st.markdown("""
                **1H PRA** (Pts+Reb+Ast)
                - Typical lines: 15-30
                - Parser detects: "1H PRA", "PRA"
                """)
    
    st.markdown("---")
    
    # Input mode selection
    input_mode = st.radio(
        "Input Method",
        options=["paste", "manual"],
        format_func=lambda x: "📋 Paste from PrizePicks" if x == "paste" else "✏️ Manual Entry",
        horizontal=True,
        key="batch_input_mode"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if input_mode == "paste":
            # PrizePicks paste mode
            st.markdown("**Paste Props from PrizePicks**")
            st.caption("Copy directly from PrizePicks and paste below")
            
            paste_input = st.text_area(
                "Paste Props",
                placeholder="""Karl-Anthony Towns
@ PHI Sat 12:10pm
19.5
Points

OG Anunoby
@ PHI Sat 12:10pm
15.5
Points""",
                height=250,
                key="paste_input",
                label_visibility="collapsed"
            )
            
            # Parse and preview
            if paste_input:
                parsed = parse_prizepicks_paste(paste_input)
                if parsed:
                    unknown_opps = [p for p in parsed if p['opponent'] == 'UNK']
                    if unknown_opps:
                        st.warning(f"⚠️ Parsed {len(parsed)} props ({len(unknown_opps)} with unknown opponent)")
                    else:
                        st.success(f"✅ Parsed {len(parsed)} props - all opponents detected!")
                    
                    with st.expander("Preview parsed data", expanded=True):
                        for p in parsed:
                            loc = "🏠 Home" if p['is_home'] else "✈️ Away"
                            opp_display = p['opponent'] if p['opponent'] != 'UNK' else "❓ Unknown"
                            st.markdown(f"**{p['player']}** | {loc} vs **{opp_display}** | {p['prop_line']} {p['prop_type']}")
                    
                    # If any unknown opponents, let user set a default
                    if unknown_opps:
                        st.markdown("---")
                        default_opp = st.selectbox(
                            "Default opponent for unknown matchups:",
                            options=get_team_list(),
                            key="default_opponent_paste"
                        )
                else:
                    st.warning("⚠️ Could not parse any props. Check format.")
                    # Debug: show first few lines to help diagnose
                    with st.expander("🔍 Debug: Show raw input lines"):
                        lines = paste_input.strip().split('\n')[:10]
                        for i, line in enumerate(lines):
                            has_pipe = '|' in line
                            st.code(f"Line {i+1} (has pipe: {has_pipe}): {repr(line)}")
            
            paste_btn = st.button("🎯 Analyze Pasted Props", type="primary", width="stretch", key="paste_analyze")
            
            if paste_btn and paste_input:
                parsed = parse_prizepicks_paste(paste_input)
                if parsed:
                    # Get default opponent if set
                    default_opp = st.session_state.get('default_opponent_paste', 'LAL')
                    results = process_parsed_props(parsed, default_opp, batch_period)
                    if results:
                        st.session_state.batch_results = results
        
        else:
            # Manual entry mode (original)
            st.markdown("**Enter Players (one per line)**")
            st.caption("Format: Player Name, Line, Type (points/pra)")
            
            batch_input = st.text_area(
                "Players",
                placeholder="LeBron James, 12.5, points\nStephen Curry, 14.5, points\nKevin Durant, 24.5, pra",
                height=200,
                key="batch_input",
                label_visibility="collapsed"
            )
            
            st.divider()
            
            # Common settings
            common_opponent = st.selectbox(
                "Common Opponent (optional)",
                options=[""] + teams,
                key="batch_opponent"
            )
            
            common_location = st.radio(
                "Common Location",
                options=[True, False],
                format_func=lambda x: "🏠 Home" if x else "✈️ Away",
                horizontal=True,
                key="batch_location"
            )
            
            batch_btn = st.button("📊 Run Batch Analysis", type="primary", width="stretch")
            
            if batch_btn and batch_input:
                results = process_batch_predictions(batch_input, common_opponent, common_location, batch_period)
                if results:
                    st.session_state.batch_results = results
    
    with col2:
        # Display results
        if st.session_state.batch_results:
            display_batch_results(st.session_state.batch_results)


def process_parsed_props(parsed_props: list, default_opponent: str = 'LAL', period: str = '1h') -> list:
    """Process parsed PrizePicks props"""
    results = []
    processed_keys = set()  # Track already processed to avoid duplicates
    skipped_props = []  # Track props skipped due to period restrictions
    
    progress = st.progress(0)
    
    for i, prop in enumerate(parsed_props):
        # Skip if we already processed this player/line/type combo
        prop_key = (prop['player'], prop['prop_line'], prop['prop_type'])
        if prop_key in processed_keys:
            continue
        processed_keys.add(prop_key)
        player_name = prop['player']
        prop_line = prop['prop_line']
        prop_type = prop['prop_type']
        opponent = prop['opponent']
        is_home = prop['is_home']
        
        # PrizePicks period restrictions:
        # 1Q: points, rebounds, assists (no PRA)
        # 1H: points, pra only (no individual REB/AST)
        valid_1h_types = ['points', 'pra']
        valid_1q_types = ['points', 'rebounds', 'assists']
        
        if period == '1h' and prop_type in ['rebounds', 'assists']:
            skipped_props.append(f"{player_name} ({prop_type})")
            continue  # Skip invalid 1H prop types
        
        if period == '1q' and prop_type == 'pra':
            skipped_props.append(f"{player_name} (pra)")
            continue  # Skip PRA for 1Q
        
        # Use default opponent if unknown
        actual_opponent = opponent if opponent != 'UNK' else default_opponent
        
        # Validate prop type
        valid_types = valid_1q_types if period == '1q' else valid_1h_types
        actual_prop_type = prop_type if prop_type in valid_types else 'points'
        
        try:
            result = cached_prediction(
                player_name=player_name,
                opponent=actual_opponent,
                prop_line=prop_line,
                prop_type=actual_prop_type,
                is_home=is_home,
                period=period
            )
            
            if result:
                result['comparison_line'] = prop_line
                result['comparison_type'] = prop_type
                result['comparison_opponent'] = actual_opponent
                result['comparison_location'] = '🏠' if is_home else '✈️'
                results.append(result)
        except Exception as e:
            st.warning(f"⚠️ Error for {player_name}: {str(e)[:50]}")
        
        progress.progress((i + 1) / len(parsed_props))
    
    progress.empty()
    
    # Warn about skipped props due to period restrictions
    if skipped_props:
        if period == '1h':
            reason = "1H only supports Points & PRA (no individual REB/AST)"
        else:
            reason = "1Q only supports Points, Rebounds, Assists (no PRA)"
        st.warning(f"⚠️ **Skipped {len(skipped_props)} props** - {reason}:\n{', '.join(skipped_props[:5])}{'...' if len(skipped_props) > 5 else ''}")
    
    return results


def process_batch_predictions(batch_input: str, opponent: str, is_home: bool, period: str = '1h') -> list:
    """Process batch prediction input"""
    results = []
    processed_keys = set()  # Track already processed to avoid duplicates
    lines = batch_input.strip().split('\n')
    
    progress = st.progress(0)
    
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        
        player_name = parts[0]
        try:
            prop_line = float(parts[1])
        except:
            continue
        
        prop_type = parts[2].lower() if len(parts) > 2 else 'points'
        if prop_type not in ['points', 'pra']:
            prop_type = 'points'
        
        # Skip if we already processed this player/line/type combo
        prop_key = (player_name, prop_line, prop_type)
        if prop_key in processed_keys:
            continue
        processed_keys.add(prop_key)
        
        player_opponent = opponent if opponent else 'LAL'
        
        result = cached_prediction(
            player_name=player_name,
            opponent=player_opponent,
            prop_line=prop_line,
            prop_type=prop_type,
            is_home=is_home,
            period=period
        )
        
        result['input_opponent'] = player_opponent
        result['input_prop_type'] = prop_type
        result['input_prop_line'] = prop_line
        results.append(result)
        
        progress.progress((i + 1) / len(lines))
    
    progress.empty()
    return results


def display_batch_results(results: list):
    """Display batch prediction results with lock scores"""
    
    # Remove duplicates from results first
    seen = set()
    unique_results = []
    for r in results:
        if 'error' not in r:
            # Create unique key based on player, line, and type
            player = r.get('player_name', 'Unknown')
            prop_line = r.get('input_prop_line') or r.get('comparison_line') or r.get('prop_line', 0)
            prop_type = r.get('input_prop_type') or r.get('comparison_type') or 'points'
            key = (player, prop_line, prop_type)
            
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
    
    st.subheader(f"📊 Results ({len(unique_results)} players)")
    
    # Key rule reminder at the top
    st.warning("⚡ **THE RULE:** You want **EDGE + CONFIDENCE** together. High score with tiny edge = trap. Look for 🎯 ELITE or ✅ SOLID picks only!")
    
    # Summary table
    summary_data = []
    for r in unique_results:
        # Handle both input_* (manual batch) and comparison_* (paste) keys
        prop_type = r.get('input_prop_type') or r.get('comparison_type') or 'points'
        prop_line = r.get('input_prop_line') or r.get('comparison_line') or r.get('prop_line', 0)
        lock_score = r.get('lock_score', 0) or 0
        edge = r.get('difference', 0) or 0
        abs_edge = abs(edge)
        
        # Determine quality rating based on BOTH lock score AND edge
        if lock_score >= 85 and abs_edge >= 3.5:
            quality = "🎯 ELITE"
        elif lock_score >= 85 and abs_edge >= 2.0:
            quality = "✅ SOLID"
        elif lock_score >= 85 and abs_edge < 2.0:
            quality = "⚠️ THIN EDGE"
        elif lock_score >= 72 and abs_edge >= 2.5:
            quality = "👍 GOOD"
        elif lock_score >= 72:
            quality = "📊 OK"
        elif lock_score >= 55:
            quality = "🎲 RISKY"
        else:
            quality = "❌ SKIP"
        
        # Check for injury conflict (UNDER pick but injuries favor OVER)
        pick = r.get('pick', '')
        injury_boost = r.get('injury_boost', 1.0) or 1.0
        is_under = 'UNDER' in pick
        injury_conflict = is_under and injury_boost > 1.05
        
        # Add warning emoji to pick if injury conflict
        display_pick = pick
        if injury_conflict:
            display_pick = f"🏥 {pick}"
        
        summary_data.append({
            'Player': r.get('player_name', 'Unknown'),
            'Type': prop_type.upper() if isinstance(prop_type, str) else 'POINTS',
            'Line': prop_line,
            'Predicted': r.get('predicted_1h', 0) or 0,
            'Pick': display_pick,
            'Score': lock_score,
            'Edge': edge,
            'Quality': quality,
            'InjBoost': f"+{(injury_boost-1)*100:.0f}%" if injury_boost > 1.01 else ""
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Sort by Score descending (best picks first)
        df = df.sort_values('Score', ascending=False)
        
        # Format numbers cleanly
        df['Line'] = df['Line'].apply(lambda x: f"{float(x):.1f}" if x else "0.0")
        df['Predicted'] = df['Predicted'].apply(lambda x: f"{float(x):.1f}" if x else "0.0")
        df['Edge'] = df['Edge'].apply(lambda x: f"{float(x):+.1f}" if x else "0.0")
        
        # Color coding based on Quality rating
        def color_row(row):
            quality = row.get('Quality', '')
            if '🎯 ELITE' in quality:
                return ['background-color: #1a4a1a; color: #00ff00'] * len(row)  # Bright green
            elif '✅ SOLID' in quality:
                return ['background-color: #3d3d00; color: #FFD700'] * len(row)  # Gold
            elif '⚠️ THIN' in quality:
                return ['background-color: #4a3a00; color: #FFA500'] * len(row)  # Orange warning
            elif '👍 GOOD' in quality:
                return ['background-color: #1a472a; color: #2ecc71'] * len(row)  # Green
            elif '📊 OK' in quality:
                return ['background-color: #1a3a4a; color: #3498db'] * len(row)  # Blue
            elif '🎲 RISKY' in quality:
                return ['background-color: #4a3a1a; color: #f39c12'] * len(row)  # Orange
            else:
                return ['background-color: #4a1a1a; color: #e74c3c'] * len(row)  # Red
        
        try:
            styled = df.style.apply(color_row, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Check if any picks have injury conflicts
        has_injury_conflicts = any('🏥' in str(d.get('Pick', '')) for d in summary_data)
        
        if has_injury_conflicts:
            st.warning("🏥 **INJURY CONFLICT:** Picks marked with 🏥 are UNDER picks where injuries suggest higher scoring. These may be riskier than the score indicates.")
        
        # Quality guide
        st.markdown("---")
        st.markdown("**📋 Quality Guide (Score + Edge combined):**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - 🎯 **ELITE** = Score 85+ AND Edge 3.5+ → Best bets
            - ✅ **SOLID** = Score 85+ AND Edge 2-3.5 → Good bets
            - ⚠️ **THIN EDGE** = Score 85+ BUT Edge <2 → Risky!
            """)
        with col2:
            st.markdown("""
            - 👍 **GOOD** = Score 72+ AND Edge 2.5+ → Decent
            - 📊 **OK** = Score 72+ with smaller edge
            - 🎲 **RISKY** / ❌ **SKIP** = Avoid
            """)
        
        st.info("⚡ **The Rule:** Even with 100 Lock Score, if edge is <2 pts, one hot/cold quarter can flip it. You want **BOTH confidence + edge**.")
        
        # =====================================================================
        # PARLAY BUILDER SECTION - Only ELITE and SOLID picks
        # =====================================================================
        st.markdown("---")
        st.markdown("## 🎰 PARLAY BUILDER")
        st.markdown("*Only 🎯 ELITE and ✅ SOLID picks - use these for Flex plays*")
        
        # Filter for ELITE and SOLID only
        elite_picks = df[df['Quality'].str.contains('ELITE', na=False)]
        solid_picks = df[df['Quality'].str.contains('SOLID', na=False)]
        parlay_picks = pd.concat([elite_picks, solid_picks]).drop_duplicates()
        
        if len(parlay_picks) > 0:
            # Sort by edge size (absolute value)
            parlay_picks = parlay_picks.copy()
            parlay_picks['Abs_Edge'] = pd.to_numeric(parlay_picks['Edge'], errors='coerce').abs()
            parlay_picks = parlay_picks.sort_values('Abs_Edge', ascending=False)
            
            # Display in a clean format
            st.success(f"✅ **{len(parlay_picks)} picks ready for Flex plays!**")
            
            # Show as a clean table with proper formatting
            parlay_display = parlay_picks[['Player', 'Type', 'Line', 'Pick', 'Edge', 'Quality']].copy()
            
            # Format numbers cleanly (remove extra decimals)

            
            def color_parlay_row(row):
                if 'ELITE' in str(row.get('Quality', '')):
                    return ['background-color: #1a4a1a; color: #00ff00'] * len(row)
                else:
                    return ['background-color: #3d3d00; color: #FFD700'] * len(row)
            
            try:
                styled_parlay = parlay_display.style.apply(color_parlay_row, axis=1)
                st.dataframe(styled_parlay, use_container_width=True, hide_index=True)
            except:
                st.dataframe(parlay_display, use_container_width=True, hide_index=True)
            
            # Quick copy format for PrizePicks
            st.markdown("**📋 Quick Copy (for reference):**")
            parlay_text = ""
            for _, row in parlay_picks.iterrows():
                try:
                    edge_val = float(row['Edge'])
                except:
                    edge_val = 0
                pick_dir = "OVER" if edge_val > 0 else "UNDER"
                line_val = float(row['Line']) if isinstance(row['Line'], str) else row['Line']
                parlay_text += f"• {row['Player']} {pick_dir} {line_val:.1f} {row['Type']} ({row['Quality']})\n"
            
            st.code(parlay_text, language=None)
            
            # Suggested combos
            if len(parlay_picks) >= 4:
                st.markdown("**🎯 Suggested 4-Pick Flex:**")
                top_4 = parlay_picks.head(4)
                for _, row in top_4.iterrows():
                    try:
                        edge_val = float(row['Edge'])
                    except:
                        edge_val = 0
                    pick_dir = "OVER" if edge_val > 0 else "UNDER"
                    line_val = float(row['Line']) if isinstance(row['Line'], str) else row['Line']
                    edge_val = float(row['Edge']) if isinstance(row['Edge'], str) else row['Edge']
                    st.markdown(f"- **{row['Player']}** {pick_dir} {line_val:.1f} ({row['Quality']}, Edge: {edge_val:+.1f})")
            
            if len(parlay_picks) >= 6:
                st.markdown("**🔥 Suggested 6-Pick Flex:**")
                top_6 = parlay_picks.head(6)
                for _, row in top_6.iterrows():
                    try:
                        edge_val = float(row['Edge'])
                    except:
                        edge_val = 0
                    pick_dir = "OVER" if edge_val > 0 else "UNDER"
                    line_val = float(row['Line']) if isinstance(row['Line'], str) else row['Line']
                    edge_val = float(row['Edge']) if isinstance(row['Edge'], str) else row['Edge']
                    st.markdown(f"- **{row['Player']}** {pick_dir} {line_val:.1f} ({row['Quality']}, Edge: {edge_val:+.1f})")
        else:
            st.warning("⚠️ No ELITE or SOLID picks found. Consider waiting for better spots or lowering your standards to 👍 GOOD picks.")
        
        st.markdown("---")
        
        # Show remaining picks summary
        thresholds = get_lock_thresholds()
        locks = df[df['Score'] >= thresholds['lock']]
        strong = df[(df['Score'] >= thresholds['strong']) & (df['Score'] < thresholds['lock'])]
        
        if len(locks) > 0:
            st.success(f"🔒 **{len(locks)} LOCK(S) FOUND:** {', '.join(locks['Player'].tolist())}")
        if len(strong) > 0:
            st.info(f"🔥 **{len(strong)} Strong Play(s):** {', '.join(strong['Player'].tolist())}")
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export to CSV",
            csv,
            f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            key="batch_export"
        )


# ============================================================================
# Tab 3: Player Comparison
# ============================================================================

def comparison_tab():
    """Compare multiple players side by side"""
    st.subheader("⚖️ Compare Players")
    st.markdown("*Compare predictions for multiple players*")
    
    teams = get_team_list()
    player_names = get_all_player_names()
    
    # Player selection
    col1, col2, col3 = st.columns(3)
    
    players_to_compare = []
    
    with col1:
        p1 = st.selectbox("Player 1", [""] + player_names, key="cmp_p1")
        l1 = st.number_input("Line 1", 0.5, 50.0, 12.5, 0.5, key="cmp_l1")
        if p1:
            players_to_compare.append({'name': p1, 'line': l1})
    
    with col2:
        p2 = st.selectbox("Player 2", [""] + player_names, key="cmp_p2")
        l2 = st.number_input("Line 2", 0.5, 50.0, 12.5, 0.5, key="cmp_l2")
        if p2:
            players_to_compare.append({'name': p2, 'line': l2})
    
    with col3:
        p3 = st.selectbox("Player 3 (optional)", [""] + player_names, key="cmp_p3")
        l3 = st.number_input("Line 3", 0.5, 50.0, 12.5, 0.5, key="cmp_l3")
        if p3:
            players_to_compare.append({'name': p3, 'line': l3})
    
    st.divider()
    
    # Period selection - clear
    st.markdown("**⏱️ Comparison Settings**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cmp_period = st.radio("Period", ["1h", "1q"], format_func=lambda x: "🏀 1H" if x == "1h" else "⏱️ 1Q", horizontal=True, key="cmp_period")
    with col2:
        # Period-specific prop types (PrizePicks rules)
        # 1Q: Points, Rebounds, Assists (no PRA)
        # 1H: Points, PRA (no individual REB/AST)
        if cmp_period == "1q":
            cmp_prop_options = ["points", "rebounds", "assists"]
        else:
            cmp_prop_options = ["points", "pra"]
        prop_type = st.selectbox("Stat Type", cmp_prop_options, 
            format_func=lambda x: {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "pra": "PRA"}.get(x, x), key="cmp_type")
    with col3:
        opponent = st.selectbox("Opponent", teams, key="cmp_opp")
    with col4:
        is_home = st.radio("Location", [True, False], format_func=lambda x: "Home" if x else "Away", horizontal=True, key="cmp_loc")
    
    # Show what they're comparing
    period_label = "1st HALF" if cmp_period == "1h" else "1st QUARTER"
    stat_label = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "pra": "PRA"}.get(prop_type, prop_type)
    st.info(f"📊 Comparing: **{period_label} {stat_label}**")
    
    compare_btn = st.button("⚖️ Compare", type="primary", width="stretch")
    
    if compare_btn and len(players_to_compare) >= 2:
        with st.spinner("Comparing players..."):
            comparison_results = []
            for p in players_to_compare:
                result = cached_prediction(p['name'], opponent, p['line'], prop_type, is_home, cmp_period)
                result['comparison_line'] = p['line']
                result['period'] = cmp_period
                comparison_results.append(result)
            
            st.session_state.comparison_results = comparison_results
    
    # Display comparison
    if st.session_state.comparison_results:
        display_comparison(st.session_state.comparison_results, prop_type)


def display_comparison(results: list, prop_type: str):
    """Display player comparison with lock scores"""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        st.error("No valid results to compare")
        return
    
    st.markdown('<div class="comparison-header"><h3>📊 Comparison Results</h3></div>', unsafe_allow_html=True)
    
    cols = st.columns(len(valid_results))
    
    for i, (col, result) in enumerate(zip(cols, valid_results)):
        with col:
            pick = result.get('pick', 'N/A')
            lock_score = result.get('lock_score', 50)
            confidence = result.get('confidence', 'N/A')
            is_over = 'OVER' in pick
            
            # Color based on lock score - using config thresholds
            thresholds = get_lock_thresholds()
            if lock_score >= thresholds['lock']:
                color = "#FFD700"  # Gold
            elif lock_score >= thresholds['strong']:
                color = "#2ecc71"  # Green
            elif lock_score >= thresholds['playable']:
                color = "#3498db"  # Blue
            elif 'SKIP' in pick:
                color = "#95a5a6"  # Gray
            else:
                color = "#e74c3c"  # Red
            
            st.markdown(f"""
            <div class="player-card" style="border-left-color: {color}; border-left-width: 5px;">
                <h4 style="margin: 0;">{result.get('player_name', 'Unknown')}</h4>
                <p style="color: #888; margin: 5px 0;">Line: {result.get('comparison_line', 0)}</p>
                <h2 style="color: {color}; margin: 10px 0; font-size: 1.3em;">{pick}</h2>
                <div style="background: #1a1a2e; padding: 8px; border-radius: 8px; margin: 10px 0;">
                    <span style="color: {color}; font-weight: bold; font-size: 1.2em;">🎯 {lock_score}/100</span>
                </div>
                <p>Predicted: <strong>{result.get('predicted_1h', 0):.1f}</strong></p>
                <p>Edge: <strong>{result.get('difference', 0):+.1f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini lock score bar
            bar_color = color
            st.markdown(f"""
            <div style="
                width: 100%;
                height: 8px;
                background: #1a1a2e;
                border-radius: 4px;
                margin: 5px 0;
            ">
                <div style="
                    width: {lock_score}%;
                    height: 100%;
                    background: {bar_color};
                    border-radius: 4px;
                "></div>
            </div>
            """, unsafe_allow_html=True)


def display_confidence_gauge(confidence: str, difference: float):
    """Display a confidence gauge visualization"""
    # Map confidence to percentage
    conf_map = {'HIGH': 85, 'MEDIUM': 50, 'LOW': 20}
    conf_pct = conf_map.get(confidence, 20)
    
    # Adjust based on difference
    if abs(difference) > 3:
        conf_pct = min(95, conf_pct + 15)
    elif abs(difference) < 0.5:
        conf_pct = max(10, conf_pct - 15)
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div class="gauge-container">
            <div class="gauge-marker" style="left: {conf_pct}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #888;">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Tab 4: Today's Games
# ============================================================================

def todays_games_tab():
    """Show today's NBA games and quick predictions"""
    st.subheader("📅 Today's Games")
    st.markdown(f"*Games for {date.today().strftime('%B %d, %Y')}*")
    
    dc = load_data_collector()
    teams = get_team_list()
    
    # Try to fetch today's games
    games = []
    with st.spinner("Loading today's schedule..."):
        try:
            games = dc['get_todays_games']()
        except Exception as e:
            st.warning(f"Could not fetch schedule: {e}")
    
    # Check for pending matchup selection (set by game buttons below)
    pending_away = st.session_state.pop('pending_away', None)
    pending_home = st.session_state.pop('pending_home', None)
    pending_side = st.session_state.pop('pending_side', None)
    
    # If we have pending values, override the widget state directly
    matchup_updated = False
    if pending_away:
        st.session_state['manual_away'] = pending_away
        matchup_updated = True
    if pending_home:
        st.session_state['manual_home'] = pending_home
    if pending_side:
        st.session_state['player_side'] = pending_side
    
    # Show success message if matchup was updated
    if matchup_updated:
        st.success(f"✅ Matchup set: {pending_away} @ {pending_home} - Select a player below!")
    
    # Manual matchup mode (always available)
    st.markdown("---")
    st.subheader("🎯 Quick Matchup Prediction")
    st.markdown("*Select a matchup and player to get instant predictions*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        away_team = st.selectbox("Away Team", teams, key="manual_away")
    with col2:
        # Filter out the selected away team
        home_options = [t for t in teams if t != away_team]
        home_team = st.selectbox("Home Team", home_options, key="manual_home")
    with col3:
        is_home = st.radio("Player's Team", ["Away", "Home"], key="player_side", horizontal=True)
    
    selected_team = home_team if is_home == "Home" else away_team
    opponent_team = away_team if is_home == "Home" else home_team
    
    st.markdown(f"**Matchup:** {away_team} @ {home_team}")
    
    # Player selection
    st.markdown("---")
    st.markdown("**🎯 Quick Prediction**")
    
    # Period selection first - very clear
    quick_period = st.radio(
        "⏱️ Period", 
        ["1h", "1q"], 
        format_func=lambda x: "🏀 1st HALF (1H)" if x == "1h" else "⏱️ 1st QUARTER (1Q)", 
        horizontal=True, 
        key="quick_period"
    )
    period_label = "1st HALF" if quick_period == "1h" else "1st QUARTER"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        all_players = get_all_player_names()
        player_name = st.selectbox(
            f"Select Player (from {selected_team})",
            [""] + all_players,
            key="quick_player",
            help="Type to search"
        )
    
    with col2:
        # Period-specific prop types (PrizePicks rules)
        # 1Q: Points, Rebounds, Assists (no PRA)
        # 1H: Points, PRA (no individual REB/AST)
        if quick_period == "1q":
            quick_prop_options = ["points", "rebounds", "assists"]
        else:
            quick_prop_options = ["points", "pra"]
        prop_type = st.selectbox("Stat Type", quick_prop_options, 
            format_func=lambda x: {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "pra": "PRA"}.get(x, x), key="quick_prop_type")
    
    with col3:
        # Smart default lines based on period AND prop type
        quick_default_lines = {
            ("1q", "points"): 6.5, ("1q", "rebounds"): 2.5, ("1q", "assists"): 2.0,
            ("1h", "points"): 12.5, ("1h", "pra"): 20.5,
        }
        default_quick_line = quick_default_lines.get((quick_period, prop_type), 6.5 if quick_period == "1q" else 12.5)
        prop_line = st.number_input(f"{period_label} Line", min_value=0.5, max_value=50.0, value=default_quick_line, step=0.5, key="quick_line")
    
    if player_name and st.button("🔮 Get Prediction", key="quick_predict", type="primary"):
        with st.spinner("Analyzing..."):
            result = cached_prediction(
                player_name=player_name, 
                opponent=opponent_team,
                prop_line=prop_line, 
                prop_type=prop_type, 
                is_home=(is_home == "Home"),
                period=quick_period
            )
            
            if result and 'error' not in result:
                add_to_history(result, prop_type, opponent_team, is_home == "Home", quick_period)
                display_prediction(result, prop_type, prop_line, quick_period)
            else:
                st.error(f"Could not get prediction: {result.get('error', 'Unknown error')}")
    
    # Show fetched games if available
    if games:
        st.markdown("---")
        st.subheader("🏀 Today's Schedule")
        
        for i, game in enumerate(games):
            away = game.get('away_team', '???')
            home = game.get('home_team', '???')
            status = game.get('game_status', '')
            away_name = game.get('away_team_name', '')
            home_name = game.get('home_team_name', '')
            
            with st.expander(f"🏀 {away} @ {home} {' - ' + status if status else ''}"):
                st.markdown(f"**{away_name}** @ **{home_name}**")
                
                # Quick action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Predict {away} players", key=f"pred_away_{i}"):
                        st.session_state['pending_away'] = away
                        st.session_state['pending_home'] = home
                        st.session_state['pending_side'] = "Away"
                        st.rerun()
                with col2:
                    if st.button(f"Predict {home} players", key=f"pred_home_{i}"):
                        st.session_state['pending_away'] = away
                        st.session_state['pending_home'] = home
                        st.session_state['pending_side'] = "Home"
                        st.rerun()
    else:
        st.info("📅 No games found in schedule. Use the matchup selector above to make predictions.")


# ============================================================================
# Tab 5: History & Export
# ============================================================================

def history_tab():
    """Show prediction history and export options"""
    st.subheader("📜 Prediction History")
    
    history = st.session_state.prediction_history
    
    if not history:
        st.info("No predictions yet. Make some predictions to see them here!")
        return
    
    st.markdown(f"*{len(history)} predictions in this session*")
    
    # Convert to DataFrame
    history_data = []
    for h in history:
        history_data.append({
            'Time': h.get('timestamp', ''),
            'Player': h.get('player_name', 'Unknown'),
            'Period': h.get('period_label', '1H'),
            'Type': h.get('prop_type', 'points'),
            'Line': h.get('prop_line', 0),
            'Predicted': h.get('predicted_1h', 0),
            'Pick': h.get('pick', 'N/A'),
            'Confidence': h.get('confidence', 'N/A'),
            'Opponent': h.get('opponent', ''),
            'Location': 'Home' if h.get('is_home') else 'Away'
        })
    
    df = pd.DataFrame(history_data)
    
    # Display
    st.dataframe(df, width="stretch", hide_index=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV",
            csv,
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2:
        json_data = json.dumps(history, indent=2, default=str)
        st.download_button(
            "📥 Export JSON",
            json_data,
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Summary stats
    st.divider()
    st.subheader("📊 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    overs = len([h for h in history if 'OVER' in h.get('pick', '')])
    unders = len(history) - overs
    high_conf = len([h for h in history if h.get('confidence') == 'HIGH'])
    
    with col1:
        st.metric("Total Predictions", len(history))
    with col2:
        st.metric("Overs", overs)
    with col3:
        st.metric("Unders", unders)
    with col4:
        st.metric("High Confidence", high_conf)


def add_to_history(result: dict, prop_type: str, opponent: str, is_home: bool, period: str = '1h'):
    """Add prediction to history"""
    period_label = "1Q" if period == '1q' else "1H"
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'player_name': result.get('player_name', 'Unknown'),
        'prop_type': prop_type,
        'period': period,
        'period_label': period_label,
        'prop_line': result.get('prop_line', 0),
        'predicted_1h': result.get('predicted_1h', 0),
        'pick': result.get('pick', 'N/A'),
        'confidence': result.get('confidence', 'N/A'),
        'opponent': opponent,
        'is_home': is_home
    }
    st.session_state.prediction_history.append(history_entry)


# ============================================================================
# Display Components
# ============================================================================

def display_prediction(result: dict, prop_type: str, prop_line: float, period: str = '1h'):
    """Display single prediction result with lock rating"""
    
    # Check if player is OUT
    if result.get('error'):
        error_msg = result.get('error', 'Unknown error')
        player_name = result.get('player_name', 'Player')
        injury_status = result.get('injury_status')
        
        st.error(f"🏥 **{error_msg}**")
        if injury_status:
            st.warning(f"Injury: {injury_status.get('injury', 'Unknown')} | Status: {injury_status.get('status', 'OUT')}")
        st.info("This player cannot be predicted while listed as OUT. Check injury reports for updates.")
        return
    
    pick = result.get('pick', 'N/A')
    confidence = result.get('confidence', 'N/A')
    confidence_desc = result.get('confidence_desc', '')
    lock_score = result.get('lock_score', 50)
    lock_factors = result.get('lock_factors', [])
    predicted = result.get('predicted_1h', 0)
    difference = result.get('difference', 0)
    abs_edge = abs(difference)
    period_label = "1Q" if period == '1q' else "1H"
    
    # Check if using real period data
    has_real_data = result.get('has_real_period_data', False)
    data_source = result.get('data_source', 'ESTIMATED')
    
    # Determine quality rating (Score + Edge combined)
    if lock_score >= 85 and abs_edge >= 3.5:
        quality = "🎯 ELITE BET"
        quality_color = "#00ff00"
    elif lock_score >= 85 and abs_edge >= 2.0:
        quality = "✅ SOLID BET"
        quality_color = "#FFD700"
    elif lock_score >= 85 and abs_edge < 2.0:
        quality = "⚠️ THIN EDGE - RISKY"
        quality_color = "#FFA500"
    elif lock_score >= 72 and abs_edge >= 2.5:
        quality = "👍 GOOD BET"
        quality_color = "#2ecc71"
    elif lock_score >= 72:
        quality = "📊 MARGINAL"
        quality_color = "#3498db"
    else:
        quality = "❌ SKIP"
        quality_color = "#e74c3c"
    
    is_over = 'OVER' in pick
    is_skip = 'SKIP' in pick
    
    # Color based on confidence level
    if confidence == 'LOCK':
        pick_color = "#FFD700"  # Gold for locks
        bg_color = '#3d3d00'
    elif confidence == 'HIGH':
        pick_color = "#2ecc71"
        bg_color = '#1a472a'
    elif confidence == 'MEDIUM':
        pick_color = "#3498db"
        bg_color = '#1a3a4a'
    elif is_skip:
        pick_color = "#95a5a6"
        bg_color = '#2d2d2d'
    else:
        pick_color = "#e74c3c" if not is_over else "#f39c12"
        bg_color = '#4a1a1a' if not is_over else '#4a3a1a'
    
    # Main card with lock score and quality rating
    stat_label = f'{period_label} {prop_type.upper()}'
    
    # Data source badge
    if has_real_data:
        data_badge = '<span style="background: #00aa00; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7em; margin-left: 5px;">REAL DATA</span>'
    else:
        data_badge = '<span style="background: #666; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7em; margin-left: 5px;">ESTIMATED</span>'
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {bg_color} 0%, #1e1e2e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid {pick_color};
        text-align: center;
        margin: 10px 0;
    ">
        <h4 style="color: #888; margin: 0;">{result.get('player_name', 'Unknown')} {data_badge}</h4>
        <h1 style="color: {pick_color}; margin: 10px 0; font-size: 2.5em;">{pick}</h1>
        <p style="color: #888; margin: 5px 0;">
            {stat_label} | Line: {prop_line}
        </p>
        <div style="
            background: #0a0a1e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            border: 2px solid {quality_color};
        ">
            <span style="color: {quality_color}; font-size: 1.8em; font-weight: bold;">{quality}</span>
            <p style="color: #aaa; margin: 8px 0 0 0; font-size: 0.9em;">
                Score: {lock_score} | Edge: {difference:+.1f} pts
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Edge + Confidence reminder
    if lock_score >= 85 and abs_edge < 2.0:
        st.warning("⚡ **THIN EDGE WARNING:** High confidence but small edge (<2 pts). One hot/cold quarter can flip this. Consider skipping for Flex plays.")
    elif lock_score >= 85 and abs_edge >= 3.5:
        st.success("✅ **EDGE + CONFIDENCE:** Both score AND edge are strong. This is an ideal pick for Flex plays.")
    
    # INJURY CONFLICT WARNING
    injury_info = result.get('injury_info')
    injury_boost = result.get('injury_boost', 1.0)
    if injury_info and injury_boost and injury_boost > 1.05:
        is_under = 'UNDER' in pick
        if is_under:
            teammate_boost = injury_info.get('teammate_boost', 1.0)
            opp_boost = injury_info.get('opponent_boost', 1.0)
            out_teammates = injury_info.get('out_teammates', [])
            
            warning_parts = []
            if teammate_boost > 1.05 and out_teammates:
                warning_parts.append(f"**{', '.join(out_teammates[:2])}** OUT (+{(teammate_boost-1)*100:.0f}% usage boost)")
            if opp_boost > 1.05:
                warning_parts.append(f"Opponent injuries make scoring easier (+{(opp_boost-1)*100:.0f}%)")
            
            if warning_parts:
                st.error(f"""
                🏥 **INJURY CONFLICT - PROCEED WITH CAUTION**
                
                This is an **UNDER** pick, but injuries suggest **higher scoring**:
                - {chr(10).join(['- ' + p for p in warning_parts])}
                
                Total injury boost: **+{(injury_boost-1)*100:.1f}%** to prediction
                
                Consider: The line may already account for injuries, OR this UNDER may be riskier than it appears.
                """)
    
    # Warning banner for unreliable predictions
    if result.get('prediction_warning'):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #8B0000 0%, #4a1a1a 100%);
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #ff4444;
            margin: 10px 0;
            text-align: center;
        ">
            <span style="color: #ff6666; font-size: 1.1em; font-weight: bold;">
                ⚠️ PREDICTION RELIABILITY WARNING
            </span>
            <p style="color: #ffaaaa; margin: 8px 0 0 0; font-size: 0.95em;">
                {result.get('prediction_warning')}
            </p>
            <p style="color: #aaa; margin: 5px 0 0 0; font-size: 0.85em;">
                Original prediction: {result.get('original_prediction', 'N/A')} 1H ({result.get('original_full_game', 'N/A')} full) → 
                Adjusted to: {result.get('predicted_1h', 'N/A')} 1H ({result.get('full_game_prediction', 'N/A')} full)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Lock score bar
    display_lock_score_bar(lock_score, period)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Predicted {period_label}", f"{predicted:.1f}")
    with col2:
        st.metric("Line", f"{prop_line}")
    with col3:
        delta_color = "normal" if is_over else "inverse"
        st.metric("Edge", f"{difference:+.1f}", delta_color=delta_color)
    with col4:
        st.metric("Full Game", f"{result.get('full_game_prediction', 0):.1f}")
    
    st.divider()
    
    # Lock Factors breakdown
    if lock_factors:
        st.subheader("🔍 Why This Rating?")
        
        # Create columns for factors
        factor_cols = st.columns(min(len(lock_factors), 3))
        for i, factor in enumerate(lock_factors):
            col_idx = i % 3
            with factor_cols[col_idx]:
                score_color = "#2ecc71" if '+' in str(factor.get('score', '')) else "#e74c3c"
                st.markdown(f"""
                <div style="
                    background: #262730;
                    padding: 12px;
                    border-radius: 8px;
                    margin: 5px 0;
                    border-left: 4px solid {score_color};
                ">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #ffffff;">{factor.get('name', '')}</strong>
                        <span style="color: {score_color}; font-weight: bold;">{factor.get('score', '')}</span>
                    </div>
                    <p style="color: #cccccc; margin: 5px 0 0 0; font-size: 0.85em;">{factor.get('desc', '')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display injury information if available
    injury_info = result.get('injury_info')
    injury_boost = result.get('injury_boost', 1.0)
    
    if injury_info or (injury_boost and injury_boost != 1.0):
        st.divider()
        st.subheader("🏥 Injury Impact")
        
        icol1, icol2 = st.columns(2)
        
        with icol1:
            if injury_info:
                out_teammates = injury_info.get('out_teammates', [])
                if out_teammates:
                    st.markdown(f"**Teammates OUT:** {', '.join(out_teammates[:3])}")
                    teammate_boost = injury_info.get('teammate_boost', 1.0)
                    if teammate_boost > 1.0:
                        boost_pct = (teammate_boost - 1) * 100
                        st.markdown(f"📈 Usage boost: **+{boost_pct:.0f}%**")
        
        with icol2:
            if injury_info:
                out_opponents = injury_info.get('out_opponents', [])
                if out_opponents:
                    st.markdown(f"**Opponents OUT:** {', '.join(out_opponents[:3])}")
                    opp_boost = injury_info.get('opponent_boost', 1.0)
                    if opp_boost > 1.0:
                        boost_pct = (opp_boost - 1) * 100
                        st.markdown(f"🎯 Easier matchup: **+{boost_pct:.0f}%**")
        
        if injury_boost and injury_boost != 1.0:
            total_pct = (injury_boost - 1) * 100
            st.info(f"**Combined Injury Adjustment:** +{total_pct:.1f}% to prediction")
    
    st.divider()
    
    # Reasoning and Models
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Analysis")
        reasons = result.get('reasons', [])
        if reasons:
            for reason in reasons:
                reason_lower = reason.lower()
                if any(w in reason_lower for w in ['strong', 'above', 'advantage', 'well-rested', 'agree', 'supports', 'hot', 'lock']):
                    icon = "✅"
                elif any(w in reason_lower for w in ['concerns', 'below', 'fatigue', 'against', 'cold', 'risk', 'caution']):
                    icon = "⚠️"
                elif '🏠' in reason or '✈️' in reason or '⚠️' in reason:
                    icon = ""  # Already has icon
                else:
                    icon = "ℹ️"
                st.markdown(f"{icon} {reason}")
    
    with col2:
        st.subheader("🎯 Model Votes")
        
        # Check if prediction was capped due to unreliability
        if result.get('prediction_warning'):
            st.warning("⚠️ **Model outputs unreliable for this player**")
            st.markdown(f"""
            <div style="background: #3d2d00; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <p style="color: #ffcc00; margin: 0; font-size: 0.9em;">
                    <strong>Adjusted Pick:</strong> {result.get('predicted_1h', 0):.1f} 1H → 
                    {'UNDER' if result.get('predicted_1h', 0) < prop_line else 'OVER'}<br>
                    <span style="color: #aaa;">Original models gave unrealistic values - use caution</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show original model votes with strikethrough/dimmed
            model_breakdown = result.get('model_breakdown', {})
            st.caption("Original (unreliable) model outputs:")
            for model_name, pred in model_breakdown.items():
                st.markdown(f"<span style='color: #666;'>~~{model_name.upper()}: {pred:.1f}~~</span>", unsafe_allow_html=True)
        else:
            model_breakdown = result.get('model_breakdown', {})
            over_count = 0
            under_count = 0
            for model_name, pred in model_breakdown.items():
                is_model_over = pred > prop_line
                if is_model_over:
                    over_count += 1
                else:
                    under_count += 1
                pred_icon = "🟢 OVER" if is_model_over else "🔴 UNDER"
                st.markdown(f"**{model_name.upper()}**: {pred:.1f} → {pred_icon}")
            
            # Model consensus
            total_models = over_count + under_count
            if over_count == total_models and total_models > 0:
                st.success(f"✅ All {total_models} models agree: OVER")
            elif under_count == total_models and total_models > 0:
                st.success(f"✅ All {total_models} models agree: UNDER")
            elif over_count > 0 and under_count > 0:
                st.warning(f"⚠️ Split decision: {over_count} OVER, {under_count} UNDER")
            else:
                st.info("ℹ️ No model predictions available")
    
    # Recent games
    recent_games = result.get('recent_games', [])
    if recent_games:
        st.divider()
        st.subheader("📊 Recent Games")
        display_recent_games(recent_games, prop_type, prop_line)


def display_lock_score_bar(lock_score: int, period: str = '1h'):
    """Display a visual lock score bar with period-specific thresholds"""
    thresholds = get_lock_thresholds(period)
    
    # Determine color and label based on score
    if lock_score >= thresholds['lock']:
        bar_color = "#FFD700"  # Gold
        label = "🔒 LOCK"
    elif lock_score >= thresholds['strong']:
        bar_color = "#2ecc71"  # Green
        label = "🔥 STRONG"
    elif lock_score >= thresholds['playable']:
        bar_color = "#3498db"  # Blue
        label = "✅ PLAYABLE"
    elif lock_score >= thresholds['lean']:
        bar_color = "#f39c12"  # Orange
        label = "⚠️ RISKY"
    else:
        bar_color = "#e74c3c"  # Red
        label = "❌ AVOID"
    
    # Simple text-based scale
    if period == '1q':
        scale_text = "0 Skip | 50 Lean | 65 Play | 80 Strong | 90+ Lock"
        title_extra = " (1Q)"
    else:
        scale_text = "0 Skip | 40 Lean | 55 Play | 72 Strong | 85+ Lock"
        title_extra = ""
    
    # Display using Streamlit native components
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Lock Score{title_extra}")
    with col2:
        st.markdown(f"**{label}**")
    
    # Progress bar
    st.progress(lock_score / 100)
    
    # Scale reference
    st.caption(scale_text)


def display_recent_games(recent_games: list, prop_type: str, prop_line: float):
    """Display recent games table"""
    df = pd.DataFrame(recent_games)
    if df.empty:
        st.info("No recent games data available")
        return
    
    display_cols = ['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'PRA', 'MIN']
    available_cols = [c for c in display_cols if c in df.columns]
    
    if not available_cols:
        st.info("No game data columns available")
        return
    
    df_display = df[available_cols].copy()
    
    stat_col = 'PTS' if prop_type == 'points' else 'PRA'
    if stat_col in df_display.columns:
        ratios = get_first_half_ratios()
        ratio = ratios.get(prop_type, 0.48)
        df_display['Est_1H'] = (df_display[stat_col] * ratio).round(1)
        
        # Style function with readable colors
        def highlight_hit(row):
            try:
                est_1h = float(row.get('Est_1H', 0) or 0)
                if est_1h > prop_line:
                    return ['background-color: #2ecc71; color: #000000'] * len(row)
                else:
                    return ['background-color: #e74c3c; color: #ffffff'] * len(row)
            except:
                return [''] * len(row)
        
        try:
            styled_df = df_display.style.apply(highlight_hit, axis=1)
            st.dataframe(styled_df, width="stretch", hide_index=True)
        except Exception as e:
            # Fallback to unstyled if styling fails
            st.dataframe(df_display, width="stretch", hide_index=True)
        
        # Calculate hit rate
        try:
            hit_count = (df_display['Est_1H'] > prop_line).sum()
            total = len(df_display)
            if total > 0:
                st.caption(f"📈 Estimated hit rate: {hit_count}/{total} ({hit_count/total*100:.0f}%)")
        except:
            pass
    else:
        st.dataframe(df_display, width="stretch", hide_index=True)


def show_instructions():
    """Show welcome instructions"""
    st.markdown("""
    ## Welcome! 👋
    
    This tool predicts NBA player stats for **1st Half** or **1st Quarter** props.
    
    ---
    
    ### ⏱️ IMPORTANT: 1H vs 1Q
    
    | Period | What it means | Typical Lines | Notes |
    |--------|--------------|---------------|-------|
    | **🏀 1st HALF (1H)** | Stats at HALFTIME | 10-15 pts | More stable, recommended |
    | **⏱️ 1st QUARTER (1Q)** | Stats after Q1 ONLY | 5-8 pts | Stricter thresholds applied |
    
    **Make sure you select the correct period that matches your PrizePicks prop!**
    
    ---
    
    ### Quick Start:
    1. **Select the PERIOD first** - 1H or 1Q (this is critical!)
    2. Enter the player name
    3. Enter the PrizePicks line
    4. Select opponent and location
    5. Click **Get Prediction**
    
    ---
    
    ### Lock Score Guide:
    
    **1st HALF (1H) Thresholds:**
    | Score | Rating | Action |
    |-------|--------|--------|
    | 85+ | 🔒 LOCK | High confidence play |
    | 72-84 | 🔥 STRONG | Good play |
    | 55-71 | ✅ PLAYABLE | Decent edge |
    | 40-54 | ⚠️ LEAN | Risky |
    | <40 | ❓ SKIP | Avoid |
    
    **1st QUARTER (1Q) Thresholds (STRICTER):**
    | Score | Rating | Action |
    |-------|--------|--------|
    | 92+ | 🔒 LOCK | High confidence play |
    | 82-91 | 🔥 STRONG | Good play |
    | 65-81 | ✅ PLAYABLE | Decent edge |
    | 50-64 | ⚠️ LEAN | Risky |
    | <50 | ❓ SKIP | Avoid |
    
    *1Q uses stricter thresholds because of higher variance in short periods.*
    """)


def show_footer():
    """Footer"""
    st.markdown("""---
<div style="text-align: center; color: #666;">
<p>NBA 1H & 1Q Prop Predictor | Ensemble ML Model | Data from NBA API</p>
<p>⚠️ For entertainment purposes only. Please gamble responsibly.</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
