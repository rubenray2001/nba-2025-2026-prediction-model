"""
Configuration settings for NBA 1H Prediction Model
EXTREME ACCURACY MODE - Maximum predictive power
"""

# Current NBA Season
CURRENT_SEASON = "2025-26"
SEASONS_TO_FETCH = ["2022-23", "2023-24", "2024-25", "2025-26"]  # More historical data

# =============================================================================
# MODEL HYPERPARAMETERS - Tuned for Maximum Accuracy
# =============================================================================

MODEL_CONFIG = {
    "lightgbm": {
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "min_split_gain": 0.01,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
        "importance_type": "gain"
    },
    "xgboost": {
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
        "n_jobs": -1
    },
    "catboost": {
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False,
        "loss_function": "MAE",
        "task_type": "CPU"
    }
}

# Ensemble Weights - Optimized based on typical model performance
ENSEMBLE_WEIGHTS = {
    "lightgbm": 0.45,
    "xgboost": 0.35,
    "catboost": 0.20
}

# =============================================================================
# FEATURE CONFIGURATION - Extended Windows for Better Patterns
# =============================================================================

ROLLING_WINDOWS = [3, 5, 7, 10, 15, 20]  # More granular + longer memory
WEIGHTED_WINDOWS = {
    3: 0.35,   # Recent games weighted highest
    5: 0.25,
    7: 0.20,
    10: 0.12,
    15: 0.05,
    20: 0.03
}

HOME_AWAY_SPLIT = True

# =============================================================================
# FIRST HALF RATIOS - Refined Estimates by Stat Type
# =============================================================================

FIRST_HALF_RATIOS = {
    "points": 0.478,     # Slightly under 50% due to 4th quarter scoring
    "rebounds": 0.465,   # Fewer rebounds in 1H typically
    "assists": 0.485,    # Fairly even distribution
    "pra": 0.476,        # Weighted average
    "minutes": 0.50      # Usually equal halves
}

# =============================================================================
# FIRST QUARTER RATIOS - Estimates for 1Q Predictions
# =============================================================================

FIRST_QUARTER_RATIOS = {
    "points": 0.242,     # ~24.2% of full game points in 1Q (slightly higher than 25% due to fresh legs)
    "rebounds": 0.230,   # ~23% of rebounds - fewer early as pace settles
    "assists": 0.245,    # ~24.5% of assists
    "pra": 0.240,        # Weighted average for PRA
    "minutes": 0.25      # ~25% of minutes (12 min quarter / 48 min game)
}

# Position-specific adjustments (some positions score more in 1H)
POSITION_1H_ADJUSTMENTS = {
    "G": 1.02,   # Guards slightly better in 1H
    "F": 1.00,   # Forwards neutral
    "C": 0.98,   # Centers slightly lower in 1H
}

# =============================================================================
# ADVANCED FEATURE THRESHOLDS
# =============================================================================

# Hot/Cold streak thresholds
STREAK_THRESHOLDS = {
    "hot": 1.15,    # 15% above average = hot streak
    "cold": 0.85,   # 15% below average = cold streak
}

# Consistency thresholds (std dev as % of mean)
CONSISTENCY_THRESHOLDS = {
    "very_consistent": 0.15,   # <15% variation
    "consistent": 0.25,        # 15-25% variation
    "volatile": 0.40,          # >40% variation = volatile
}

# Rest day impact multipliers
REST_DAY_IMPACT = {
    0: 0.92,   # B2B - 8% reduction expected
    1: 0.96,   # 1 day rest - 4% reduction
    2: 1.00,   # Normal rest
    3: 1.02,   # Well rested - 2% boost
    4: 1.03,   # Very rested
    5: 1.02,   # Diminishing returns
}

# =============================================================================
# TEAM MAPPINGS
# =============================================================================

TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

ABBREVIATION_TO_TEAM = {v: k for k, v in TEAM_ABBREVIATIONS.items()}

# Team defensive tier rankings (1 = best defense, 5 = worst)
# Updated periodically based on defensive ratings
TEAM_DEFENSIVE_TIERS = {
    "BOS": 1, "CLE": 1, "OKC": 1,  # Elite defenses
    "MIA": 2, "NYK": 2, "MEM": 2, "LAC": 2,  # Good defenses
    "DEN": 3, "PHX": 3, "MIL": 3, "DAL": 3, "MIN": 3, "GSW": 3,  # Average
    "LAL": 4, "SAC": 4, "NOP": 4, "IND": 4, "ATL": 4, "CHI": 4,  # Below average
    "HOU": 5, "SAS": 5, "POR": 5, "UTA": 5, "DET": 5, "WAS": 5, "CHA": 5,  # Poor
    "BKN": 4, "TOR": 4, "PHI": 3, "ORL": 2  # Others
}

# Team pace tiers (1 = fastest, 5 = slowest)
# Updated based on 2025-26 season data
TEAM_PACE_TIERS = {
    "IND": 1, "ATL": 1, "SAC": 1, "PHI": 1,  # Fastest pace (PHI added - high scoring)
    "MIL": 2, "MIN": 2, "DEN": 2, "OKC": 2, "HOU": 2, "POR": 2,  # Fast
    "PHX": 3, "LAL": 3, "GSW": 3, "DAL": 3, "NOP": 3, "UTA": 3, "CHI": 3,  # Average
    "BOS": 4, "NYK": 4, "MIA": 4, "CLE": 4, "MEM": 4, "TOR": 4, "BKN": 4, "LAC": 4,  # Slower
    "ORL": 5, "DET": 5, "WAS": 5, "SAS": 5, "CHA": 5,  # Slowest
}

# =============================================================================
# ACCURACY & CONFIDENCE CALIBRATION - ENHANCED FOR MAX ACCURACY
# =============================================================================

# Lock score thresholds
LOCK_THRESHOLDS = {
    "lock": 85,       # High confidence
    "strong": 72,     # Good confidence
    "playable": 55,   # Decent confidence
    "lean": 40,       # Marginal
    "skip": 0
}

# 1Q-SPECIFIC thresholds - stricter due to variance
LOCK_THRESHOLDS_1Q = {
    "lock": 90,       # Higher bar for 1Q
    "strong": 80,     # Good confidence
    "playable": 65,   # Decent confidence
    "lean": 50,       # Marginal
    "skip": 0
}

# 1Q-SPECIFIC adjustments - penalize variance more
Q1_ADJUSTMENTS = {
    "min_edge_multiplier": 1.5,      # Need 50% larger edge for 1Q
    "consistency_weight": 1.5,        # Weight consistency 50% more for 1Q
    "volatility_penalty": 1.5,        # Penalize volatile players 50% more
    "min_games_multiplier": 1.25,     # Need 25% more games for confidence
    "model_consensus_weight": 1.3,    # Weight model agreement more
}

# =============================================================================
# STAT-SPECIFIC ADJUSTMENTS - Makes rebounds/assists as strong as points
# =============================================================================

# Stat-specific variance (how much the stat varies game-to-game)
STAT_VARIANCE = {
    "points": 1.0,      # Baseline - points are most predictable
    "rebounds": 1.25,   # 25% more variance - rebounds can swing more
    "assists": 1.20,    # 20% more variance - depends on team flow
    "pra": 1.10,        # Combined stat smooths out variance
}

# Minimum edge required by stat type (as % of line)
STAT_MIN_EDGE_PCT = {
    "points": 8,       # 8% edge minimum for points
    "rebounds": 12,    # 12% edge for rebounds (more volatile)
    "assists": 12,     # 12% edge for assists
    "pra": 7,          # 7% for PRA (smoothed)
}

# Stat-specific consistency thresholds (std dev / mean)
STAT_CONSISTENCY = {
    "points": {"elite": 0.15, "good": 0.25, "volatile": 0.40},
    "rebounds": {"elite": 0.20, "good": 0.35, "volatile": 0.50},
    "assists": {"elite": 0.25, "good": 0.40, "volatile": 0.55},
    "pra": {"elite": 0.12, "good": 0.22, "volatile": 0.35},
}

# Position-specific bonuses by stat type
POSITION_STAT_BONUSES = {
    # Centers excel at rebounds
    "C": {"rebounds": 1.08, "assists": 0.95, "points": 1.0, "pra": 1.02},
    # Guards excel at assists
    "G": {"rebounds": 0.95, "assists": 1.08, "points": 1.02, "pra": 1.0},
    # Forwards are balanced
    "F": {"rebounds": 1.02, "assists": 1.0, "points": 1.0, "pra": 1.01},
}

# Players known for early rebounds (great rebounders who start strong)
EARLY_REBOUNDERS = [
    "Rudy Gobert", "Anthony Davis", "Domantas Sabonis", "Nikola Jokic", 
    "Joel Embiid", "Bam Adebayo", "Karl-Anthony Towns", "Jarrett Allen",
    "Evan Mobley", "Alperen Sengun", "Ivica Zubac", "Nic Claxton"
]

# Players known for early assists (high-assist players who set tone early)
EARLY_PLAYMAKERS = [
    "Tyrese Haliburton", "Trae Young", "Luka Doncic", "Nikola Jokic",
    "James Harden", "Chris Paul", "Darius Garland", "Dejounte Murray",
    "LaMelo Ball", "De'Aaron Fox", "Jalen Brunson", "Fred VanVleet"
]

# Stat-specific floor/ceiling factors (low lines more volatile)
STAT_LINE_VOLATILITY = {
    # Low lines are harder to predict
    "points": {"low_threshold": 8, "penalty": -5},
    "rebounds": {"low_threshold": 4, "penalty": -8},   # Very low reb lines are risky
    "assists": {"low_threshold": 3, "penalty": -8},    # Very low ast lines are risky
    "pra": {"low_threshold": 12, "penalty": -5},
}

# Player scoring style indicators (affects Q1 predictions)
# "early" = tends to score more in Q1, "late" = closer/4th quarter scorer, "even" = consistent throughout
# This is approximated from typical player patterns
PLAYER_SCORING_STYLES = {
    # Early scorers - good for 1Q OVER
    "fast_starters": ["Stephen Curry", "Damian Lillard", "Trae Young", "Luka Doncic", "Ja Morant",
                      "De'Aaron Fox", "Tyrese Haliburton", "Donovan Mitchell", "Kyrie Irving"],
    # Late scorers/closers - risky for 1Q OVER
    "closers": ["LeBron James", "Kevin Durant", "Kawhi Leonard", "Jimmy Butler", "Paul George",
                "Jayson Tatum", "Devin Booker", "Giannis Antetokounmpo"],
}

# 1Q scoring ratio adjustments by player type
Q1_RATIO_ADJUSTMENTS = {
    "fast_starter": 1.08,    # 8% higher than base 1Q ratio
    "closer": 0.92,          # 8% lower than base 1Q ratio  
    "default": 1.0           # Use standard ratio
}

# Minimum edge required for each confidence level - STRICTER
MIN_EDGE_REQUIREMENTS = {
    "lock": 3.0,      # Need 3.0+ point edge for lock (was 2.5)
    "strong": 2.2,    # Raised (was 1.8)
    "playable": 1.3,  # Raised (was 1.0)
    "lean": 0.7       # Raised (was 0.5)
}

# Minimum games required for confident picks
MIN_GAMES_FOR_CONFIDENCE = {
    "lock": 15,       # Need 15+ games for lock
    "strong": 10,
    "playable": 7,
    "lean": 5
}

# Suspicious edge thresholds (edges too large may indicate bad data)
SUSPICIOUS_EDGE_THRESHOLDS = {
    "warning": 5.0,   # 5+ points is unusual
    "critical": 7.0   # 7+ points likely data issue
}

# =============================================================================
# HIGH VARIANCE PLAYERS - UNDER picks are DANGEROUS on these players
# These players have extreme game-to-game variance and can explode any night
# =============================================================================

HIGH_VARIANCE_STARS = [
    # Superstars with massive upside ceilings - UNDER is always risky
    "Victor Wembanyama", "Luka Doncic", "Luka Dončić",
    "Anthony Edwards", "Ja Morant", "Trae Young", "LaMelo Ball",
    "Shai Gilgeous-Alexander", "Jayson Tatum", "Giannis Antetokounmpo",
    "Stephen Curry", "Kevin Durant", "Donovan Mitchell",
    "Damian Lillard", "Devin Booker", "Kyrie Irving",
    "Anthony Davis", "Nikola Jokic", "Joel Embiid",
    "Tyrese Maxey", "De'Aaron Fox", "Cade Cunningham",
    "Jalen Brunson", "Paolo Banchero", "Zion Williamson",
]

# Role-expanding young players - their averages don't reflect current usage
ROLE_EXPANDING_PLAYERS = [
    "Isaiah Collier", "Cooper Flagg", "Dylan Harper", "Stephon Castle",
    "Matas Buzelis", "Moussa Diabaté", "Asa Newell", "Derik Queen",
    "Dalton Knecht", "Chet Holmgren", "Cam Thomas", "Jalen Green",
    "Amen Thompson", "Ausar Thompson", "Jaime Jaquez Jr",
    "Bub Carrington", "Jaylen Wells", "Donovan Clingan",
]

# Recently traded players - their stats from old team may not apply
# Format: "Player Name": {"new_team": "ABR", "old_team": "ABR", "trade_date": "YYYY-MM-DD"}
# NOTE: Remove players from this list once they've played ~15+ games on new team
# and their period stats reflect the new team context
RECENTLY_TRADED_PLAYERS = {
    "Luka Doncic": {"new_team": "LAL", "old_team": "DAL", "trade_date": "2026-01-01"},
    "Luka Dončić": {"new_team": "LAL", "old_team": "DAL", "trade_date": "2026-01-01"},
    "CJ McCollum": {"new_team": "ATL", "old_team": "NOP", "trade_date": "2026-01-15"},
    # Kevin Durant removed - has enough games on HOU for stats to be valid
}

# =============================================================================
# UNDER PICK SAFETY THRESHOLDS
# =============================================================================

# Minimum edge (in points) required for UNDER picks by player type
UNDER_MIN_EDGE = {
    "star": 3.5,          # Need 3.5+ pts edge for UNDER on a star
    "high_variance": 3.0, # Need 3.0+ pts edge for UNDER on volatile player
    "role_expanding": 2.5,# Need 2.5+ pts edge for UNDER on role-expanding player
    "default": 1.5        # Standard UNDER minimum edge
}

# Volatility penalty for UNDER picks (applied to lock score)
UNDER_VOLATILITY_PENALTY = {
    "star": -15,          # -15 lock score for UNDER on a star player
    "high_variance": -10, # -10 for high variance player
    "role_expanding": -8, # -8 for role expanding player
}

# =============================================================================
# ROLE CHANGE DETECTION THRESHOLDS
# =============================================================================

# If recent 5-game avg deviates from season avg by this %, flag as role change
ROLE_CHANGE_THRESHOLD = 0.30  # 30% deviation = likely role change
# Use recent form more when role change detected
ROLE_CHANGE_RECENT_WEIGHT = 0.70  # 70% weight to recent 5 games when role change detected

# =============================================================================
# DATA PATHS
# =============================================================================

DATA_DIR = "data"
MODELS_DIR = "models"
CACHE_DIR = "cache"
