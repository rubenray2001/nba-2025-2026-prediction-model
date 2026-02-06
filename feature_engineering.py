"""
Feature Engineering Module for NBA 1H Predictions
EXTREME ACCURACY MODE - 50+ Advanced Features

Performance Optimized:
- Reduced DataFrame copies
- Vectorized operations
- In-memory caching for repeated operations
- Efficient rolling calculations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from functools import lru_cache

from config import (
    ROLLING_WINDOWS, 
    WEIGHTED_WINDOWS,
    FIRST_HALF_RATIOS,
    FIRST_QUARTER_RATIOS,
    CURRENT_SEASON,
    TEAM_ABBREVIATIONS,
    STREAK_THRESHOLDS,
    CONSISTENCY_THRESHOLDS,
    REST_DAY_IMPACT,
    TEAM_DEFENSIVE_TIERS,
    TEAM_PACE_TIERS
)
from data_collector import (
    get_player_game_logs,
    get_player_multi_season_logs,
    get_team_defensive_stats,
    get_team_pace_stats,
    get_matchup_history,
    search_player,
    get_team_by_abbreviation
)


def calculate_pra(pts: float, reb: float, ast: float) -> float:
    """Calculate Points + Rebounds + Assists"""
    return pts + reb + ast


def calculate_rolling_stats(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Calculate rolling statistics with extended features.
    Returns features for each game based on PREVIOUS games only (no data leakage).
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    
    if df.empty:
        return df
    
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    
    # Core stats to track
    core_stats = ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']
    stats = [s for s in core_stats if s in df.columns]
    
    # Add derived stats
    if 'PTS' in df.columns and 'REB' in df.columns and 'AST' in df.columns:
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        stats.append('PRA')
    
    if 'FGM' in df.columns and 'FGA' in df.columns:
        df['FG_PCT'] = (df['FGM'] / df['FGA'].replace(0, 1)).fillna(0)
        stats.append('FG_PCT')
    
    if 'FG3M' in df.columns and 'FG3A' in df.columns:
        df['FG3_PCT'] = (df['FG3M'] / df['FG3A'].replace(0, 1)).fillna(0)
        stats.append('FG3_PCT')
    
    if 'FTM' in df.columns and 'FTA' in df.columns:
        df['FT_PCT'] = (df['FTM'] / df['FTA'].replace(0, 1)).fillna(0)
        stats.append('FT_PCT')
    
    new_columns = {}
    
    for stat in stats:
        stat_values = df[stat].values.astype(float)
        shifted = np.roll(stat_values, 1).astype(float)
        shifted[0] = np.nan
        
        for window in windows:
            shifted_series = pd.Series(shifted)
            
            # Rolling mean
            new_columns[f'{stat}_L{window}_avg'] = shifted_series.rolling(
                window=window, min_periods=1
            ).mean().values
            
            # Rolling std
            new_columns[f'{stat}_L{window}_std'] = shifted_series.rolling(
                window=window, min_periods=2
            ).std().values
            
            # Rolling max
            new_columns[f'{stat}_L{window}_max'] = shifted_series.rolling(
                window=window, min_periods=1
            ).max().values
            
            # Rolling min
            new_columns[f'{stat}_L{window}_min'] = shifted_series.rolling(
                window=window, min_periods=1
            ).min().values
            
            # Rolling median (more robust than mean)
            new_columns[f'{stat}_L{window}_median'] = shifted_series.rolling(
                window=window, min_periods=1
            ).median().values
        
        # Season averages
        expanding_mean = pd.Series(stat_values).expanding().mean().values.astype(float)
        shifted_expanding = np.roll(expanding_mean, 1).astype(float)
        shifted_expanding[0] = np.nan
        new_columns[f'{stat}_season_avg'] = shifted_expanding
        
        # Season std
        expanding_std = pd.Series(stat_values).expanding().std().values.astype(float)
        shifted_std = np.roll(expanding_std, 1).astype(float)
        shifted_std[0] = np.nan
        new_columns[f'{stat}_season_std'] = shifted_std
    
    # Use pd.concat for better performance (avoids fragmentation)
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_weighted_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate exponentially weighted averages - recent games matter more.
    """
    if df.empty:
        return df
    
    new_columns = {}
    
    for stat in ['PTS', 'REB', 'AST', 'PRA', 'MIN']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        
        # Exponential weighted mean (span = number of periods)
        ewm_5 = pd.Series(stat_values).ewm(span=5, adjust=False).mean().values.astype(float)
        ewm_10 = pd.Series(stat_values).ewm(span=10, adjust=False).mean().values.astype(float)
        
        # Shift to prevent leakage
        new_columns[f'{stat}_ewm_5'] = np.roll(ewm_5, 1).astype(float)
        new_columns[f'{stat}_ewm_10'] = np.roll(ewm_10, 1).astype(float)
        new_columns[f'{stat}_ewm_5'][0] = np.nan
        new_columns[f'{stat}_ewm_10'][0] = np.nan
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum/trend features - is player trending up or down?
    """
    if df.empty or len(df) < 5:
        return df
    
    new_columns = {}
    
    for stat in ['PTS', 'PRA', 'MIN']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        
        # 3-game vs 10-game trend (positive = trending up)
        l3_avg = pd.Series(stat_values).rolling(3, min_periods=1).mean().values
        l10_avg = pd.Series(stat_values).rolling(10, min_periods=3).mean().values
        
        momentum = (l3_avg / np.where(l10_avg > 0, l10_avg, 1)) - 1
        new_columns[f'{stat}_momentum'] = np.roll(momentum, 1).astype(float)
        new_columns[f'{stat}_momentum'][0] = np.nan
        
        # Game-over-game change
        diff = np.diff(stat_values, prepend=stat_values[0])
        new_columns[f'{stat}_last_change'] = np.roll(diff, 1).astype(float)
        new_columns[f'{stat}_last_change'][0] = np.nan
        
        # Streak detection (consecutive games above/below average)
        season_avg = pd.Series(stat_values).expanding().mean().values
        above_avg = (stat_values > season_avg).astype(int)
        
        streak = np.zeros(len(df))
        for i in range(1, len(df)):
            if above_avg[i-1] == above_avg[i-2] if i > 1 else True:
                streak[i] = streak[i-1] + (1 if above_avg[i-1] else -1)
            else:
                streak[i] = 1 if above_avg[i-1] else -1
        
        new_columns[f'{stat}_streak'] = streak
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency metrics - how predictable is the player?
    """
    if df.empty:
        return df
    
    new_columns = {}
    
    for stat in ['PTS', 'PRA']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        shifted = pd.Series(np.roll(stat_values, 1).astype(float))
        shifted.iloc[0] = np.nan
        
        # Coefficient of variation over last 10 games (lower = more consistent)
        rolling_mean = shifted.rolling(10, min_periods=3).mean()
        rolling_std = shifted.rolling(10, min_periods=3).std()
        cv = (rolling_std / rolling_mean.replace(0, np.nan)).fillna(1)
        new_columns[f'{stat}_consistency'] = cv.values
        
        # Hit rate for common lines (how often over certain thresholds)
        for threshold in [10, 15, 20, 25]:
            hit_rate = shifted.rolling(10, min_periods=3).apply(
                lambda x: (x > threshold).mean(), raw=True
            )
            new_columns[f'{stat}_over_{threshold}_rate'] = hit_rate.values
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def add_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home/away indicator and split performance.
    """
    if df.empty:
        return df
    
    df['IS_HOME'] = (~df['MATCHUP'].str.contains('@', na=False)).astype(int)
    
    if 'PRA' not in df.columns and all(c in df.columns for c in ['PTS', 'REB', 'AST']):
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    
    new_columns = {}
    
    for stat in ['PTS', 'REB', 'AST', 'PRA', 'MIN']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        is_home = df['IS_HOME'].values
        
        home_avg = np.full(len(df), np.nan)
        away_avg = np.full(len(df), np.nan)
        home_sum, home_count = 0.0, 0
        away_sum, away_count = 0.0, 0
        
        for i in range(len(df)):
            if home_count > 0:
                home_avg[i] = home_sum / home_count
            if away_count > 0:
                away_avg[i] = away_sum / away_count
            
            if is_home[i]:
                home_sum += stat_values[i]
                home_count += 1
            else:
                away_sum += stat_values[i]
                away_count += 1
        
        home_avg = pd.Series(home_avg).ffill().values
        away_avg = pd.Series(away_avg).ffill().values
        
        new_columns[f'{stat}_home_avg'] = home_avg
        new_columns[f'{stat}_away_avg'] = away_avg
        
        # Home/away differential
        new_columns[f'{stat}_home_away_diff'] = home_avg - away_avg
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def add_rest_days_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rest days and back-to-back performance patterns.
    """
    if df.empty:
        return df
    
    game_dates = pd.to_datetime(df['GAME_DATE'])
    sort_idx = game_dates.argsort()
    df = df.iloc[sort_idx].reset_index(drop=True)
    game_dates = pd.to_datetime(df['GAME_DATE'])
    
    rest_days = game_dates.diff().dt.days.fillna(3).clip(0, 7)
    
    df = df.assign(
        REST_DAYS=rest_days,
        IS_B2B=(rest_days <= 1).astype(int),
        IS_WELL_RESTED=(rest_days >= 3).astype(int)
    )
    
    # B2B performance history
    if 'PTS' in df.columns:
        new_columns = {}
        for stat in ['PTS', 'PRA']:
            if stat not in df.columns:
                continue
            
            stat_values = df[stat].values.astype(float)
            is_b2b = df['IS_B2B'].values
            
            b2b_avg = np.full(len(df), np.nan)
            non_b2b_avg = np.full(len(df), np.nan)
            b2b_sum, b2b_count = 0.0, 0
            non_b2b_sum, non_b2b_count = 0.0, 0
            
            for i in range(len(df)):
                if b2b_count > 0:
                    b2b_avg[i] = b2b_sum / b2b_count
                if non_b2b_count > 0:
                    non_b2b_avg[i] = non_b2b_sum / non_b2b_count
                
                if is_b2b[i]:
                    b2b_sum += stat_values[i]
                    b2b_count += 1
                else:
                    non_b2b_sum += stat_values[i]
                    non_b2b_count += 1
            
            new_columns[f'{stat}_b2b_avg'] = pd.Series(b2b_avg).ffill().values
            new_columns[f'{stat}_non_b2b_avg'] = pd.Series(non_b2b_avg).ffill().values
            new_columns[f'{stat}_b2b_impact'] = (
                np.array(new_columns[f'{stat}_b2b_avg']) / 
                np.where(new_columns[f'{stat}_non_b2b_avg'] > 0, 
                        new_columns[f'{stat}_non_b2b_avg'], 1)
            )
        
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    return df


def add_day_of_week_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add day of week patterns (some players perform differently on certain days).
    """
    if df.empty:
        return df
    
    game_dates = pd.to_datetime(df['GAME_DATE'])
    df['DAY_OF_WEEK'] = game_dates.dt.dayofweek  # 0=Monday, 6=Sunday
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
    
    return df


def add_opponent_features(df: pd.DataFrame, defensive_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add opponent defensive ratings and matchup features.
    """
    if df.empty:
        return df
    
    def extract_opponent(matchup):
        if pd.isna(matchup):
            return None
        if '@' in matchup:
            return matchup.split('@')[-1].strip()
        elif 'vs.' in matchup:
            return matchup.split('vs.')[-1].strip()
        else:
            return matchup.split(' ')[-1].strip()
    
    df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)
    
    # Add defensive tier
    df['OPP_DEF_TIER'] = df['OPPONENT'].map(TEAM_DEFENSIVE_TIERS).fillna(3)
    
    # Add pace tier
    df['OPP_PACE_TIER'] = df['OPPONENT'].map(TEAM_PACE_TIERS).fillna(3)
    
    # If we have detailed defensive stats, merge them
    if defensive_stats is not None and not defensive_stats.empty:
        if 'TEAM_ABBREVIATION' in defensive_stats.columns:
            def_cols = ['TEAM_ABBREVIATION']
            for col in defensive_stats.columns:
                col_upper = col.upper()
                if any(x in col_upper for x in ['OPP', 'DEF', 'PACE', 'PTS']):
                    def_cols.append(col)
            
            opp_stats = defensive_stats[list(set(def_cols))].drop_duplicates(subset=['TEAM_ABBREVIATION'])
            opp_stats = opp_stats.rename(columns={'TEAM_ABBREVIATION': 'OPPONENT'})
            
            df = df.merge(opp_stats, on='OPPONENT', how='left')
    
    return df


def calculate_vs_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling performance vs different opponent tiers.
    """
    if df.empty or 'OPP_DEF_TIER' not in df.columns:
        return df
    
    new_columns = {}
    
    for stat in ['PTS', 'PRA']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        def_tiers = df['OPP_DEF_TIER'].values
        
        # Performance vs good defenses (tier 1-2)
        vs_good_def = np.full(len(df), np.nan)
        good_sum, good_count = 0.0, 0
        
        # Performance vs bad defenses (tier 4-5)
        vs_bad_def = np.full(len(df), np.nan)
        bad_sum, bad_count = 0.0, 0
        
        for i in range(len(df)):
            if good_count > 0:
                vs_good_def[i] = good_sum / good_count
            if bad_count > 0:
                vs_bad_def[i] = bad_sum / bad_count
            
            if def_tiers[i] <= 2:
                good_sum += stat_values[i]
                good_count += 1
            elif def_tiers[i] >= 4:
                bad_sum += stat_values[i]
                bad_count += 1
        
        new_columns[f'{stat}_vs_good_def'] = pd.Series(vs_good_def).ffill().values
        new_columns[f'{stat}_vs_bad_def'] = pd.Series(vs_bad_def).ffill().values
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_matchup_history_features(player_id: int, opponent_team_id: int) -> Dict:
    """
    Calculate extensive features based on player's history vs specific opponent.
    """
    features = {
        'vs_opp_games': 0,
        'vs_opp_pts_avg': np.nan,
        'vs_opp_reb_avg': np.nan,
        'vs_opp_ast_avg': np.nan,
        'vs_opp_pra_avg': np.nan,
        'vs_opp_pts_max': np.nan,
        'vs_opp_pts_min': np.nan,
        'vs_opp_pts_std': np.nan,
        'vs_opp_pra_max': np.nan,
        'vs_opp_pra_min': np.nan,
        'vs_opp_consistency': np.nan,
        'vs_opp_over_10_rate': np.nan,
        'vs_opp_over_15_rate': np.nan,
        'vs_opp_over_20_rate': np.nan,
    }
    
    history = get_matchup_history(player_id, opponent_team_id, num_games=15)
    
    if history.empty:
        return features
    
    pts = history['PTS'].values.astype(float)
    reb = history['REB'].values.astype(float)
    ast = history['AST'].values.astype(float)
    pra = pts + reb + ast
    
    features['vs_opp_games'] = len(history)
    features['vs_opp_pts_avg'] = float(pts.mean())
    features['vs_opp_reb_avg'] = float(reb.mean())
    features['vs_opp_ast_avg'] = float(ast.mean())
    features['vs_opp_pra_avg'] = float(pra.mean())
    features['vs_opp_pts_max'] = float(pts.max())
    features['vs_opp_pts_min'] = float(pts.min())
    features['vs_opp_pts_std'] = float(pts.std()) if len(pts) > 1 else np.nan
    features['vs_opp_pra_max'] = float(pra.max())
    features['vs_opp_pra_min'] = float(pra.min())
    
    if features['vs_opp_pts_avg'] > 0:
        features['vs_opp_consistency'] = features['vs_opp_pts_std'] / features['vs_opp_pts_avg']
    
    features['vs_opp_over_10_rate'] = float((pts > 10).mean())
    features['vs_opp_over_15_rate'] = float((pts > 15).mean())
    features['vs_opp_over_20_rate'] = float((pts > 20).mean())
    
    return features


def calculate_floor_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate floor/ceiling projections based on historical distribution.
    """
    if df.empty:
        return df
    
    new_columns = {}
    
    for stat in ['PTS', 'PRA']:
        if stat not in df.columns:
            continue
        
        stat_values = df[stat].values.astype(float)
        shifted = pd.Series(np.roll(stat_values, 1).astype(float))
        shifted.iloc[0] = np.nan
        
        # Floor (10th percentile of last 15 games)
        floor = shifted.rolling(15, min_periods=5).quantile(0.10)
        new_columns[f'{stat}_floor'] = floor.values
        
        # Ceiling (90th percentile of last 15 games)
        ceiling = shifted.rolling(15, min_periods=5).quantile(0.90)
        new_columns[f'{stat}_ceiling'] = ceiling.values
        
        # Most likely range (25th to 75th percentile)
        q25 = shifted.rolling(15, min_periods=5).quantile(0.25)
        q75 = shifted.rolling(15, min_periods=5).quantile(0.75)
        new_columns[f'{stat}_q25'] = q25.values
        new_columns[f'{stat}_q75'] = q75.values
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minutes-related features (minutes stability predicts scoring).
    """
    if df.empty or 'MIN' not in df.columns:
        return df
    
    min_values = df['MIN'].values.astype(float)
    shifted = pd.Series(np.roll(min_values, 1).astype(float))
    shifted.iloc[0] = np.nan
    
    new_columns = {}
    
    # Minutes stability (std over last 5 games)
    new_columns['MIN_stability'] = shifted.rolling(5, min_periods=2).std().values
    
    # Minutes trend (are minutes increasing or decreasing?)
    l3_min = shifted.rolling(3, min_periods=1).mean()
    l10_min = shifted.rolling(10, min_periods=3).mean()
    new_columns['MIN_trend'] = ((l3_min / l10_min.replace(0, np.nan)) - 1).values
    
    # Recent minutes vs season
    season_min = shifted.expanding().mean()
    new_columns['MIN_vs_season'] = ((shifted / season_min.replace(0, np.nan)) - 1).values
    
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def calculate_usage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate usage rate from available stats.
    """
    if df.empty:
        return df
    
    new_columns = {}
    
    # Usage proxy: (FGA + 0.44*FTA + TOV) / MIN * 48
    # Simplified version without TOV
    if all(c in df.columns for c in ['FGA', 'FTA', 'MIN']):
        fga = df['FGA'].values.astype(float)
        fta = df['FTA'].values.astype(float)
        minutes = df['MIN'].values.astype(float)
        
        usage = (fga + 0.44 * fta) / np.where(minutes > 0, minutes, 1) * 36
        shifted = np.roll(usage, 1).astype(float)
        shifted[0] = np.nan
        
        # Rolling usage
        usage_series = pd.Series(shifted)
        new_columns['USAGE_L5'] = usage_series.rolling(5, min_periods=1).mean().values
        new_columns['USAGE_season'] = usage_series.expanding().mean().values
    
    # Points per minute (efficiency proxy)
    if all(c in df.columns for c in ['PTS', 'MIN']):
        pts = df['PTS'].values.astype(float)
        minutes = df['MIN'].values.astype(float)
        
        ppm = pts / np.where(minutes > 0, minutes, 1)
        shifted_ppm = np.roll(ppm, 1).astype(float)
        shifted_ppm[0] = np.nan
        
        ppm_series = pd.Series(shifted_ppm)
        new_columns['PTS_per_MIN_L5'] = ppm_series.rolling(5, min_periods=1).mean().values
    
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def estimate_first_half_stats(full_game_prediction: float, stat_type: str = 'points') -> float:
    """Estimate first half stats based on full game prediction."""
    ratio = FIRST_HALF_RATIOS.get(stat_type, 0.48)
    return full_game_prediction * ratio


def estimate_period_stats(full_game_prediction: float, stat_type: str = 'points', period: str = '1h') -> float:
    """Estimate period stats (1H or 1Q) based on full game prediction."""
    if period == '1q':
        ratio = FIRST_QUARTER_RATIOS.get(stat_type, 0.24)
    else:
        ratio = FIRST_HALF_RATIOS.get(stat_type, 0.48)
    return full_game_prediction * ratio


def create_prediction_features(
    player_name: str,
    opponent_abbrev: str,
    is_home: bool = True,
    prop_line: float = None,
    prop_type: str = 'points'
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """
    Create all features needed for a prediction with EXTREME accuracy mode.
    Returns feature dict and recent game log for context.
    """
    player = search_player(player_name)
    if not player:
        return None, None
    
    player_id = player['id']
    
    game_logs = get_player_multi_season_logs(player_id)
    if game_logs.empty:
        return None, None
    
    # Apply all feature engineering
    df = calculate_rolling_stats(game_logs)
    df = calculate_weighted_averages(df)
    df = calculate_momentum_features(df)
    df = calculate_consistency_score(df)
    df = add_home_away_features(df)
    df = add_rest_days_feature(df)
    df = add_day_of_week_features(df)
    
    # Get defensive stats
    def_stats = get_team_defensive_stats()
    if not def_stats.empty:
        df = add_opponent_features(df, def_stats)
    else:
        df = add_opponent_features(df)
    
    df = calculate_vs_opponent_features(df)
    df = calculate_floor_ceiling(df)
    df = calculate_minutes_features(df)
    df = calculate_usage_proxy(df)
    
    # Defragment the DataFrame after all feature additions
    df = df.copy()
    
    if len(df) == 0:
        return None, None
    
    latest = df.iloc[-1].to_dict()
    
    # Add matchup history
    opp_team = get_team_by_abbreviation(opponent_abbrev)
    if opp_team:
        matchup_features = calculate_matchup_history_features(player_id, opp_team['id'])
        latest.update(matchup_features)
        
        # Add opponent defensive tier for current matchup
        latest['CURRENT_OPP_DEF_TIER'] = TEAM_DEFENSIVE_TIERS.get(opponent_abbrev, 3)
        latest['CURRENT_OPP_PACE_TIER'] = TEAM_PACE_TIERS.get(opponent_abbrev, 3)
    
    # Override with current game info
    latest['IS_HOME'] = 1 if is_home else 0
    latest['OPPONENT'] = opponent_abbrev.upper()
    
    # Add prop line as feature
    if prop_line is not None:
        latest['PROP_LINE'] = prop_line
        latest['PROP_TYPE'] = prop_type
    
    # Add player info
    latest['PLAYER_ID'] = player_id
    latest['PLAYER_NAME'] = player['full_name']
    
    # Calculate composite confidence features
    latest = _calculate_composite_features(latest, prop_line, prop_type)
    
    # Get recent games
    cols = ['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'MIN']
    available_cols = [c for c in cols if c in df.columns]
    recent_games = df.tail(10)[available_cols].copy()
    
    if all(c in recent_games.columns for c in ['PTS', 'REB', 'AST']):
        recent_games['PRA'] = recent_games['PTS'] + recent_games['REB'] + recent_games['AST']
    
    return latest, recent_games


def _safe_float(val, default=0):
    """Safely convert a value to float, handling None, NaN, and non-numeric types."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _calculate_composite_features(features: Dict, prop_line: float, prop_type: str) -> Dict:
    """
    Calculate composite features that combine multiple signals.
    """
    # Properly map all prop types to their stat columns
    stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pra': 'PRA'}
    stat_col = stat_map.get(prop_type, 'PTS')
    ratio = FIRST_HALF_RATIOS.get(prop_type, 0.48)
    
    # Estimate 1H values for comparison - use NaN-safe helper
    l5_avg = _safe_float(features.get(f'{stat_col}_L5_avg'))
    l10_avg = _safe_float(features.get(f'{stat_col}_L10_avg'))
    season_avg = _safe_float(features.get(f'{stat_col}_season_avg'))
    ewm_5 = _safe_float(features.get(f'{stat_col}_ewm_5'))
    
    # Weighted prediction (recent weighted more)
    weighted_pred = (
        ewm_5 * 0.35 +
        l5_avg * 0.30 +
        l10_avg * 0.20 +
        season_avg * 0.15
    )
    features['WEIGHTED_PREDICTION'] = weighted_pred
    features['WEIGHTED_PREDICTION_1H'] = weighted_pred * ratio
    
    # Edge vs prop line
    if prop_line and prop_line > 0:
        pred_1h = weighted_pred * ratio
        features['EDGE_VS_LINE'] = pred_1h - prop_line
        features['EDGE_PCT'] = (pred_1h - prop_line) / prop_line * 100
    
    # Consistency score (lower = more consistent)
    std = _safe_float(features.get(f'{stat_col}_L10_std'))
    mean = _safe_float(features.get(f'{stat_col}_L10_avg'), default=1)
    features['CONSISTENCY_SCORE'] = std / mean if mean > 0 else 1
    
    # Momentum score
    momentum = _safe_float(features.get(f'{stat_col}_momentum'))
    features['MOMENTUM_SCORE'] = momentum
    
    # Floor/ceiling for prop comparison
    floor = _safe_float(features.get(f'{stat_col}_floor'))
    ceiling = _safe_float(features.get(f'{stat_col}_ceiling'))
    features['FLOOR_1H'] = floor * ratio
    features['CEILING_1H'] = ceiling * ratio
    
    return features


def _apply_per_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all per-player feature engineering to a single player's game log.
    
    Rolling stats, expanding means, rest days, streaks, etc. MUST be computed
    per-player to prevent cross-player data leakage in training.
    
    Assumes row-level features (opponent tiers, day-of-week) are already present.
    """
    if player_df.empty or len(player_df) < 2:
        return player_df
    
    player_df = calculate_rolling_stats(player_df)
    player_df = calculate_weighted_averages(player_df)
    player_df = calculate_momentum_features(player_df)
    player_df = calculate_consistency_score(player_df)
    player_df = add_home_away_features(player_df)
    player_df = add_rest_days_feature(player_df)
    player_df = calculate_vs_opponent_features(player_df)
    player_df = calculate_floor_ceiling(player_df)
    player_df = calculate_minutes_features(player_df)
    player_df = calculate_usage_proxy(player_df)
    
    return player_df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare training data with all features for EXTREME accuracy.
    
    CRITICAL: Per-player feature engineering prevents cross-player data leakage.
    Rolling averages, expanding stats, rest days, streaks, etc. are all computed
    within each player's own game history, NOT across the combined dataset.
    """
    # ── Step 1: Row-level features (safe to apply globally) ──
    # These don't use rolling/expanding windows, so cross-player is fine
    df = add_day_of_week_features(df)
    
    def_stats = get_team_defensive_stats()
    if not def_stats.empty:
        df = add_opponent_features(df, def_stats)
    else:
        df = add_opponent_features(df)
    
    # ── Step 2: Per-player feature engineering ──
    # Rolling stats, expanding means, rest days, etc. MUST be per-player
    if 'PLAYER_ID' in df.columns and df['PLAYER_ID'].nunique() > 1:
        print("  Applying per-player feature engineering (prevents cross-player contamination)...")
        player_groups = []
        player_ids = df['PLAYER_ID'].unique()
        
        for i, pid in enumerate(player_ids):
            player_df = df[df['PLAYER_ID'] == pid].copy()
            if len(player_df) >= 2:
                player_df = _apply_per_player_features(player_df)
                player_groups.append(player_df)
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(player_ids)} players...")
        
        if player_groups:
            df = pd.concat(player_groups, ignore_index=True)
        print(f"  Completed: {len(player_ids)} players, {len(df)} game records")
    else:
        # Single player or no PLAYER_ID column (prediction path / legacy)
        df = calculate_rolling_stats(df)
        df = calculate_weighted_averages(df)
        df = calculate_momentum_features(df)
        df = calculate_consistency_score(df)
        df = add_home_away_features(df)
        df = add_rest_days_feature(df)
        df = calculate_vs_opponent_features(df)
        df = calculate_floor_ceiling(df)
        df = calculate_minutes_features(df)
        df = calculate_usage_proxy(df)
    
    # Define all feature columns
    feature_cols = []
    
    # Rolling stat features
    for stat in ['PTS', 'REB', 'AST', 'PRA', 'MIN', 'FG_PCT']:
        for window in ROLLING_WINDOWS:
            feature_cols.extend([
                f'{stat}_L{window}_avg',
                f'{stat}_L{window}_std',
                f'{stat}_L{window}_median',
            ])
        feature_cols.extend([
            f'{stat}_season_avg',
            f'{stat}_season_std',
            f'{stat}_ewm_5',
            f'{stat}_ewm_10'
        ])
    
    # Momentum features
    for stat in ['PTS', 'PRA', 'MIN']:
        feature_cols.extend([
            f'{stat}_momentum',
            f'{stat}_last_change',
            f'{stat}_streak'
        ])
    
    # Consistency features
    for stat in ['PTS', 'PRA']:
        feature_cols.extend([
            f'{stat}_consistency',
            f'{stat}_over_10_rate',
            f'{stat}_over_15_rate',
            f'{stat}_over_20_rate'
        ])
    
    # Location features
    feature_cols.extend(['IS_HOME'])
    for stat in ['PTS', 'PRA', 'MIN']:
        feature_cols.extend([
            f'{stat}_home_avg',
            f'{stat}_away_avg',
            f'{stat}_home_away_diff'
        ])
    
    # Rest features
    feature_cols.extend(['REST_DAYS', 'IS_B2B', 'IS_WELL_RESTED'])
    for stat in ['PTS', 'PRA']:
        feature_cols.extend([
            f'{stat}_b2b_avg',
            f'{stat}_non_b2b_avg',
            f'{stat}_b2b_impact'
        ])
    
    # Day features
    feature_cols.extend(['DAY_OF_WEEK', 'IS_WEEKEND'])
    
    # Opponent features
    feature_cols.extend(['OPP_DEF_TIER', 'OPP_PACE_TIER'])
    for stat in ['PTS', 'PRA']:
        feature_cols.extend([
            f'{stat}_vs_good_def',
            f'{stat}_vs_bad_def'
        ])
    
    # Floor/ceiling features
    for stat in ['PTS', 'PRA']:
        feature_cols.extend([
            f'{stat}_floor',
            f'{stat}_ceiling',
            f'{stat}_q25',
            f'{stat}_q75'
        ])
    
    # Minutes features
    feature_cols.extend(['MIN_stability', 'MIN_trend', 'MIN_vs_season'])
    
    # Usage features
    feature_cols.extend(['USAGE_L5', 'USAGE_season', 'PTS_per_MIN_L5'])
    
    # Keep only features that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Drop rows with too many NaN values
    if len(feature_cols) >= 10:
        df = df.dropna(subset=feature_cols[:10], how='all')
    
    return df, feature_cols


def get_feature_importance_context(feature_values: Dict, feature_importances: Dict) -> List[Tuple[str, float, str, any]]:
    """Get top features contributing to prediction with context."""
    descriptions = {
        # Points features
        'PTS_L3_avg': 'Points avg last 3 games',
        'PTS_L5_avg': 'Points avg last 5 games',
        'PTS_L10_avg': 'Points avg last 10 games',
        'PTS_ewm_5': 'Weighted recent points',
        'PTS_season_avg': 'Season points average',
        'PTS_momentum': 'Points momentum trend',
        'PTS_consistency': 'Scoring consistency',
        'PTS_vs_good_def': 'Points vs good defenses',
        'PTS_floor': 'Expected points floor',
        'PTS_ceiling': 'Expected points ceiling',
        
        # Rebounds features
        'REB_L3_avg': 'Rebounds avg last 3 games',
        'REB_L5_avg': 'Rebounds avg last 5 games',
        'REB_L10_avg': 'Rebounds avg last 10 games',
        'REB_ewm_5': 'Weighted recent rebounds',
        'REB_season_avg': 'Season rebounds average',
        'REB_momentum': 'Rebounds momentum trend',
        'REB_consistency': 'Rebounding consistency',
        'REB_floor': 'Expected rebounds floor',
        'REB_ceiling': 'Expected rebounds ceiling',
        'vs_opp_reb_avg': 'Avg rebounds vs opponent',
        
        # Assists features
        'AST_L3_avg': 'Assists avg last 3 games',
        'AST_L5_avg': 'Assists avg last 5 games',
        'AST_L10_avg': 'Assists avg last 10 games',
        'AST_ewm_5': 'Weighted recent assists',
        'AST_season_avg': 'Season assists average',
        'AST_momentum': 'Assists momentum trend',
        'AST_consistency': 'Playmaking consistency',
        'AST_floor': 'Expected assists floor',
        'AST_ceiling': 'Expected assists ceiling',
        'vs_opp_ast_avg': 'Avg assists vs opponent',
        
        # PRA features
        'PRA_L5_avg': 'PRA avg last 5 games',
        'PRA_ewm_5': 'Weighted recent PRA',
        'PRA_season_avg': 'Season PRA average',
        'PRA_consistency': 'PRA consistency',
        
        # Game context
        'IS_HOME': 'Home game advantage',
        'REST_DAYS': 'Days of rest',
        'IS_B2B': 'Back-to-back game',
        'vs_opp_pts_avg': 'Avg points vs opponent',
        'vs_opp_pra_avg': 'Avg PRA vs opponent',
        'MIN_L5_avg': 'Minutes avg last 5',
        'OPP_DEF_TIER': 'Opponent defense rating',
        'USAGE_L5': 'Recent usage rate',
    }
    
    sorted_features = sorted(feature_importances.items(), key=lambda x: -x[1])[:15]
    
    results = []
    for feature, importance in sorted_features:
        value = feature_values.get(feature, 'N/A')
        desc = descriptions.get(feature, feature.replace('_', ' ').title())
        results.append((feature, importance, desc, value))
    
    return results


if __name__ == "__main__":
    print("Testing EXTREME accuracy feature engineering...")
    
    features, recent = create_prediction_features(
        player_name="LeBron James",
        opponent_abbrev="GSW",
        is_home=True,
        prop_line=12.5,
        prop_type="points"
    )
    
    if features:
        print(f"\nFeatures for {features.get('PLAYER_NAME', 'Unknown')}:")
        print(f"  Weighted Prediction: {features.get('WEIGHTED_PREDICTION', 'N/A'):.1f}")
        print(f"  Weighted 1H: {features.get('WEIGHTED_PREDICTION_1H', 'N/A'):.1f}")
        print(f"  Edge vs Line: {features.get('EDGE_VS_LINE', 'N/A'):.2f}")
        print(f"  Momentum: {features.get('MOMENTUM_SCORE', 'N/A'):.3f}")
        print(f"  Consistency: {features.get('CONSISTENCY_SCORE', 'N/A'):.3f}")
        print(f"  Floor 1H: {features.get('FLOOR_1H', 'N/A'):.1f}")
        print(f"  Ceiling 1H: {features.get('CEILING_1H', 'N/A'):.1f}")
