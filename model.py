"""
Ensemble Model for NBA 1H Predictions
Primary: LightGBM, Secondary: XGBoost, CatBoost

Performance Optimized:
- Model caching (singleton pattern)
- Lazy model loading
- Vectorized operations
- Reduced memory copies
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from functools import lru_cache

# Suppress sklearn feature names warning (happens when predicting with arrays)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from config import (
    MODEL_CONFIG, ENSEMBLE_WEIGHTS, MODELS_DIR, FIRST_HALF_RATIOS, FIRST_QUARTER_RATIOS,
    TEAM_PACE_TIERS, TEAM_DEFENSIVE_TIERS, LOCK_THRESHOLDS, LOCK_THRESHOLDS_1Q, Q1_ADJUSTMENTS,
    MIN_EDGE_REQUIREMENTS, MIN_GAMES_FOR_CONFIDENCE, SUSPICIOUS_EDGE_THRESHOLDS,
    PLAYER_SCORING_STYLES, Q1_RATIO_ADJUSTMENTS,
    STAT_VARIANCE, STAT_MIN_EDGE_PCT, STAT_CONSISTENCY, POSITION_STAT_BONUSES,
    EARLY_REBOUNDERS, EARLY_PLAYMAKERS, STAT_LINE_VOLATILITY,
    HIGH_VARIANCE_STARS, ROLE_EXPANDING_PLAYERS, RECENTLY_TRADED_PLAYERS,
    UNDER_MIN_EDGE, UNDER_VOLATILITY_PENALTY, ROLE_CHANGE_THRESHOLD, ROLE_CHANGE_RECENT_WEIGHT
)

# ============================================================================
# Global Model Cache (Singleton Pattern)
# ============================================================================
_model_cache: Dict[str, 'NBA1HEnsembleModel'] = {}


def get_cached_model(target: str = 'points') -> 'NBA1HEnsembleModel':
    """Get or create cached model instance"""
    if target not in _model_cache:
        model = NBA1HEnsembleModel(target=target)
        model._load_model()  # Try to load trained model
        _model_cache[target] = model
    return _model_cache[target]


class NBA1HEnsembleModel:
    """
    Ensemble model for predicting NBA 1st half player stats.
    Combines LightGBM, XGBoost, and CatBoost with weighted averaging.
    
    Performance Optimizations:
    - Lazy model initialization
    - Pre-computed feature importance
    - Vectorized predictions
    """
    
    __slots__ = ['target', 'target_col', 'models', 'scaler', 'feature_cols', 
                 'is_trained', 'feature_importances', '_models_initialized', '_catboost_available']
    
    def __init__(self, target: str = 'points'):
        """Initialize ensemble model with lazy loading"""
        self.target = target
        # Map target to column name
        target_col_map = {
            'points': 'PTS',
            'rebounds': 'REB', 
            'assists': 'AST',
            'pra': 'PRA'
        }
        self.target_col = target_col_map.get(target, 'PTS')
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.is_trained = False
        self.feature_importances = {}
        self._models_initialized = False
        self._catboost_available = True
        
        # Ensure model directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def _init_models(self):
        """Lazy initialize all ensemble models"""
        if self._models_initialized:
            return
        
        self.models['lightgbm'] = lgb.LGBMRegressor(**MODEL_CONFIG['lightgbm'])
        self.models['xgboost'] = xgb.XGBRegressor(**MODEL_CONFIG['xgboost'])
        
        # Try to add CatBoost (may fail with newer sklearn versions)
        try:
            self.models['catboost'] = CatBoostRegressor(**MODEL_CONFIG['catboost'])
        except Exception as e:
            print(f"CatBoost not available: {e}")
            # Adjust weights to exclude catboost
            self._catboost_available = False
        else:
            self._catboost_available = True
        
        self._models_initialized = True
    
    def train(self, df: pd.DataFrame, feature_cols: List[str] = None) -> Dict:
        """Train the ensemble model with optimized operations"""
        from feature_engineering import prepare_training_data
        
        # Initialize models
        self._init_models()
        
        # Prepare data
        if feature_cols is None:
            df, feature_cols = prepare_training_data(df)
        
        self.feature_cols = feature_cols
        
        # Ensure target exists
        if self.target_col not in df.columns:
            if self.target == 'pra':
                df = df.assign(PRA=df['PTS'] + df['REB'] + df['AST'])
            else:
                raise ValueError(f"Target column {self.target_col} not found")
        
        # Drop NaN rows - use subset of columns for efficiency
        valid_cols = self.feature_cols + [self.target_col]
        df_clean = df[valid_cols].dropna()
        
        if len(df_clean) < 100:
            raise ValueError(f"Not enough training data: {len(df_clean)} samples")
        
        # Extract arrays (avoid repeated DataFrame operations)
        X = df_clean[self.feature_cols].values
        y = df_clean[self.target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        metrics = {
            'n_samples': len(X),
            'models': {}
        }
        
        # Train each model
        failed_models = []
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # CatBoost doesn't work with sklearn 1.8+ cross_val_score
                # Train it directly without sklearn CV wrapper
                if name == 'catboost':
                    # Manual cross-validation for CatBoost
                    cv_mae_scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        # Create fresh CatBoost model for each fold
                        from catboost import CatBoostRegressor
                        fold_model = CatBoostRegressor(**MODEL_CONFIG['catboost'])
                        fold_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                        y_val_pred = fold_model.predict(X_val)
                        cv_mae_scores.append(mean_absolute_error(y_val, y_val_pred))
                    cv_scores_mean = np.mean(cv_mae_scores)
                    cv_scores_std = np.std(cv_mae_scores)
                    # Final fit on ALL data (no early stopping for final model)
                    # This ensures the model sees all training data
                    model = CatBoostRegressor(
                        iterations=1000,
                        depth=8,
                        learning_rate=0.03,
                        l2_leaf_reg=3,
                        random_seed=42,
                        verbose=False,
                        loss_function="MAE"
                    )
                    model.fit(X_scaled, y, verbose=False)
                    self.models['catboost'] = model
                else:
                    # Standard sklearn cross-validation for LightGBM and XGBoost
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                    cv_scores_mean = -cv_scores.mean()
                    cv_scores_std = cv_scores.std()
                    # Fit on full data
                    model.fit(X_scaled, y)
                
                # Get predictions for evaluation
                y_pred = model.predict(X_scaled)
                
                metrics['models'][name] = {
                    'cv_mae': cv_scores_mean,
                    'cv_std': cv_scores_std,
                    'train_mae': mean_absolute_error(y, y_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'train_r2': r2_score(y, y_pred)
                }
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances[name] = dict(zip(
                        self.feature_cols, 
                        model.feature_importances_
                    ))
                    
            except Exception as e:
                print(f"  Warning: {name} failed - {e}")
                failed_models.append(name)
                continue
        
        # Remove failed models from ensemble
        for name in failed_models:
            if name in self.models:
                del self.models[name]
                print(f"  Removed {name} from ensemble")
        
        self.is_trained = True
        self._save_model()
        
        return metrics
    
    def predict(self, features: Dict, period: str = '1h') -> Tuple[float, Dict]:
        """
        Make prediction for a single game.
        Uses STATS-BASED weighted average of player's actual performance.
        
        Args:
            features: Dict of feature values
            period: '1h' for first half or '1q' for first quarter
        """
        stat_col = self.target_col
        stat_type = self.target
        ratio = get_period_ratio(stat_type, period)
        
        # Get player's actual averages
        l3_avg = features.get(f'{stat_col}_L3_avg')
        l5_avg = features.get(f'{stat_col}_L5_avg')
        l10_avg = features.get(f'{stat_col}_L10_avg')
        season_avg = features.get(f'{stat_col}_season_avg')
        ewm_5 = features.get(f'{stat_col}_ewm_5')  # Exponentially weighted mean
        
        # =====================================================================
        # STATS-BASED PREDICTION (Primary Method)
        # Weighted average: Recent games weighted more heavily
        # =====================================================================
        weights = []
        values = []
        
        # L3 (most recent) - highest weight
        if l3_avg and l3_avg > 0:
            weights.append(0.30)
            values.append(l3_avg)
        
        # L5 - high weight
        if l5_avg and l5_avg > 0:
            weights.append(0.30)
            values.append(l5_avg)
        
        # EWM (trend-adjusted) - medium weight
        if ewm_5 and ewm_5 > 0:
            weights.append(0.20)
            values.append(ewm_5)
        
        # L10 - medium weight
        if l10_avg and l10_avg > 0:
            weights.append(0.15)
            values.append(l10_avg)
        
        # Season - lower weight (baseline)
        if season_avg and season_avg > 0:
            weights.append(0.15)
            values.append(season_avg)
        
        # Calculate stats-based prediction
        if weights and values:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            stats_pred = sum(v * w for v, w in zip(values, normalized_weights))
        else:
            # Fallback if no stats available
            stats_pred = 0
        
        # =====================================================================
        # ML MODEL PREDICTION (Ensemble of trained models)
        # =====================================================================
        ml_predictions = {}
        ml_pred = 0
        ml_available = False
        
        if self.is_trained or self._load_model():
            try:
                X = np.zeros((1, len(self.feature_cols)))
                for i, col in enumerate(self.feature_cols):
                    X[0, i] = features.get(col, 0) or 0
                
                X_scaled = self.scaler.transform(X)
                
                for name, model in self.models.items():
                    ml_predictions[name] = float(model.predict(X_scaled)[0])
                
                available_weights = {k: v for k, v in ENSEMBLE_WEIGHTS.items() if k in ml_predictions}
                weight_sum = sum(available_weights.values())
                if weight_sum > 0:
                    norm_weights = {k: v/weight_sum for k, v in available_weights.items()}
                    ml_pred = sum(ml_predictions[n] * norm_weights.get(n, 0) for n in ml_predictions)
                    ml_available = ml_pred > 0
            except Exception:
                pass
        
        # =====================================================================
        # CHECK FOR REAL PERIOD DATA
        # =====================================================================
        player_name = features.get('PLAYER_NAME', '')
        real_period_avg, has_real_data = get_player_period_average(player_name, stat_type, period)
        real_ratio, has_real_ratio = get_player_period_ratio(player_name, stat_type, period)
        
        # Use real ratio if available
        if has_real_ratio:
            ratio = real_ratio
        
        # =====================================================================
        # FINAL PREDICTION
        # Priority: Real period data > ML+Stats blend > Stats estimation
        # Real 1H/1Q averages are the most accurate predictor.
        # ML models are used as a VALIDATION signal in the lock score,
        # not to distort the core prediction.
        # =====================================================================
        if has_real_data and real_period_avg is not None:
            # USE REAL PERIOD AVERAGE - proven most accurate
            period_pred = real_period_avg
            full_game_pred = period_pred / ratio if ratio > 0 else period_pred * 2
            data_source = "REAL"
        else:
            # No real period data - use stats-based estimation
            full_game_pred = stats_pred
            
            # Apply contextual adjustments (only on estimated path)
            is_home = features.get('IS_HOME', 0)
            is_b2b = features.get('IS_B2B', 0)
            
            if is_home:
                full_game_pred *= 1.02
            if is_b2b:
                full_game_pred *= 0.95
            
            period_pred = full_game_pred * ratio
            data_source = "ESTIMATED"
        
        details = {
            'full_game_prediction': full_game_pred,
            'first_half_prediction': period_pred,
            'model_predictions': ml_predictions,
            'stats_based_prediction': stats_pred,
            'ensemble_weights': ENSEMBLE_WEIGHTS,
            'first_half_ratio': ratio,
            'period': period,
            'prediction_method': 'stats_weighted_avg',
            'data_source': data_source,
            'has_real_period_data': has_real_data,
            'stats_used': {
                'L3_avg': l3_avg,
                'L5_avg': l5_avg,
                'L10_avg': l10_avg,
                'season_avg': season_avg,
                'ewm_5': ewm_5
            }
        }
        
        return period_pred, details
    
    def predict_over_under(
        self, 
        features: Dict, 
        prop_line: float,
        confidence_threshold: float = 0.5,
        recent_games: list = None,
        period: str = '1h'
    ) -> Dict:
        """Predict over/under for a prop bet with lock rating"""
        pred_1h, details = self.predict(features, period=period)
        
        stat_col = self.target_col
        ratio = get_period_ratio(self.target, period)
        
        full_game_pred = details['full_game_prediction']
        
        # Handle case where we have no data at all
        if full_game_pred <= 0:
            details['prediction_warning'] = f"⚠️ INSUFFICIENT DATA: No stats available for this player"
            # Use prop line as fallback estimate
            pred_1h = prop_line
            details['full_game_prediction'] = prop_line / ratio
            details['first_half_prediction'] = prop_line
        
        difference = pred_1h - prop_line
        pct_diff = (difference / prop_line) * 100 if prop_line > 0 else 0
        abs_diff = abs(difference)
        is_over = difference > 0
        
        # Calculate lock score (0-100) based on multiple factors
        has_real_data = details.get('has_real_period_data', False)
        lock_score, lock_factors = self._calculate_lock_score(features, details, prop_line, difference, recent_games, has_real_data, period=period)
        
        # Apply 1Q-specific adjustments if this is a 1Q prediction
        if period == '1q':
            features['games_played'] = len(recent_games) if recent_games else 0
            features['_prop_type'] = self.target  # Pass prop type for correct stat lookup
            lock_score, lock_factors = apply_1q_adjustments(
                lock_score, lock_factors, features, prop_line, difference, period,
                player_name=features.get('PLAYER_NAME'),
                recent_games=recent_games,
                has_real_period_data=has_real_data
            )
        
        # Determine pick and confidence based on lock score - USE PERIOD-SPECIFIC THRESHOLDS
        thresholds = get_lock_thresholds(period)
        period_label = "1Q" if period == '1q' else "1H"
        
        if lock_score >= thresholds['lock']:
            pick = "🔒 LOCK " + ("OVER" if is_over else "UNDER")
            confidence = "LOCK"
            confidence_desc = f"Extremely high confidence for {period_label}"
        elif lock_score >= thresholds['strong']:
            pick = "🔥 STRONG " + ("OVER" if is_over else "UNDER")
            confidence = "HIGH"
            confidence_desc = f"High confidence for {period_label}"
        elif lock_score >= thresholds['playable']:
            pick = "✅ " + ("OVER" if is_over else "UNDER")
            confidence = "MEDIUM"
            confidence_desc = f"Moderate confidence for {period_label}"
        elif lock_score >= thresholds['lean']:
            pick = "⚠️ LEAN " + ("OVER" if is_over else "UNDER")
            confidence = "LOW"
            confidence_desc = f"Low confidence for {period_label} - proceed with caution"
        else:
            pick = "❓ SKIP"
            confidence = "AVOID"
            confidence_desc = f"Too risky for {period_label} - insufficient edge"
        
        # Add 1Q warning if applicable
        if period == '1q' and confidence not in ['LOCK', 'AVOID']:
            confidence_desc += " (1Q requires higher standards)"
        
        # Generate reasoning
        reasons = self._generate_reasoning(features, details, prop_line, difference, period=period)
        
        # Add warning if prediction was capped
        if details.get('prediction_capped'):
            reasons.insert(0, details.get('prediction_warning', '⚠️ Prediction adjusted for reliability'))
        
        result = {
            'pick': pick,
            'confidence': confidence,
            'confidence_desc': confidence_desc,
            'lock_score': lock_score,
            'lock_factors': lock_factors,
            'predicted_1h': round(pred_1h, 1),
            'prop_line': prop_line,
            'difference': round(difference, 1),
            'pct_difference': round(pct_diff, 1),
            'full_game_prediction': round(details['full_game_prediction'], 1),
            'reasons': reasons,
            'model_breakdown': {
                name: round(pred * details['first_half_ratio'], 1)
                for name, pred in details['model_predictions'].items()
            },
            'data_source': details.get('data_source', 'ESTIMATED'),
            'has_real_period_data': details.get('has_real_period_data', False)
        }
        
        # Include warning info if prediction was capped
        if details.get('prediction_capped'):
            result['prediction_warning'] = details.get('prediction_warning')
            result['original_prediction'] = round(details.get('original_prediction', pred_1h), 1)
            result['original_full_game'] = round(details.get('original_full_game', details['full_game_prediction']), 1)
        
        return result
    
    def _calculate_lock_score(
        self,
        features: Dict,
        details: Dict,
        prop_line: float,
        difference: float,
        recent_games: list = None,
        has_real_period_data: bool = False,
        period: str = '1h'
    ) -> Tuple[int, List[Dict]]:
        """
        EXTREME ACCURACY lock score calculation.
        Uses 12+ factors for maximum precision.
        Returns score and list of factors with their contributions.
        """
        score = 50  # Start at neutral
        factors = []
        stat_col = self.target_col
        # Use period-aware ratio (1Q ~0.24 vs 1H ~0.48)
        ratio = get_period_ratio(self.target, period)
        is_over = difference > 0
        abs_diff = abs(difference)
        
        # =================================================================
        # FACTOR 0A: REAL PERIOD DATA BOOST (NEW - CRITICAL!)
        # Having actual 1Q/1H data dramatically increases prediction accuracy
        # =================================================================
        if has_real_period_data:
            score += 20
            factors.append({
                'name': '📊 Real Period Data', 
                'score': '+20', 
                'desc': 'Using actual 1Q/1H game data - high accuracy'
            })
        
        # =================================================================
        # FACTOR 0B: Prediction Reliability Check (MAJOR PENALTY)
        # =================================================================
        if details.get('prediction_capped'):
            score -= 25
            original = details.get('original_full_game', 0)
            actual_avg = features.get(f'{stat_col}_L5_avg') or features.get(f'{stat_col}_season_avg', 0)
            factors.append({
                'name': '🚨 Unreliable Model', 
                'score': '-25', 
                'desc': f'Model predicted {original:.0f} but player averages {actual_avg:.0f} - capped'
            })
        
        # =================================================================
        # FACTOR 1: Edge Size (0-20 points)
        # =================================================================
        if abs_diff >= 3.5:
            score += 20
            factors.append({'name': '🎯 Huge Edge', 'score': '+20', 'desc': f'{abs_diff:.1f} pts from line - elite edge'})
        elif abs_diff >= 2.5:
            score += 14
            factors.append({'name': '🎯 Large Edge', 'score': '+14', 'desc': f'{abs_diff:.1f} pts from line - strong edge'})
        elif abs_diff >= 1.5:
            score += 8
            factors.append({'name': '🎯 Good Edge', 'score': '+8', 'desc': f'{abs_diff:.1f} pts from line - solid edge'})
        elif abs_diff >= 0.8:
            score += 4
            factors.append({'name': '🎯 Small Edge', 'score': '+4', 'desc': f'{abs_diff:.1f} pts from line - minor edge'})
        else:
            score -= 8
            factors.append({'name': '⚠️ Tiny Edge', 'score': '-8', 'desc': f'Only {abs_diff:.1f} pts from line - risky'})
        
        # =================================================================
        # FACTOR 2: Model Consensus (0-12 points)
        # =================================================================
        model_preds = details.get('model_predictions', {})
        if model_preds:
            preds_1h = [p * details['first_half_ratio'] for p in model_preds.values()]
            all_agree_over = all(p > prop_line for p in preds_1h)
            all_agree_under = all(p < prop_line for p in preds_1h)
            
            # Calculate model spread (how close are the models)
            model_spread = max(preds_1h) - min(preds_1h)
            
            if (all_agree_over and is_over) or (all_agree_under and not is_over):
                if model_spread < 1.5:
                    score += 12
                    factors.append({'name': '🤝 Strong Consensus', 'score': '+12', 'desc': 'All models agree tightly'})
                else:
                    score += 8
                    factors.append({'name': '🤝 Consensus', 'score': '+8', 'desc': 'All models agree on direction'})
            elif not all_agree_over and not all_agree_under:
                score -= 10
                factors.append({'name': '❌ Model Split', 'score': '-10', 'desc': 'Models disagree on direction'})
        
        # =================================================================
        # FACTOR 3: Weighted Recent Form (0-15 points)
        # =================================================================
        ewm_5 = features.get(f'{stat_col}_ewm_5')
        if ewm_5 is not None:
            ewm_1h = ewm_5 * ratio
            if is_over and ewm_1h > prop_line * 1.15:
                score += 15
                factors.append({'name': '🔥 On Fire', 'score': '+15', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) way above line'})
            elif not is_over and ewm_1h < prop_line * 0.85:
                score += 15
                factors.append({'name': '❄️ Ice Cold', 'score': '+15', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) way below line'})
            elif is_over and ewm_1h > prop_line * 1.05:
                score += 8
                factors.append({'name': '📈 Trending Up', 'score': '+8', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) above line'})
            elif not is_over and ewm_1h < prop_line * 0.95:
                score += 8
                factors.append({'name': '📉 Trending Down', 'score': '+8', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) below line'})
            elif (is_over and ewm_1h < prop_line * 0.95) or (not is_over and ewm_1h > prop_line * 1.05):
                score -= 10
                factors.append({'name': '⚠️ Form Against', 'score': '-10', 'desc': f'Recent form contradicts pick'})
        
        # =================================================================
        # FACTOR 4: Momentum (0-8 points)
        # =================================================================
        momentum = features.get(f'{stat_col}_momentum', 0)
        if momentum is not None:
            if is_over and momentum > 0.1:
                score += 8
                factors.append({'name': '🚀 Positive Momentum', 'score': '+8', 'desc': f'Performance trending up ({momentum:.1%})'})
            elif not is_over and momentum < -0.1:
                score += 8
                factors.append({'name': '📉 Negative Momentum', 'score': '+8', 'desc': f'Performance trending down ({momentum:.1%})'})
            elif (is_over and momentum < -0.1) or (not is_over and momentum > 0.1):
                score -= 6
                factors.append({'name': '⚠️ Wrong Momentum', 'score': '-6', 'desc': 'Momentum against pick direction'})
        
        # =================================================================
        # FACTOR 5: Consistency Score (0-12 points)
        # =================================================================
        consistency = features.get(f'{stat_col}_consistency')
        l10_std = features.get(f'{stat_col}_L10_std')
        
        if consistency is not None:
            if consistency < 0.15:
                score += 12
                factors.append({'name': '🎯 Elite Consistency', 'score': '+12', 'desc': 'Extremely predictable performer'})
            elif consistency < 0.25:
                score += 8
                factors.append({'name': '✅ Consistent', 'score': '+8', 'desc': 'Reliable, low variance'})
            elif consistency > 0.40:
                score -= 10
                factors.append({'name': '🎰 Volatile', 'score': '-10', 'desc': 'High variance - unpredictable'})
        elif l10_std is not None:
            if l10_std < 3:
                score += 10
                factors.append({'name': '✅ Consistent', 'score': '+10', 'desc': 'Low recent variance'})
            elif l10_std > 7:
                score -= 8
                factors.append({'name': '🎰 Volatile', 'score': '-8', 'desc': 'High variance recently'})
        
        # =================================================================
        # FACTOR 6: Matchup History (0-12 points)
        # =================================================================
        vs_opp_avg = features.get(f'vs_opp_{stat_col.lower()}_avg')
        vs_opp_games = features.get('vs_opp_games', 0)
        
        if vs_opp_avg is not None and vs_opp_avg > 0 and vs_opp_games >= 3:
            vs_1h = vs_opp_avg * ratio
            vs_consistency = features.get('vs_opp_consistency', 1)
            
            if is_over and vs_1h > prop_line * 1.1:
                score += 12 if vs_opp_games >= 5 else 8
                factors.append({'name': '💪 Owns Matchup', 'score': f'+{12 if vs_opp_games >= 5 else 8}', 
                              'desc': f'Avg {vs_1h:.1f} 1H in {vs_opp_games} games vs opponent'})
            elif not is_over and vs_1h < prop_line * 0.9:
                score += 12 if vs_opp_games >= 5 else 8
                factors.append({'name': '😰 Struggles vs Opp', 'score': f'+{12 if vs_opp_games >= 5 else 8}', 
                              'desc': f'Only {vs_1h:.1f} 1H in {vs_opp_games} games vs opponent'})
        
        # =================================================================
        # FACTOR 7: Floor/Ceiling Analysis (0-10 points)
        # =================================================================
        floor_1h = features.get(f'{stat_col}_floor', 0) * ratio if features.get(f'{stat_col}_floor') else None
        ceiling_1h = features.get(f'{stat_col}_ceiling', 0) * ratio if features.get(f'{stat_col}_ceiling') else None
        
        if floor_1h is not None and ceiling_1h is not None:
            if is_over and floor_1h > prop_line:
                score += 10
                factors.append({'name': '🛡️ Floor Above Line', 'score': '+10', 'desc': f'Even floor ({floor_1h:.1f}) beats line'})
            elif not is_over and ceiling_1h < prop_line:
                score += 10
                factors.append({'name': '🛡️ Ceiling Below Line', 'score': '+10', 'desc': f'Even ceiling ({ceiling_1h:.1f}) under line'})
            elif is_over and ceiling_1h < prop_line:
                score -= 8
                factors.append({'name': '⚠️ Ceiling Risk', 'score': '-8', 'desc': f'Ceiling ({ceiling_1h:.1f}) below line'})
            elif not is_over and floor_1h > prop_line:
                score -= 8
                factors.append({'name': '⚠️ Floor Risk', 'score': '-8', 'desc': f'Floor ({floor_1h:.1f}) above line'})
        
        # =================================================================
        # FACTOR 8: Opponent Defense (0-8 points)
        # =================================================================
        opp_def_tier = features.get('CURRENT_OPP_DEF_TIER') or features.get('OPP_DEF_TIER')
        vs_good_def = features.get(f'{stat_col}_vs_good_def')
        vs_bad_def = features.get(f'{stat_col}_vs_bad_def')
        
        if opp_def_tier is not None:
            if is_over and opp_def_tier >= 4:  # Bad defense
                score += 8
                factors.append({'name': '🧀 Soft Defense', 'score': '+8', 'desc': f'Facing tier {int(opp_def_tier)} (weak) defense'})
            elif not is_over and opp_def_tier <= 2:  # Good defense
                score += 8
                factors.append({'name': '🔒 Elite Defense', 'score': '+8', 'desc': f'Facing tier {int(opp_def_tier)} (elite) defense'})
            elif is_over and opp_def_tier <= 2:
                score -= 5
                factors.append({'name': '⚠️ Tough Defense', 'score': '-5', 'desc': 'Facing elite defense'})
            elif not is_over and opp_def_tier >= 4:
                score -= 5
                factors.append({'name': '⚠️ Soft Matchup', 'score': '-5', 'desc': 'Facing weak defense'})
        
        # =================================================================
        # FACTOR 8.5: Game Pace Factor
        # =================================================================
        opponent = features.get('OPPONENT', '')
        player_team = features.get('TEAM', '')
        
        opp_pace = TEAM_PACE_TIERS.get(opponent, 3)
        team_pace = TEAM_PACE_TIERS.get(player_team, 3)
        combined_pace = (opp_pace + team_pace) / 2  # Lower = faster
        
        if combined_pace <= 1.5:  # Both teams are fast-paced
            if is_over:
                score += 8
                factors.append({'name': '🏃 High Pace Game', 'score': '+8', 'desc': 'Fast-paced matchup boosts scoring'})
            else:
                score -= 10
                factors.append({'name': '🚨 Pace Risk', 'score': '-10', 'desc': 'Fast-paced game - UNDER risky!'})
        elif combined_pace <= 2.5:  # Moderately fast
            if is_over:
                score += 4
                factors.append({'name': '🏃 Fast Matchup', 'score': '+4', 'desc': 'Above-average pace game'})
            else:
                score -= 5
                factors.append({'name': '⚠️ Pace Concern', 'score': '-5', 'desc': 'Faster than average matchup'})
        elif combined_pace >= 4.5:  # Both teams are slow
            if not is_over:
                score += 6
                factors.append({'name': '🐢 Slow Pace', 'score': '+6', 'desc': 'Slow-paced game favors UNDER'})
            else:
                score -= 4
                factors.append({'name': '⚠️ Slow Game', 'score': '-4', 'desc': 'Low-pace matchup limits scoring'})
        
        # =================================================================
        # FACTOR 9: Rest Impact (0-6 points)
        # =================================================================
        is_b2b = features.get('IS_B2B', 0)
        is_well_rested = features.get('IS_WELL_RESTED', 0)
        rest_days = features.get('REST_DAYS', 2)
        b2b_impact = features.get(f'{stat_col}_b2b_impact')
        
        if is_b2b:
            if b2b_impact and b2b_impact < 0.9:  # Significantly worse on B2B
                if not is_over:
                    score += 6
                    factors.append({'name': '😴 B2B Drag', 'score': '+6', 'desc': f'Drops to {b2b_impact:.0%} on B2B'})
                else:
                    score -= 6
                    factors.append({'name': '😴 B2B Risk', 'score': '-6', 'desc': f'Only {b2b_impact:.0%} performance on B2B'})
            else:
                if not is_over:
                    score += 4
                    factors.append({'name': '😴 B2B Fatigue', 'score': '+4', 'desc': 'Back-to-back may limit output'})
                else:
                    score -= 4
                    factors.append({'name': '😴 B2B Risk', 'score': '-4', 'desc': 'Back-to-back typically reduces stats'})
        elif is_well_rested and is_over:
            score += 4
            factors.append({'name': '💪 Well Rested', 'score': '+4', 'desc': f'{int(rest_days)} days rest - fresh legs'})
        
        # =================================================================
        # FACTOR 10: Minutes Stability (0-6 points)
        # =================================================================
        min_stability = features.get('MIN_stability')
        min_trend = features.get('MIN_trend')
        
        if min_stability is not None:
            if min_stability < 2:
                score += 6
                factors.append({'name': '⏱️ Stable Minutes', 'score': '+6', 'desc': 'Very consistent playing time'})
            elif min_stability > 5:
                score -= 4
                factors.append({'name': '⏱️ Minutes Variance', 'score': '-4', 'desc': 'Inconsistent playing time'})
        
        if min_trend is not None:
            if is_over and min_trend > 0.1:
                score += 3
                factors.append({'name': '⏱️ Minutes Up', 'score': '+3', 'desc': 'Getting more playing time'})
            elif not is_over and min_trend < -0.1:
                score += 3
                factors.append({'name': '⏱️ Minutes Down', 'score': '+3', 'desc': 'Playing time decreasing'})
        
        # =================================================================
        # FACTOR 11: Season Average Alignment (0-5 points)
        # =================================================================
        season_avg = features.get(f'{stat_col}_season_avg')
        if season_avg is not None:
            season_1h = season_avg * ratio
            if (is_over and season_1h > prop_line * 1.05) or (not is_over and season_1h < prop_line * 0.95):
                score += 5
                factors.append({'name': '📊 Season Supports', 'score': '+5', 'desc': f'Season avg ({season_1h:.1f} 1H) aligns'})
        
        # =================================================================
        # FACTOR 12: Hit Rate History - CALCULATED FROM ACTUAL RECENT GAMES
        # =================================================================
        hit_rate = None
        games_counted = 0
        
        # Calculate hit rate from actual recent games against the specific prop line
        if recent_games and len(recent_games) > 0:
            try:
                hits = 0
                total = 0
                for game in recent_games:
                    # Get full game stat - properly handle PRA by summing components
                    if stat_col == 'PRA':
                        full_stat = (game.get('PTS', 0) or 0) + (game.get('REB', 0) or 0) + (game.get('AST', 0) or 0)
                    else:
                        full_stat = game.get(stat_col) or game.get('PTS', 0) or 0
                    est_1h = full_stat * ratio
                    if est_1h > 0 or full_stat > 0:  # Valid game
                        total += 1
                        if est_1h > prop_line:
                            hits += 1
                if total >= 5:
                    hit_rate = hits / total
                    games_counted = total
            except Exception as e:
                # Log but continue - hit rate is supplementary, not critical
                print(f"[DEBUG] Hit rate calculation error: {e}")
        
        # Fallback to feature-based rate only if no recent games
        if hit_rate is None:
            threshold_key = f'{stat_col}_over_{int(prop_line)}_rate' if prop_line < 26 else f'{stat_col}_over_20_rate'
            hit_rate = features.get(threshold_key)
            if hit_rate is None:
                for t in [10, 15, 20, 25]:
                    hr = features.get(f'{stat_col}_over_{t}_rate')
                    if hr is not None:
                        hit_rate = hr
                        break
        
        if hit_rate is not None:
            # Include game count in description if calculated from recent games
            rate_source = f' ({games_counted} games vs {prop_line} line)' if games_counted > 0 else ''
            
            if is_over:
                # Predicting OVER - high hit rate is good, low is bad
                if hit_rate > 0.7:
                    score += 8
                    factors.append({'name': '📈 High Hit Rate', 'score': '+8', 'desc': f'{hit_rate:.0%} over rate{rate_source} - strong history'})
                elif hit_rate > 0.5:
                    score += 4
                    factors.append({'name': '📊 Decent Hit Rate', 'score': '+4', 'desc': f'{hit_rate:.0%} over rate{rate_source}'})
                elif hit_rate < 0.3:
                    score -= 20  # MAJOR penalty for low hit rate on OVER
                    factors.append({'name': '🚨 Poor Hit Rate', 'score': '-20', 'desc': f'Only {hit_rate:.0%} over rate{rate_source} - risky!'})
                elif hit_rate < 0.45:
                    score -= 10
                    factors.append({'name': '⚠️ Low Hit Rate', 'score': '-10', 'desc': f'{hit_rate:.0%} over rate{rate_source} - below average'})
            else:
                # Predicting UNDER - low hit rate (low over rate) is good
                if hit_rate < 0.3:
                    score += 8
                    factors.append({'name': '📉 Low Over Rate', 'score': '+8', 'desc': f'Only {hit_rate:.0%} over rate{rate_source} - supports UNDER'})
                elif hit_rate < 0.5:
                    score += 4
                    factors.append({'name': '📊 Moderate Rate', 'score': '+4', 'desc': f'{hit_rate:.0%} over rate{rate_source}'})
                elif hit_rate > 0.7:
                    score -= 20  # MAJOR penalty for high hit rate on UNDER
                    factors.append({'name': '🚨 High Over Rate', 'score': '-20', 'desc': f'{hit_rate:.0%} usually goes over{rate_source} - risky!'})
                elif hit_rate > 0.55:
                    score -= 10
                    factors.append({'name': '⚠️ High Hit Rate', 'score': '-10', 'desc': f'{hit_rate:.0%} over rate{rate_source} - against pick'})
        
        # =================================================================
        # FACTOR 13: Sample Size / Data Quality
        # =================================================================
        games_played = features.get('games_played') or (len(recent_games) if recent_games else 0)
        l10_avg = features.get(f'{stat_col}_L10_avg')
        
        if games_played < 5:
            score -= 20
            factors.append({'name': '🚨 Limited Data', 'score': '-20', 'desc': f'Only {games_played} games - very risky'})
        elif games_played < 10:
            score -= 10
            factors.append({'name': '⚠️ Small Sample', 'score': '-10', 'desc': f'Only {games_played} games of data'})
        elif games_played < 15:
            score -= 5
            factors.append({'name': '📊 Limited Sample', 'score': '-5', 'desc': f'{games_played} games - moderate confidence'})
        
        # =================================================================
        # FACTOR 14: Regression to Mean (NEW)
        # Extreme recent performance tends to revert
        # =================================================================
        l5_avg = features.get(f'{stat_col}_L5_avg')
        season_avg = features.get(f'{stat_col}_season_avg')
        l10_std = features.get(f'{stat_col}_L10_std')
        
        if l5_avg is not None and season_avg is not None and season_avg > 0:
            l5_ewm = features.get(f'{stat_col}_ewm_5') or l5_avg
            deviation_pct = (l5_ewm - season_avg) / season_avg
            
            # If recent form is way above season (hot streak may cool)
            if deviation_pct > 0.25:  # 25%+ above season avg
                if is_over:
                    score -= 8
                    factors.append({'name': '📉 Regression Risk', 'score': '-8', 'desc': f'Recent {deviation_pct:.0%} above average - may cool down'})
                else:
                    score += 6
                    factors.append({'name': '📉 Due for Regression', 'score': '+6', 'desc': f'Unsustainable {deviation_pct:.0%} above average'})
            # If recent form is way below season (cold streak may end)
            elif deviation_pct < -0.25:  # 25%+ below season avg
                if not is_over:
                    score -= 8
                    factors.append({'name': '📈 Regression Risk', 'score': '-8', 'desc': f'Recent {abs(deviation_pct):.0%} below average - may bounce back'})
                else:
                    score += 6
                    factors.append({'name': '📈 Due for Bounce', 'score': '+6', 'desc': f'Due to revert from {abs(deviation_pct):.0%} slump'})
        
        # =================================================================
        # FACTOR 15: Suspicious Edge Check (NEW)
        # Edges that are too large may indicate data issues
        # =================================================================
        if abs_diff >= 7.0:
            score -= 15
            factors.append({'name': '🚨 Suspicious Edge', 'score': '-15', 'desc': f'{abs_diff:.1f} pts edge is abnormally large - verify data'})
        elif abs_diff >= 5.0:
            score -= 8
            factors.append({'name': '⚠️ Large Edge Warning', 'score': '-8', 'desc': f'{abs_diff:.1f} pts edge unusual - proceed with caution'})
        
        # =================================================================
        # FACTOR 16: Variance-Adjusted Confidence (NEW)
        # High variance players need larger edges
        # =================================================================
        if l10_std is not None and l10_std > 0:
            # Edge in standard deviations
            std_ratio = l10_std * ratio  # Convert to 1H scale
            if std_ratio > 0:
                edge_in_stds = abs_diff / std_ratio
                
                if edge_in_stds < 0.5:  # Edge is less than half a std dev
                    score -= 10
                    factors.append({'name': '⚠️ Edge Within Noise', 'score': '-10', 'desc': f'Edge ({abs_diff:.1f}) < 0.5 std devs - coinflip territory'})
                elif edge_in_stds >= 1.5:  # Edge is 1.5+ std devs
                    score += 8
                    factors.append({'name': '📊 Statistically Significant', 'score': '+8', 'desc': f'Edge is {edge_in_stds:.1f} std devs - meaningful gap'})
        
        # =================================================================
        # FACTOR 17: Line Proximity to Average (NEW)
        # Lines very close to averages are harder to beat
        # =================================================================
        if season_avg is not None:
            season_1h = season_avg * ratio
            line_vs_avg_diff = abs(prop_line - season_1h)
            
            if line_vs_avg_diff < 0.5:
                score -= 8
                factors.append({'name': '⚖️ Tight Line', 'score': '-8', 'desc': f'Line ({prop_line}) very close to avg ({season_1h:.1f}) - coinflip'})
        
        # =================================================================
        # FACTOR 18: L3 Hit Rate - CRITICAL (Last 3 games most predictive)
        # =================================================================
        if recent_games and len(recent_games) >= 3:
            try:
                l3_games = recent_games[:3]  # Most recent 3
                l3_hits = 0
                l3_total = 0
                for game in l3_games:
                    # Properly handle PRA by summing components
                    if stat_col == 'PRA':
                        full_stat = (game.get('PTS', 0) or 0) + (game.get('REB', 0) or 0) + (game.get('AST', 0) or 0)
                    else:
                        full_stat = game.get(stat_col) or game.get('PTS', 0) or 0
                    est_1h = full_stat * ratio
                    if est_1h > 0 or full_stat > 0:
                        l3_total += 1
                        if est_1h > prop_line:
                            l3_hits += 1
                
                if l3_total >= 2:
                    l3_over_rate = l3_hits / l3_total
                    
                    if is_over:
                        if l3_over_rate >= 1.0:  # 3/3 over
                            score += 12
                            factors.append({'name': '🔥 L3 Perfect', 'score': '+12', 'desc': f'Last 3 games: ALL would hit OVER'})
                        elif l3_over_rate >= 0.66:  # 2/3 over
                            score += 6
                            factors.append({'name': '📈 L3 Strong', 'score': '+6', 'desc': f'Last 3 games: {l3_hits}/{l3_total} would hit OVER'})
                        elif l3_over_rate == 0:  # 0/3 over
                            score -= 15
                            factors.append({'name': '🚨 L3 Against', 'score': '-15', 'desc': f'Last 3 games: NONE would hit OVER'})
                        elif l3_over_rate <= 0.34:  # 1/3 over
                            score -= 8
                            factors.append({'name': '⚠️ L3 Weak', 'score': '-8', 'desc': f'Last 3 games: only {l3_hits}/{l3_total} would hit OVER'})
                    else:  # UNDER
                        l3_under_rate = 1 - l3_over_rate
                        if l3_under_rate >= 1.0:  # 0/3 over = all under
                            score += 12
                            factors.append({'name': '❄️ L3 Perfect', 'score': '+12', 'desc': f'Last 3 games: ALL would hit UNDER'})
                        elif l3_under_rate >= 0.66:  # 2/3 under
                            score += 6
                            factors.append({'name': '📉 L3 Strong', 'score': '+6', 'desc': f'Last 3 games: {l3_total - l3_hits}/{l3_total} would hit UNDER'})
                        elif l3_under_rate == 0:  # 3/3 over = 0 under
                            score -= 15
                            factors.append({'name': '🚨 L3 Against', 'score': '-15', 'desc': f'Last 3 games: NONE would hit UNDER'})
                        elif l3_under_rate <= 0.34:  # 1/3 under
                            score -= 8
                            factors.append({'name': '⚠️ L3 Weak', 'score': '-8', 'desc': f'Last 3 games: only {l3_total - l3_hits}/{l3_total} would hit UNDER'})
            except Exception:
                pass  # L3 is supplementary
        
        # =================================================================
        # FACTOR 19: Sharp Line Detection
        # If our prediction is very close to line, books may have it right
        # =================================================================
        pred_1h = details.get('first_half_prediction', 0)
        if pred_1h > 0:
            pred_line_diff = abs(pred_1h - prop_line)
            if pred_line_diff < 0.3:
                score -= 12
                factors.append({'name': '🎯 Line is Sharp', 'score': '-12', 'desc': f'Prediction ({pred_1h:.1f}) = line ({prop_line}) - no edge'})
            elif pred_line_diff < 0.6:
                score -= 6
                factors.append({'name': '⚠️ Thin Edge', 'score': '-6', 'desc': f'Only {pred_line_diff:.1f} pts from line - marginal'})
        
        # =================================================================
        # FACTOR 20: L3 vs L10 Direction Agreement
        # If L3 and L10 trends agree, more confident
        # =================================================================
        l3_avg = features.get(f'{stat_col}_L3_avg')
        l10_avg = features.get(f'{stat_col}_L10_avg')
        if l3_avg is not None and l10_avg is not None and l10_avg > 0:
            l3_1h = l3_avg * ratio
            l10_1h = l10_avg * ratio
            
            l3_over = l3_1h > prop_line
            l10_over = l10_1h > prop_line
            
            if l3_over == l10_over == is_over:
                score += 8
                factors.append({'name': '✅ L3/L10 Agree', 'score': '+8', 'desc': f'Both L3 ({l3_1h:.1f}) and L10 ({l10_1h:.1f}) support pick'})
            elif l3_over != l10_over:
                score -= 5
                factors.append({'name': '⚠️ L3/L10 Conflict', 'score': '-5', 'desc': f'L3 ({l3_1h:.1f}) and L10 ({l10_1h:.1f}) disagree'})
        
        # =================================================================
        # FACTOR 21: Recent Minutes Check
        # If recent minutes are significantly different from season, adjust
        # =================================================================
        l3_min = features.get('MIN_L3_avg')
        season_min = features.get('MIN_season_avg')
        if l3_min is not None and season_min is not None and season_min > 0:
            min_change_pct = (l3_min - season_min) / season_min
            
            if min_change_pct < -0.15:  # 15%+ fewer minutes recently
                if not is_over:
                    score += 5
                    factors.append({'name': '⏱️ Minutes Down', 'score': '+5', 'desc': f'Recent mins ({l3_min:.0f}) down {abs(min_change_pct):.0%} - supports UNDER'})
                else:
                    score -= 6
                    factors.append({'name': '⏱️ Minutes Risk', 'score': '-6', 'desc': f'Recent mins ({l3_min:.0f}) down {abs(min_change_pct):.0%}'})
            elif min_change_pct > 0.15:  # 15%+ more minutes recently
                if is_over:
                    score += 5
                    factors.append({'name': '⏱️ Minutes Up', 'score': '+5', 'desc': f'Recent mins ({l3_min:.0f}) up {min_change_pct:.0%} - supports OVER'})
                else:
                    score -= 6
                    factors.append({'name': '⏱️ Minutes Risk', 'score': '-6', 'desc': f'Recent mins ({l3_min:.0f}) up {min_change_pct:.0%}'})
        
        # =================================================================
        # FACTOR 22: STAT-SPECIFIC ADJUSTMENTS (rebounds/assists same strength as points)
        # =================================================================
        stat_type = self.target  # 'points', 'rebounds', 'assists', 'pra'
        
        # Get stat-specific variance multiplier
        stat_variance = STAT_VARIANCE.get(stat_type, 1.0)
        
        # Adjust edge requirements by stat type
        min_edge_pct = STAT_MIN_EDGE_PCT.get(stat_type, 8)
        edge_pct = (abs_diff / max(prop_line, 1)) * 100
        
        if edge_pct >= min_edge_pct * 1.5:  # 50% above minimum
            score += 6
            factors.append({'name': f'📊 Strong {stat_type.title()} Edge', 'score': '+6', 
                          'desc': f'{edge_pct:.0f}% edge exceeds {min_edge_pct}% min for {stat_type}'})
        elif edge_pct < min_edge_pct * 0.7:  # Below 70% of minimum
            score -= 8
            factors.append({'name': f'⚠️ Weak {stat_type.title()} Edge', 'score': '-8', 
                          'desc': f'Only {edge_pct:.0f}% edge (need {min_edge_pct}% for {stat_type})'})
        
        # Stat-specific consistency thresholds
        stat_consistency_thresholds = STAT_CONSISTENCY.get(stat_type, {"elite": 0.15, "good": 0.25, "volatile": 0.40})
        if consistency is not None:
            if consistency < stat_consistency_thresholds["elite"]:
                score += 5
                factors.append({'name': f'🎯 Elite {stat_type.title()} Consistency', 'score': '+5', 
                              'desc': f'Exceptionally consistent {stat_type} producer'})
            elif consistency > stat_consistency_thresholds["volatile"]:
                score -= 6
                factors.append({'name': f'🎰 Volatile {stat_type.title()}', 'score': '-6', 
                              'desc': f'{stat_type.title()} highly unpredictable'})
        
        # Low line volatility penalty
        line_vol = STAT_LINE_VOLATILITY.get(stat_type, {"low_threshold": 8, "penalty": -5})
        if prop_line < line_vol["low_threshold"]:
            penalty = line_vol["penalty"]
            score += penalty  # It's negative, so this subtracts
            factors.append({'name': f'⚠️ Low {stat_type.title()} Line', 'score': f'{penalty}', 
                          'desc': f'{stat_type.title()} lines under {line_vol["low_threshold"]} are volatile'})
        
        # Early rebounder/playmaker bonus for 1Q (if applicable - this gets applied in 1Q adjustments too)
        player_name = features.get('PLAYER_NAME', '')
        if stat_type == 'rebounds' and player_name in EARLY_REBOUNDERS:
            score += 5
            factors.append({'name': '💪 Early Rebounder', 'score': '+5', 
                          'desc': f'{player_name} dominates boards early'})
        elif stat_type == 'assists' and player_name in EARLY_PLAYMAKERS:
            score += 5
            factors.append({'name': '🎯 Early Playmaker', 'score': '+5', 
                          'desc': f'{player_name} creates early in games'})
        
        # Clamp score between 0-100
        score = max(0, min(100, score))
        
        return score, factors
    
    def _generate_reasoning(
        self, 
        features: Dict, 
        details: Dict, 
        prop_line: float,
        difference: float,
        period: str = '1h'
    ) -> List[str]:
        """Generate human-readable reasoning for the prediction"""
        reasons = []
        stat_name = self.target
        stat_col = self.target_col
        # Use period-aware ratio (1Q ~0.24 vs 1H ~0.48)
        ratio = get_period_ratio(self.target, period)
        period_label = "1Q" if period == '1q' else "1H"
        
        # Recent form
        l5_avg = features.get(f'{stat_col}_L5_avg')
        if l5_avg is not None:
            l5_1h = l5_avg * ratio
            if l5_1h > prop_line * 1.1:
                reasons.append(f"Recent form strong: averaging {l5_avg:.1f} {stat_name}/game (est. {l5_1h:.1f} {period_label}) over last 5")
            elif l5_1h < prop_line * 0.9:
                reasons.append(f"Recent form concerns: averaging {l5_avg:.1f} {stat_name}/game (est. {l5_1h:.1f} {period_label}) over last 5")
            else:
                reasons.append(f"Recent form neutral: averaging {l5_avg:.1f} {stat_name}/game over last 5")
        
        # Season average comparison
        season_avg = features.get(f'{stat_col}_season_avg')
        if season_avg is not None:
            season_1h = season_avg * ratio
            if season_1h > prop_line:
                reasons.append(f"Season average ({season_avg:.1f}) suggests {period_label} of {season_1h:.1f}, above line")
            else:
                reasons.append(f"Season average ({season_avg:.1f}) suggests {period_label} of {season_1h:.1f}, below line")
        
        # Home/Away
        is_home = features.get('IS_HOME', 0)
        home_avg = features.get(f'{stat_col}_home_avg')
        away_avg = features.get(f'{stat_col}_away_avg')
        
        if is_home and home_avg and away_avg:
            if home_avg > away_avg:
                reasons.append(f"Home game advantage: averages {home_avg:.1f} at home vs {away_avg:.1f} on road")
            else:
                reasons.append(f"Better on road: averages {away_avg:.1f} away vs {home_avg:.1f} at home")
        elif not is_home and home_avg and away_avg:
            if away_avg > home_avg:
                reasons.append(f"Road warrior: performs better away ({away_avg:.1f}) than home ({home_avg:.1f})")
        
        # Rest days
        rest_days = features.get('REST_DAYS', 2)
        is_b2b = features.get('IS_B2B', 0)
        if is_b2b:
            reasons.append("Back-to-back game - potential fatigue factor")
        elif rest_days and rest_days >= 3:
            reasons.append(f"Well-rested with {int(rest_days)} days off")
        
        # Matchup history
        vs_opp_avg = features.get(f'vs_opp_{stat_col.lower()}_avg')
        if vs_opp_avg is not None and vs_opp_avg > 0:
            vs_1h = vs_opp_avg * ratio
            reasons.append(f"Historical vs opponent: {vs_opp_avg:.1f} {stat_name}/game (est. {vs_1h:.1f} {period_label})")
        
        # Variance/consistency
        l5_std = features.get(f'{stat_col}_L5_std')
        if l5_std is not None:
            if l5_std < 3:
                reasons.append("Highly consistent performer recently")
            elif l5_std > 7:
                reasons.append("High variance - boom or bust potential")
        
        # Model agreement - but note if prediction was capped
        model_preds = details.get('model_predictions', {})
        if model_preds:
            num_models = len(model_preds)
            preds_1h = [p * details['first_half_ratio'] for p in model_preds.values()]
            
            # Check if prediction was capped (models gave unrealistic values)
            if details.get('prediction_capped'):
                reasons.append(f"⚠️ Raw model outputs ignored (unrealistic for this player)")
            elif all(p > prop_line for p in preds_1h):
                reasons.append(f"All {num_models} models agree on OVER")
            elif all(p < prop_line for p in preds_1h):
                reasons.append(f"All {num_models} models agree on UNDER")
            else:
                reasons.append("Models show mixed signals")
        
        return reasons
    
    def get_feature_importance(self) -> Dict:
        """Get aggregated feature importance across all models"""
        if not self.feature_importances:
            return {}
        
        # Pre-compute aggregated importance
        all_features = set()
        for imp_dict in self.feature_importances.values():
            all_features.update(imp_dict.keys())
        
        avg_importance = {}
        for feature in all_features:
            total = sum(
                self.feature_importances.get(name, {}).get(feature, 0) * ENSEMBLE_WEIGHTS[name]
                for name in ENSEMBLE_WEIGHTS
            )
            avg_importance[feature] = total
        
        # Normalize
        total = sum(avg_importance.values())
        if total > 0:
            avg_importance = {k: v/total for k, v in avg_importance.items()}
        
        return dict(sorted(avg_importance.items(), key=lambda x: -x[1]))
    
    def _save_model(self):
        """Save trained model to disk"""
        model_path = os.path.join(MODELS_DIR, f'ensemble_{self.target}.joblib')
        
        save_dict = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importances': self.feature_importances,
            'target': self.target,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(save_dict, model_path, compress=3)  # Compression for smaller files
        print(f"Model saved to {model_path}")
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        model_path = os.path.join(MODELS_DIR, f'ensemble_{self.target}.joblib')
        
        if not os.path.exists(model_path):
            return False
        
        try:
            save_dict = joblib.load(model_path)
            self.models = save_dict['models']
            self.scaler = save_dict['scaler']
            self.feature_cols = save_dict['feature_cols']
            self.feature_importances = save_dict['feature_importances']
            self.is_trained = True
            self._models_initialized = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def create_quick_model(target: str = 'points') -> NBA1HEnsembleModel:
    """
    Create a model with default/estimated parameters.
    Used when not enough training data is available.
    """
    model = NBA1HEnsembleModel(target=target)
    
    # Set default feature columns
    model.feature_cols = [
        'PTS_L3_avg', 'PTS_L5_avg', 'PTS_L10_avg',
        'PRA_L3_avg', 'PRA_L5_avg', 'PRA_L10_avg',
        'REB_L5_avg', 'AST_L5_avg', 'MIN_L5_avg',
        'PTS_season_avg', 'PRA_season_avg',
        'IS_HOME', 'REST_DAYS', 'IS_B2B',
        'PTS_L5_std', 'PRA_L5_std'
    ]
    
    return model


# Pre-computed ratio lookup for 1H (DEFAULT estimates)
_RATIO_CACHE_1H = {
    'points': FIRST_HALF_RATIOS.get('points', 0.48),
    'rebounds': FIRST_HALF_RATIOS.get('rebounds', 0.48),
    'assists': FIRST_HALF_RATIOS.get('assists', 0.48),
    'pra': FIRST_HALF_RATIOS.get('pra', 0.48)
}

# Pre-computed ratio lookup for 1Q (DEFAULT estimates)
_RATIO_CACHE_1Q = {
    'points': FIRST_QUARTER_RATIOS.get('points', 0.24),
    'rebounds': FIRST_QUARTER_RATIOS.get('rebounds', 0.23),
    'assists': FIRST_QUARTER_RATIOS.get('assists', 0.245),
    'pra': FIRST_QUARTER_RATIOS.get('pra', 0.24)
}


def get_period_ratio(prop_type: str, period: str = '1h') -> float:
    """Get the DEFAULT ratio based on period (1h or 1q) - used when no player data available"""
    if period == '1q':
        return _RATIO_CACHE_1Q.get(prop_type, 0.24)
    return _RATIO_CACHE_1H.get(prop_type, 0.48)


def get_player_period_ratio(player_name: str, prop_type: str, period: str = '1h') -> tuple:
    """
    Get player-specific ratio if available, otherwise return default.
    
    Returns:
        (ratio, is_real_data) - ratio value and whether it's from real data
    """
    try:
        from period_boxscore_collector import load_player_period_stats
        
        stats = load_player_period_stats(player_name)
        if stats:
            period_key = '1H' if period.lower() == '1h' else '1Q'
            
            if prop_type == 'pra':
                # PRA ratio: weighted average of PTS, REB, AST ratios
                pts_ratio = stats.get(f'{period_key}_PTS_ratio')
                reb_ratio = stats.get(f'{period_key}_REB_ratio')
                ast_ratio = stats.get(f'{period_key}_AST_ratio')
                
                # Get full-game averages to weight the ratios properly
                full_pts = stats.get('FULL_PTS_avg', 0) or 0
                full_reb = stats.get('FULL_REB_avg', 0) or 0
                full_ast = stats.get('FULL_AST_avg', 0) or 0
                full_pra = full_pts + full_reb + full_ast
                
                if pts_ratio and reb_ratio and ast_ratio and full_pra > 0:
                    # Weight each ratio by its proportion of total PRA
                    pra_ratio = (pts_ratio * full_pts + reb_ratio * full_reb + ast_ratio * full_ast) / full_pra
                    return (pra_ratio, True)
            else:
                # Single stat ratio
                stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}
                stat_key = stat_map.get(prop_type, 'PTS')
                ratio_key = f'{period_key}_{stat_key}_ratio'
                
                real_ratio = stats.get(ratio_key)
                if real_ratio and real_ratio > 0:
                    return (real_ratio, True)  # Real data!
    except ImportError:
        pass  # Module not available, use defaults
    except Exception:
        pass  # Any error, use defaults
    
    # Fall back to default estimate
    default_ratio = get_period_ratio(prop_type, period)
    return (default_ratio, False)


def get_player_period_average(player_name: str, prop_type: str, period: str = '1h') -> tuple:
    """
    Get player's ACTUAL period average if available.
    
    Returns:
        (average, is_real_data) - average value and whether it's from real data
    """
    try:
        from period_boxscore_collector import load_player_period_stats
        
        stats = load_player_period_stats(player_name)
        if stats:
            period_key = '1H' if period.lower() == '1h' else '1Q'
            
            if prop_type == 'pra':
                # PRA = Points + Rebounds + Assists
                pts_avg = stats.get(f'{period_key}_PTS_avg')
                reb_avg = stats.get(f'{period_key}_REB_avg')
                ast_avg = stats.get(f'{period_key}_AST_avg')
                
                if pts_avg is not None and reb_avg is not None and ast_avg is not None:
                    pra_avg = pts_avg + reb_avg + ast_avg
                    if pra_avg >= 0:
                        return (pra_avg, True)  # Real PRA data!
            else:
                # Single stat average
                stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}
                stat_key = stat_map.get(prop_type, 'PTS')
                avg_key = f'{period_key}_{stat_key}_avg'
                
                real_avg = stats.get(avg_key)
                if real_avg is not None and real_avg >= 0:
                    return (real_avg, True)  # Real data!
    except ImportError:
        pass
    except Exception:
        pass
    
    return (None, False)  # No real data available


def get_lock_thresholds(period: str = '1h') -> dict:
    """Get the appropriate lock thresholds based on period"""
    if period == '1q':
        return LOCK_THRESHOLDS_1Q
    return LOCK_THRESHOLDS


def get_player_scoring_style(player_name: str) -> str:
    """Determine if player is a fast starter, closer, or even scorer"""
    if not player_name:
        return "default"
    
    player_lower = player_name.lower()
    
    for fast_starter in PLAYER_SCORING_STYLES.get("fast_starters", []):
        if fast_starter.lower() in player_lower or player_lower in fast_starter.lower():
            return "fast_starter"
    
    for closer in PLAYER_SCORING_STYLES.get("closers", []):
        if closer.lower() in player_lower or player_lower in closer.lower():
            return "closer"
    
    return "default"


def get_adjusted_1q_ratio(player_name: str, base_ratio: float) -> float:
    """Get player-adjusted 1Q ratio based on their scoring style"""
    style = get_player_scoring_style(player_name)
    adjustment = Q1_RATIO_ADJUSTMENTS.get(style, 1.0)
    return base_ratio * adjustment


def apply_1q_adjustments(lock_score: int, lock_factors: list, features: dict, 
                          prop_line: float, difference: float, period: str,
                          player_name: str = None, recent_games: list = None,
                          has_real_period_data: bool = False) -> tuple:
    """
    Apply comprehensive 1Q-specific adjustments for higher accuracy.
    1Q has higher variance, so we need stricter criteria and smarter analysis.
    """
    if period != '1q':
        return lock_score, lock_factors
    
    adjusted_score = lock_score
    is_over = difference > 0
    abs_diff = abs(difference)
    
    # =========================================================================
    # 1Q FACTOR 0: REAL DATA BONUS - Offsets variance when we have actual 1Q data
    # =========================================================================
    if has_real_period_data:
        adjusted_score += 15  # Big bonus for having real 1Q data
        lock_factors.append({
            'name': '✅ Real 1Q Data',
            'score': '+15',
            'desc': 'Using actual 1Q stats - more reliable than estimates'
        })
    
    # =========================================================================
    # 1Q FACTOR 1: Base variance penalty (reduced if we have real data)
    # =========================================================================
    variance_penalty = -4 if has_real_period_data else -8  # Less penalty with real data
    adjusted_score += variance_penalty
    lock_factors.append({
        'name': '⏱️ 1Q Variance',
        'score': str(variance_penalty),
        'desc': '1st Quarter has higher variance than 1st Half'
    })
    
    # =========================================================================
    # 1Q FACTOR 2: Player scoring style (fast starter vs closer)
    # =========================================================================
    if player_name:
        style = get_player_scoring_style(player_name)
        if style == "fast_starter":
            if is_over:
                adjusted_score += 8
                lock_factors.append({
                    'name': '🚀 Fast Starter',
                    'score': '+8',
                    'desc': f'{player_name} typically scores early - good for 1Q OVER'
                })
            else:
                adjusted_score -= 6
                lock_factors.append({
                    'name': '⚠️ Fast Starter Risk',
                    'score': '-6',
                    'desc': f'{player_name} typically scores early - risky for 1Q UNDER'
                })
        elif style == "closer":
            if is_over:
                adjusted_score -= 8
                lock_factors.append({
                    'name': '⚠️ Closer/Late Scorer',
                    'score': '-8',
                    'desc': f'{player_name} typically scores late - risky for 1Q OVER'
                })
            else:
                adjusted_score += 6
                lock_factors.append({
                    'name': '✅ Closer Pattern',
                    'score': '+6',
                    'desc': f'{player_name} typically scores late - good for 1Q UNDER'
                })
    
    # =========================================================================
    # 1Q FACTOR 3: Edge size (stricter requirements - reduced if real data)
    # =========================================================================
    min_edge_1q = 1.0 if has_real_period_data else 1.5  # Lower bar with real data
    if abs_diff < 0.5:  # Only penalize truly tiny edges
        penalty = -8 if has_real_period_data else -15
        adjusted_score += penalty
        lock_factors.append({
            'name': '🚨 Tiny 1Q Edge',
            'score': str(penalty),
            'desc': f'1Q edge only {abs_diff:.1f} - risky'
        })
    elif abs_diff < min_edge_1q:
        penalty = int((min_edge_1q - abs_diff) * (6 if has_real_period_data else 12))
        adjusted_score -= penalty
        lock_factors.append({
            'name': '⚠️ Small 1Q Edge',
            'score': f'-{penalty}',
            'desc': f'1Q edge {abs_diff:.1f} (want {min_edge_1q}+)'
        })
    elif abs_diff >= 3.0:
        adjusted_score += 10
        lock_factors.append({
            'name': '✅ Large 1Q Edge',
            'score': '+10',
            'desc': f'Strong {abs_diff:.1f} pt edge helps offset 1Q variance'
        })
    elif abs_diff >= 2.0:
        adjusted_score += 5
        lock_factors.append({
            'name': '✅ Good 1Q Edge',
            'score': '+5',
            'desc': f'Solid {abs_diff:.1f} pt edge for 1Q'
        })
    
    # =========================================================================
    # 1Q FACTOR 4: Consistency (reduced penalties with real data)
    # =========================================================================
    _1q_prop = features.get('_prop_type', 'points')
    _1q_stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pra': 'PRA'}
    _1q_col = _1q_stat_map.get(_1q_prop, 'PTS')
    consistency = features.get(f'{_1q_col}_consistency')
    l10_std = features.get(f'{_1q_col}_L10_std')
    
    if consistency is not None:
        if consistency > 0.35:  # Very inconsistent
            penalty = -8 if has_real_period_data else -15  # Less harsh with real data
            adjusted_score += penalty
            lock_factors.append({
                'name': '🎰 High 1Q Risk',
                'score': str(penalty),
                'desc': f'Inconsistent ({consistency:.0%} CV) - use caution'
            })
        elif consistency > 0.28:
            adjusted_score -= 8
            lock_factors.append({
                'name': '🎰 1Q Volatility',
                'score': '-8',
                'desc': f'Inconsistent player ({consistency:.0%} CV) - risky for 1Q'
            })
        elif consistency < 0.15:
            adjusted_score += 10
            lock_factors.append({
                'name': '🎯 Elite 1Q Consistency',
                'score': '+10',
                'desc': f'Very consistent ({consistency:.0%} CV) - ideal for 1Q'
            })
        elif consistency < 0.22:
            adjusted_score += 5
            lock_factors.append({
                'name': '🎯 Good 1Q Consistency',
                'score': '+5',
                'desc': f'Consistent player ({consistency:.0%} CV)'
            })
    elif l10_std is not None:
        if l10_std > 6:
            adjusted_score -= 10
            lock_factors.append({
                'name': '🎰 High Variance',
                'score': '-10',
                'desc': f'High std dev ({l10_std:.1f}) - risky for 1Q'
            })
        elif l10_std < 3:
            adjusted_score += 8
            lock_factors.append({
                'name': '🎯 Low Variance',
                'score': '+8',
                'desc': f'Low std dev ({l10_std:.1f}) - good for 1Q'
            })
    
    # =========================================================================
    # 1Q FACTOR 5: Sample size (skip if we have real period data)
    # =========================================================================
    if not has_real_period_data:  # Only penalize sample size if using estimates
        games_played = features.get('games_played', 0)
        if games_played < 10:
            adjusted_score -= 15
            lock_factors.append({
                'name': '🚨 1Q Data Risk',
                'score': '-15',
                'desc': f'Only {games_played} games - not enough for 1Q prediction'
            })
        elif games_played < 20:
            penalty = int((20 - games_played) * 0.5)
            adjusted_score -= penalty
            lock_factors.append({
                'name': '📊 Limited 1Q Data',
                'score': f'-{penalty}',
                'desc': f'1Q needs 20+ games, have {games_played}'
            })
    elif features.get('games_played', 0) >= 30:
        adjusted_score += 5
        lock_factors.append({
            'name': '📊 Strong Sample',
            'score': '+5',
            'desc': f'{features.get("games_played", 0)} games - good data for 1Q'
        })
    
    # =========================================================================
    # 1Q FACTOR 6: Line vs average analysis (skip coinflip if real data)
    # =========================================================================
    # When we have real period data, we're using actual 1Q avg, not estimated
    if not has_real_period_data:
        l5_avg = features.get(f'{_1q_col}_L5_avg')
        
        if l5_avg:
            ratio = get_period_ratio(_1q_prop, '1q')
            avg_1q = l5_avg * ratio
            line_vs_avg = prop_line - avg_1q
            
            if abs(line_vs_avg) < 0.5:
                adjusted_score -= 10
                lock_factors.append({
                    'name': '⚖️ 1Q Coinflip',
                    'score': '-10',
                    'desc': f'Line ({prop_line}) ≈ estimated 1Q avg ({avg_1q:.1f})'
                })
            elif is_over and line_vs_avg > 1.5:
                # Line is well above average, OVER is risky
                adjusted_score -= 8
                lock_factors.append({
                    'name': '⚠️ Line Above Avg',
                    'score': '-8',
                    'desc': f'Line ({prop_line}) above 1Q avg ({avg_1q:.1f}) - OVER harder'
                })
            elif not is_over and line_vs_avg < -1.5:
                # Line is well below average, UNDER is risky
                adjusted_score -= 8
                lock_factors.append({
                    'name': '⚠️ Line Below Avg',
                    'score': '-8',
                    'desc': f'Line ({prop_line}) below 1Q avg ({avg_1q:.1f}) - UNDER harder'
                })
    
    # =========================================================================
    # 1Q FACTOR 7: Recent momentum (more important for 1Q)
    # =========================================================================
    momentum = features.get(f'{_1q_col}_momentum')
    if momentum is not None:
        if is_over and momentum > 0.15:
            adjusted_score += 6
            lock_factors.append({
                'name': '🔥 Hot Momentum',
                'score': '+6',
                'desc': f'Strong upward trend ({momentum:.0%}) helps 1Q OVER'
            })
        elif not is_over and momentum < -0.15:
            adjusted_score += 6
            lock_factors.append({
                'name': '❄️ Cold Momentum',
                'score': '+6',
                'desc': f'Downward trend ({momentum:.0%}) helps 1Q UNDER'
            })
        elif is_over and momentum < -0.10:
            adjusted_score -= 8
            lock_factors.append({
                'name': '⚠️ Cold for OVER',
                'score': '-8',
                'desc': f'Player trending down ({momentum:.0%}) - risky 1Q OVER'
            })
        elif not is_over and momentum > 0.10:
            adjusted_score -= 8
            lock_factors.append({
                'name': '⚠️ Hot for UNDER',
                'score': '-8',
                'desc': f'Player trending up ({momentum:.0%}) - risky 1Q UNDER'
            })
    
    # =========================================================================
    # 1Q FACTOR 8: Back-to-back impact (stronger for 1Q)
    # =========================================================================
    is_b2b = features.get('IS_B2B', 0)
    if is_b2b:
        if not is_over:
            adjusted_score += 8
            lock_factors.append({
                'name': '😴 B2B 1Q Advantage',
                'score': '+8',
                'desc': 'B2B fatigue often shows in Q1 - good for UNDER'
            })
        else:
            adjusted_score -= 10
            lock_factors.append({
                'name': '😴 B2B 1Q Risk',
                'score': '-10',
                'desc': 'B2B fatigue hurts early - risky for 1Q OVER'
            })
    
    # =========================================================================
    # 1Q FACTOR 9: Home court (stronger impact in Q1 due to crowd energy)
    # =========================================================================
    is_home = features.get('IS_HOME', 0)
    home_avg = features.get(f'{_1q_col}_home_avg')
    away_avg = features.get(f'{_1q_col}_away_avg')
    
    if home_avg and away_avg:
        home_away_diff = home_avg - away_avg
        if is_home and home_away_diff > 3 and is_over:
            adjusted_score += 5
            lock_factors.append({
                'name': '🏠 1Q Home Boost',
                'score': '+5',
                'desc': f'Strong home scorer (+{home_away_diff:.1f}) - helps 1Q OVER'
            })
        elif not is_home and home_away_diff > 3 and not is_over:
            adjusted_score += 4
            lock_factors.append({
                'name': '✈️ 1Q Road Factor',
                'score': '+4',
                'desc': f'Weaker on road (-{home_away_diff:.1f}) - helps 1Q UNDER'
            })
    
    # =========================================================================
    # 1Q FACTOR 10: Floor/Ceiling analysis (critical for 1Q)
    # =========================================================================
    floor = features.get(f'{_1q_col}_floor')
    ceiling = features.get(f'{_1q_col}_ceiling')
    
    if floor and ceiling:
        ratio = get_period_ratio(_1q_prop, '1q')
        floor_1q = floor * ratio
        ceiling_1q = ceiling * ratio
        
        if is_over and floor_1q > prop_line:
            adjusted_score += 12
            lock_factors.append({
                'name': '🛡️ 1Q Floor Safe',
                'score': '+12',
                'desc': f'Even floor ({floor_1q:.1f}) beats line - strong 1Q OVER'
            })
        elif not is_over and ceiling_1q < prop_line:
            adjusted_score += 12
            lock_factors.append({
                'name': '🛡️ 1Q Ceiling Safe',
                'score': '+12',
                'desc': f'Even ceiling ({ceiling_1q:.1f}) under line - strong 1Q UNDER'
            })
        elif is_over and ceiling_1q < prop_line * 1.1:
            adjusted_score -= 10
            lock_factors.append({
                'name': '⚠️ 1Q Ceiling Risk',
                'score': '-10',
                'desc': f'Ceiling ({ceiling_1q:.1f}) barely above line - tough 1Q OVER'
            })
        elif not is_over and floor_1q > prop_line * 0.9:
            adjusted_score -= 10
            lock_factors.append({
                'name': '⚠️ 1Q Floor Risk',
                'score': '-10',
                'desc': f'Floor ({floor_1q:.1f}) close to line - tough 1Q UNDER'
            })
    
    # =========================================================================
    # 1Q FACTOR 11: L3 recent performance (most predictive for 1Q)
    # =========================================================================
    if recent_games and len(recent_games) >= 3:
        try:
            # Determine actual prop type from the features context
            _prop_type = features.get('_prop_type', 'points')
            ratio = get_period_ratio(_prop_type, '1q')
            l3_games = recent_games[:3]
            l3_hits = 0
            l3_total = 0
            
            for game in l3_games:
                # Use the correct stat column based on prop type
                _stat_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pra': 'PRA'}
                _stat_key = _stat_map.get(_prop_type, 'PTS')
                if _stat_key == 'PRA':
                    stat = (game.get('PTS', 0) or 0) + (game.get('REB', 0) or 0) + (game.get('AST', 0) or 0)
                else:
                    stat = game.get(_stat_key) or game.get('PTS', 0) or 0
                if stat and stat > 0:
                    est_1q = stat * ratio
                    l3_total += 1
                    if est_1q > prop_line:
                        l3_hits += 1
            
            if l3_total >= 2:
                l3_rate = l3_hits / l3_total
                if is_over:
                    if l3_rate >= 1.0:
                        adjusted_score += 10
                        lock_factors.append({
                            'name': '🔥 L3 1Q Perfect',
                            'score': '+10',
                            'desc': 'Last 3 games ALL would hit 1Q OVER'
                        })
                    elif l3_rate == 0:
                        adjusted_score -= 15
                        lock_factors.append({
                            'name': '🚨 L3 1Q Miss',
                            'score': '-15',
                            'desc': 'Last 3 games NONE would hit 1Q OVER'
                        })
                else:
                    if l3_rate == 0:
                        adjusted_score += 10
                        lock_factors.append({
                            'name': '❄️ L3 1Q Perfect',
                            'score': '+10',
                            'desc': 'Last 3 games ALL would hit 1Q UNDER'
                        })
                    elif l3_rate >= 1.0:
                        adjusted_score -= 15
                        lock_factors.append({
                            'name': '🚨 L3 1Q Miss',
                            'score': '-15',
                            'desc': 'Last 3 games NONE would hit 1Q UNDER'
                        })
        except:
            pass
    
    # =========================================================================
    # 1Q FACTOR 12: STAT-SPECIFIC 1Q ADJUSTMENTS (rebounds/assists)
    # =========================================================================
    # Use prop type from features (set by caller), not heuristic detection
    stat_type = features.get('_prop_type', 'points')
    
    # Get stat-specific 1Q ratios
    from config import STAT_VARIANCE, EARLY_REBOUNDERS, EARLY_PLAYMAKERS
    stat_var = STAT_VARIANCE.get(stat_type, 1.0)
    
    # Apply stat-specific variance penalty (rebounds/assists more volatile in 1Q)
    if stat_type in ['rebounds', 'assists']:
        extra_penalty = int((stat_var - 1.0) * 15)  # Extra penalty for variance
        if extra_penalty > 0:
            adjusted_score -= extra_penalty
            lock_factors.append({
                'name': f'⚠️ 1Q {stat_type.title()} Variance',
                'score': f'-{extra_penalty}',
                'desc': f'{stat_type.title()} have higher 1Q variance than points'
            })
    
    # Early rebounder bonus
    if stat_type == 'rebounds' and player_name:
        if player_name in EARLY_REBOUNDERS:
            if is_over:
                adjusted_score += 10
                lock_factors.append({
                    'name': '💪 Early Rebounder',
                    'score': '+10',
                    'desc': f'{player_name} attacks the glass early - great for 1Q REB OVER'
                })
            else:
                adjusted_score -= 6
                lock_factors.append({
                    'name': '⚠️ Early Rebounder Risk',
                    'score': '-6',
                    'desc': f'{player_name} attacks the glass early - risky for 1Q REB UNDER'
                })
    
    # Early playmaker bonus
    if stat_type == 'assists' and player_name:
        if player_name in EARLY_PLAYMAKERS:
            if is_over:
                adjusted_score += 10
                lock_factors.append({
                    'name': '🎯 Early Playmaker',
                    'score': '+10',
                    'desc': f'{player_name} runs the offense early - great for 1Q AST OVER'
                })
            else:
                adjusted_score -= 6
                lock_factors.append({
                    'name': '⚠️ Early Playmaker Risk',
                    'score': '-6',
                    'desc': f'{player_name} runs the offense early - risky for 1Q AST UNDER'
                })
    
    # Rebounds are more consistent early in games (less fatigue)
    if stat_type == 'rebounds':
        reb_consistency = features.get('REB_consistency')
        if reb_consistency and reb_consistency < 0.25:
            adjusted_score += 5
            lock_factors.append({
                'name': '🎯 1Q Reb Consistency',
                'score': '+5',
                'desc': f'Consistent rebounder - rebounds more predictable in 1Q'
            })
    
    # Assists depend on team flow - check team pace
    if stat_type == 'assists':
        team_pace = features.get('TEAM', '')
        if team_pace:
            from config import TEAM_PACE_TIERS
            pace_tier = TEAM_PACE_TIERS.get(team_pace, 3)
            if pace_tier <= 2:  # Fast team
                if is_over:
                    adjusted_score += 5
                    lock_factors.append({
                        'name': '🏃 Fast Team 1Q Assists',
                        'score': '+5',
                        'desc': 'Fast-paced team generates early assists'
                    })
    
    return max(0, min(100, adjusted_score)), lock_factors


def make_prediction(
    player_name: str,
    opponent: str,
    prop_line: float,
    prop_type: str = 'points',
    is_home: bool = True,
    period: str = '1h'
) -> Dict:
    """
    Convenience function to make a prediction.
    Uses cached model for performance.
    
    Args:
        player_name: Player's full name
        opponent: Opponent team abbreviation
        prop_line: The betting line
        prop_type: 'points' or 'pra'
        is_home: Whether player is at home
        period: '1h' for first half or '1q' for first quarter
    """
    from feature_engineering import create_prediction_features
    
    # Validate period
    period = period.lower() if period else '1h'
    if period not in ['1h', '1q']:
        period = '1h'
    
    # Get features
    features, recent_games = create_prediction_features(
        player_name=player_name,
        opponent_abbrev=opponent,
        is_home=is_home,
        prop_line=prop_line,
        prop_type=prop_type
    )
    
    if features is None:
        return {
            'error': f"Could not find player: {player_name}",
            'pick': None
        }
    
    # Get cached model
    model = get_cached_model(target=prop_type)
    
    if not model.is_trained:
        # EXTREME ACCURACY quick estimation using advanced features
        stat_col_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pra': 'PRA'}
        stat_col = stat_col_map.get(prop_type, 'PTS')
        period_label = "1Q" if period == '1q' else "1H"
        
        # =====================================================================
        # NEW: Try to use REAL period data first!
        # =====================================================================
        real_period_avg, has_real_data = get_player_period_average(player_name, prop_type, period)
        ratio, has_real_ratio = get_player_period_ratio(player_name, prop_type, period)
        
        # Get all available full-game averages
        l3_avg = features.get(f'{stat_col}_L3_avg')
        l5_avg = features.get(f'{stat_col}_L5_avg')
        l10_avg = features.get(f'{stat_col}_L10_avg')
        season_avg = features.get(f'{stat_col}_season_avg')
        ewm_5 = features.get(f'{stat_col}_ewm_5')
        l5_std = features.get(f'{stat_col}_L5_std')
        l10_std = features.get(f'{stat_col}_L10_std')
        
        # Default to prop_line / ratio if no data
        default_val = prop_line / ratio
        l5_avg = l5_avg if l5_avg else default_val
        l10_avg = l10_avg if l10_avg else l5_avg
        season_avg = season_avg if season_avg else l5_avg
        ewm_5 = ewm_5 if ewm_5 else l5_avg
        
        # =====================================================================
        # PREDICTION: Use REAL data if available, otherwise estimate
        # =====================================================================
        if has_real_data and real_period_avg is not None:
            # USE REAL PERIOD AVERAGE - This is the gold standard!
            first_half_est = real_period_avg
            full_game_est = first_half_est / ratio if ratio > 0 else first_half_est * 2
            data_source = "REAL"
        else:
            # Fall back to estimated prediction
            # Weighted ensemble prediction (recent weighted heavily)
            full_game_est = (
                ewm_5 * 0.35 +      # Exponentially weighted recent
                l5_avg * 0.30 +     # Last 5 games
                l10_avg * 0.20 +    # Last 10 games
                season_avg * 0.15   # Season baseline
            )
            first_half_est = full_game_est * ratio
            data_source = "ESTIMATED"
        
        # =====================================================================
        # APPLY ADVANCED STATS BOOST (usage%, pace, etc.)
        # =====================================================================
        advanced_boost = 1.0
        advanced_boost_info = None
        try:
            from advanced_stats_collector import get_comprehensive_boost
            boost_data = get_comprehensive_boost(player_name, prop_type)
            if boost_data and boost_data.get('combined_boost', 1.0) != 1.0:
                advanced_boost = boost_data['combined_boost']
                first_half_est = first_half_est * advanced_boost
                full_game_est = full_game_est * advanced_boost
                advanced_boost_info = boost_data
        except ImportError:
            pass  # Advanced stats not available
        except Exception:
            pass  # Any error, skip boost
        
        # =====================================================================
        # INJURY IMPACT ADJUSTMENT
        # =====================================================================
        injury_boost = 1.0
        injury_info = None
        try:
            from injury_tracker import get_injury_features
            
            # Get player's team - try multiple sources
            player_team = features.get('TEAM', '')
            
            if not player_team:
                # Try to get from matchup in features
                matchup = features.get('MATCHUP', '')
                if matchup:
                    # Matchup format: "PHI @ GSW" or "PHI vs. BOS"
                    if '@' in matchup:
                        player_team = matchup.split('@')[0].strip()
                    elif 'vs.' in matchup:
                        player_team = matchup.split('vs.')[0].strip()
                    # Handle case like "PHI @ GSW" -> "PHI"
                    if ' ' in player_team:
                        player_team = player_team.split()[-1]
            
            if not player_team and recent_games is not None:
                # Try to get from recent games
                if hasattr(recent_games, 'iloc') and len(recent_games) > 0:
                    last_matchup = recent_games.iloc[-1].get('MATCHUP', '')
                    if last_matchup:
                        if '@' in last_matchup:
                            player_team = last_matchup.split('@')[0].strip()
                        elif 'vs.' in last_matchup:
                            player_team = last_matchup.split('vs.')[0].strip()
                        if ' ' in player_team:
                            player_team = player_team.split()[-1]
            
            if player_team and opponent:
                injury_data = get_injury_features(
                    player_name=player_name,
                    player_team=player_team,
                    opponent_team=opponent,
                    stat_type=prop_type
                )
                
                if injury_data:
                    injury_info = injury_data
                    combined_boost = injury_data.get('combined_injury_boost', 1.0)
                    
                    # Check if the player themselves is OUT
                    if injury_data.get('player_is_out'):
                        # Player is OUT - return warning
                        return {
                            'error': f"{player_name} is listed as OUT",
                            'pick': None,
                            'player_name': player_name,
                            'injury_status': injury_data.get('player_status')
                        }
                    
                    # Apply injury boost
                    if combined_boost != 1.0:
                        injury_boost = combined_boost
                        first_half_est = first_half_est * injury_boost
                        full_game_est = full_game_est * injury_boost
        except ImportError:
            pass  # Injury tracker not available
        except Exception as e:
            pass  # Any error, skip injury adjustment
        
        # =====================================================================
        # ROLE CHANGE DETECTION: If recent form drastically differs from
        # overall average, trust recent form more (player's role is changing)
        # =====================================================================
        role_change_detected = False
        role_change_direction = None  # 'up' or 'down'
        
        if has_real_data and real_period_avg is not None and l5_avg and season_avg and season_avg > 0:
            # Compare recent 5-game full-game avg to season avg
            recent_deviation = (l5_avg - season_avg) / season_avg
            
            if abs(recent_deviation) >= ROLE_CHANGE_THRESHOLD:
                role_change_detected = True
                role_change_direction = 'up' if recent_deviation > 0 else 'down'
                
                # Also check if this player is in the known role-expanding list
                is_known_role_change = player_name in ROLE_EXPANDING_PLAYERS
                
                # Blend: use more recent data
                recent_weight = ROLE_CHANGE_RECENT_WEIGHT if is_known_role_change else 0.55
                
                # Get recent period average from last ~5 games of period data
                try:
                    period_prefix = "1H" if period == '1h' else "1Q"
                    period_file = os.path.join('data', 'period_stats', 
                        f"{player_name.replace(' ', '_')}_period_stats.json")
                    if os.path.exists(period_file):
                        with open(period_file, 'r') as f:
                            pdata = json.load(f)
                        recent_period_games = pdata.get('games_data', [])[-5:]
                        if recent_period_games:
                            recent_period_vals = []
                            for g in recent_period_games:
                                if prop_type == 'pra':
                                    val = (float(g.get(f'{period_prefix}_PTS', 0)) + 
                                           float(g.get(f'{period_prefix}_REB', 0)) + 
                                           float(g.get(f'{period_prefix}_AST', 0)))
                                else:
                                    val = float(g.get(f'{period_prefix}_{stat_col}', 0))
                                recent_period_vals.append(val)
                            if recent_period_vals:
                                recent_5_period_avg = sum(recent_period_vals) / len(recent_period_vals)
                                # Blend recent period avg with overall period avg
                                first_half_est = (recent_5_period_avg * recent_weight + 
                                                 real_period_avg * (1 - recent_weight))
                                full_game_est = first_half_est / ratio if ratio > 0 else first_half_est * 2
                except Exception:
                    pass  # If anything fails, keep original estimate
        
        # =====================================================================
        # TRADE DETECTION: Flag recently traded players
        # =====================================================================
        is_recently_traded = False
        trade_info = RECENTLY_TRADED_PLAYERS.get(player_name)
        if trade_info:
            is_recently_traded = True
            # Reduce confidence - stats from old team may not apply
            # We'll handle this in lock score below
        
        # SANITY CHECK: Cap unrealistic predictions (too high OR too low)
        prediction_warning = None
        prediction_capped = False
        original_full_game = full_game_est
        original_1h = first_half_est
        actual_avg = l5_avg if l5_avg != default_val else season_avg
        
        # Check for unrealistically HIGH predictions
        if actual_avg and actual_avg > 0 and full_game_est > actual_avg * 2:
            prediction_warning = f"⚠️ UNRELIABLE: Model predicts {full_game_est:.1f} but player averages only {actual_avg:.1f}"
            prediction_capped = True
            full_game_est = actual_avg * 1.5
            first_half_est = full_game_est * ratio
        
        # Check for unrealistically LOW predictions (0 or near 0)
        elif full_game_est < 2 and actual_avg and actual_avg > 3:
            prediction_warning = f"⚠️ UNRELIABLE: Model predicts {full_game_est:.1f} but player averages {actual_avg:.1f}"
            prediction_capped = True
            full_game_est = actual_avg * 0.8  # Use 80% of average as floor
            first_half_est = full_game_est * ratio
        
        # Check for no data at all (prediction is 0 or default)
        elif full_game_est <= 0 or (actual_avg is None or actual_avg <= 0):
            prediction_warning = f"⚠️ INSUFFICIENT DATA: Cannot reliably predict for this player"
            prediction_capped = True
            # Use prop line as estimate if no data
            full_game_est = prop_line / ratio
            first_half_est = prop_line
        
        difference = first_half_est - prop_line
        pct_diff = (difference / prop_line) * 100 if prop_line > 0 else 0
        abs_diff = abs(difference)
        is_over = difference > 0
        
        # EXTREME ACCURACY lock score calculation
        lock_score = 50
        lock_factors = []
        
        # REAL DATA BOOST - Having actual 1Q/1H data is the gold standard!
        if has_real_data and data_source == "REAL":
            lock_score += 20
            lock_factors.append({
                'name': '📊 Real Period Data', 
                'score': '+20', 
                'desc': 'Using actual 1Q/1H game data - high accuracy'
            })
        
        # Unreliable prediction penalty
        if prediction_capped:
            lock_score -= 25
            lock_factors.append({
                'name': '🚨 Unreliable Model', 
                'score': '-25', 
                'desc': f'Model predicted {original_full_game:.0f} but player averages {actual_avg:.0f} - capped'
            })
        
        # Edge size factor (0-20 points)
        if abs_diff >= 3:
            lock_score += 20
            lock_factors.append({'name': '🎯 Large Edge', 'score': '+20', 'desc': f'Prediction {abs_diff:.1f} pts from line'})
        elif abs_diff >= 2:
            lock_score += 14
            lock_factors.append({'name': '🎯 Good Edge', 'score': '+14', 'desc': f'Prediction {abs_diff:.1f} pts from line'})
        elif abs_diff >= 1:
            lock_score += 6
            lock_factors.append({'name': '🎯 Small Edge', 'score': '+6', 'desc': f'Prediction {abs_diff:.1f} pts from line'})
        elif abs_diff < 0.5:
            lock_score -= 10
            lock_factors.append({'name': '⚠️ Tiny Edge', 'score': '-10', 'desc': f'Only {abs_diff:.1f} pts from line - risky'})
        
        # Weighted form factor (0-15 points)
        ewm_1h = ewm_5 * ratio if ewm_5 else None
        l5_1h = l5_avg * ratio if l5_avg else None
        
        if ewm_1h:
            if (is_over and ewm_1h > prop_line * 1.15) or (not is_over and ewm_1h < prop_line * 0.85):
                lock_score += 15
                lock_factors.append({'name': '🔥 Strong Form', 'score': '+15', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) strongly supports'})
            elif (is_over and ewm_1h > prop_line * 1.05) or (not is_over and ewm_1h < prop_line * 0.95):
                lock_score += 8
                lock_factors.append({'name': '📈 Good Form', 'score': '+8', 'desc': f'Weighted avg ({ewm_1h:.1f} 1H) supports pick'})
            elif (is_over and ewm_1h < prop_line * 0.95) or (not is_over and ewm_1h > prop_line * 1.05):
                lock_score -= 12
                lock_factors.append({'name': '⚠️ Form Against', 'score': '-12', 'desc': f'Recent form contradicts pick'})
        
        # Momentum factor (0-8 points)
        momentum = features.get(f'{stat_col}_momentum', 0)
        if momentum:
            if (is_over and momentum > 0.1) or (not is_over and momentum < -0.1):
                lock_score += 8
                lock_factors.append({'name': '🚀 Momentum', 'score': '+8', 'desc': f'Trending in right direction ({momentum:.1%})'})
            elif (is_over and momentum < -0.1) or (not is_over and momentum > 0.1):
                lock_score -= 6
                lock_factors.append({'name': '⚠️ Wrong Trend', 'score': '-6', 'desc': 'Momentum against pick'})
        
        # Consistency factor (0-12 points)
        consistency = features.get(f'{stat_col}_consistency')
        if consistency:
            if consistency < 0.2:
                lock_score += 12
                lock_factors.append({'name': '🎯 Consistent', 'score': '+12', 'desc': 'Highly predictable performer'})
            elif consistency > 0.4:
                lock_score -= 10
                lock_factors.append({'name': '🎰 Volatile', 'score': '-10', 'desc': 'High variance - unpredictable'})
        elif l10_std is not None:
            if l10_std < 3:
                lock_score += 10
                lock_factors.append({'name': '🎯 Consistent', 'score': '+10', 'desc': 'Low variance recently'})
            elif l10_std > 7:
                lock_score -= 8
                lock_factors.append({'name': '🎰 Volatile', 'score': '-8', 'desc': 'High variance - unpredictable'})
        
        # Matchup history (0-10 points)
        vs_opp_avg = features.get(f'vs_opp_{stat_col.lower()}_avg')
        vs_opp_games = features.get('vs_opp_games', 0)
        if vs_opp_avg and vs_opp_games >= 3:
            vs_1h = vs_opp_avg * ratio
            if (is_over and vs_1h > prop_line) or (not is_over and vs_1h < prop_line):
                lock_score += 10
                lock_factors.append({'name': '📊 Matchup History', 'score': '+10', 'desc': f'Avg {vs_1h:.1f} 1H in {vs_opp_games} vs this team'})
        
        # Opponent defense factor (0-8 points)
        opp_def_tier = features.get('CURRENT_OPP_DEF_TIER') or features.get('OPP_DEF_TIER')
        if opp_def_tier:
            if is_over and opp_def_tier >= 4:
                lock_score += 8
                lock_factors.append({'name': '🧀 Soft Defense', 'score': '+8', 'desc': f'Facing weak (tier {int(opp_def_tier)}) defense'})
            elif not is_over and opp_def_tier <= 2:
                lock_score += 8
                lock_factors.append({'name': '🔒 Elite Defense', 'score': '+8', 'desc': f'Facing elite (tier {int(opp_def_tier)}) defense'})
            elif is_over and opp_def_tier <= 2:
                lock_score -= 5
                lock_factors.append({'name': '⚠️ Tough Defense', 'score': '-5', 'desc': 'Facing elite defense'})
        
        # Game Pace Factor - high pace = more points
        opponent = features.get('OPPONENT', '')
        player_team = features.get('TEAM', '')
        opp_pace = TEAM_PACE_TIERS.get(opponent, 3)
        team_pace = TEAM_PACE_TIERS.get(player_team, 3)
        combined_pace = (opp_pace + team_pace) / 2
        
        if combined_pace <= 1.5:  # Both fast-paced
            if is_over:
                lock_score += 8
                lock_factors.append({'name': '🏃 High Pace Game', 'score': '+8', 'desc': 'Fast-paced matchup boosts scoring'})
            else:
                lock_score -= 10
                lock_factors.append({'name': '🚨 Pace Risk', 'score': '-10', 'desc': 'Fast-paced game - UNDER risky!'})
        elif combined_pace <= 2.5:
            if is_over:
                lock_score += 4
                lock_factors.append({'name': '🏃 Fast Matchup', 'score': '+4', 'desc': 'Above-average pace game'})
            else:
                lock_score -= 5
                lock_factors.append({'name': '⚠️ Pace Concern', 'score': '-5', 'desc': 'Faster than average matchup'})
        elif combined_pace >= 4.5:  # Both slow
            if not is_over:
                lock_score += 6
                lock_factors.append({'name': '🐢 Slow Pace', 'score': '+6', 'desc': 'Slow-paced game favors UNDER'})
            else:
                lock_score -= 4
                lock_factors.append({'name': '⚠️ Slow Game', 'score': '-4', 'desc': 'Low-pace matchup limits scoring'})
        
        # Floor/ceiling analysis (0-8 points)
        floor = features.get(f'{stat_col}_floor')
        ceiling = features.get(f'{stat_col}_ceiling')
        if floor and ceiling:
            floor_1h = floor * ratio
            ceiling_1h = ceiling * ratio
            if is_over and floor_1h > prop_line:
                lock_score += 8
                lock_factors.append({'name': '🛡️ Safe Floor', 'score': '+8', 'desc': f'Floor ({floor_1h:.1f}) above line'})
            elif not is_over and ceiling_1h < prop_line:
                lock_score += 8
                lock_factors.append({'name': '🛡️ Low Ceiling', 'score': '+8', 'desc': f'Ceiling ({ceiling_1h:.1f}) below line'})
        
        # INJURY IMPACT FACTORS (0-15 points)
        if injury_info:
            teammate_boost = injury_info.get('teammate_out_boost', 1.0)
            opp_boost = injury_info.get('opponent_injury_boost', 1.0)
            out_teammates = injury_info.get('out_teammates', [])
            out_opponents = injury_info.get('out_opponents', [])
            
            # Teammate OUT boost (more usage for this player)
            if teammate_boost > 1.05 and is_over:
                boost_pct = (teammate_boost - 1) * 100
                lock_score += 10
                teammates_str = ', '.join(out_teammates[:2]) if out_teammates else 'star'
                lock_factors.append({
                    'name': '🏥 Teammate OUT', 
                    'score': '+10', 
                    'desc': f'{teammates_str} OUT - {boost_pct:.0f}% usage boost expected'
                })
            elif teammate_boost > 1.02 and is_over:
                lock_score += 5
                lock_factors.append({
                    'name': '🏥 Teammate OUT', 
                    'score': '+5', 
                    'desc': 'Minor teammate absence - slight usage boost'
                })
            
            # Opponent injury boost (easier matchup)
            if opp_boost > 1.05 and is_over:
                boost_pct = (opp_boost - 1) * 100
                lock_score += 8
                opps_str = ', '.join(out_opponents[:2]) if out_opponents else 'defender'
                lock_factors.append({
                    'name': '🏥 Opponent OUT', 
                    'score': '+8', 
                    'desc': f'{opps_str} OUT - easier matchup'
                })
            elif opp_boost > 1.02 and is_over:
                lock_score += 4
                lock_factors.append({
                    'name': '🏥 Opponent OUT', 
                    'score': '+4', 
                    'desc': 'Minor opponent absence - slightly easier matchup'
                })
            
            # Warning for UNDER picks when injuries favor OVER
            if (teammate_boost > 1.05 or opp_boost > 1.05) and not is_over:
                lock_score -= 8
                lock_factors.append({
                    'name': '⚠️ Injury Favors OVER', 
                    'score': '-8', 
                    'desc': 'Injuries suggest higher production - UNDER risky'
                })
        
        # Hit rate from recent games
        if recent_games is not None and len(recent_games) > 0:
            try:
                recent_list = recent_games.to_dict('records') if hasattr(recent_games, 'to_dict') else recent_games
                hits = 0
                total = 0
                for game in recent_list:
                    # Properly handle PRA by summing components
                    if stat_col == 'PRA':
                        full_stat = (game.get('PTS', 0) or 0) + (game.get('REB', 0) or 0) + (game.get('AST', 0) or 0)
                    else:
                        full_stat = game.get(stat_col) or game.get('PTS', 0) or 0
                    est_1h = full_stat * ratio
                    if est_1h > 0 or full_stat > 0:
                        total += 1
                        if est_1h > prop_line:
                            hits += 1
                if total >= 5:
                    hit_rate = hits / total
                    rate_info = f'({total} games vs {prop_line} line)'
                    
                    if is_over:
                        # Predicting OVER - high hit rate good, low is bad
                        if hit_rate > 0.7:
                            lock_score += 8
                            lock_factors.append({'name': '📈 High Hit Rate', 'score': '+8', 'desc': f'{hit_rate:.0%} over rate {rate_info}'})
                        elif hit_rate > 0.5:
                            lock_score += 4
                            lock_factors.append({'name': '📊 Decent Hit Rate', 'score': '+4', 'desc': f'{hit_rate:.0%} over rate {rate_info}'})
                        elif hit_rate < 0.3:
                            lock_score -= 20
                            lock_factors.append({'name': '🚨 Poor Hit Rate', 'score': '-20', 'desc': f'Only {hit_rate:.0%} over rate {rate_info} - risky!'})
                        elif hit_rate < 0.45:
                            lock_score -= 10
                            lock_factors.append({'name': '⚠️ Low Hit Rate', 'score': '-10', 'desc': f'{hit_rate:.0%} over rate {rate_info}'})
                    else:
                        # Predicting UNDER - low hit rate good, high is bad
                        if hit_rate < 0.3:
                            lock_score += 8
                            lock_factors.append({'name': '📉 Low Over Rate', 'score': '+8', 'desc': f'Only {hit_rate:.0%} over rate {rate_info}'})
                        elif hit_rate < 0.5:
                            lock_score += 4
                            lock_factors.append({'name': '📊 Moderate Rate', 'score': '+4', 'desc': f'{hit_rate:.0%} over rate {rate_info}'})
                        elif hit_rate > 0.7:
                            lock_score -= 20
                            lock_factors.append({'name': '🚨 High Over Rate', 'score': '-20', 'desc': f'{hit_rate:.0%} usually goes over {rate_info} - risky!'})
                        elif hit_rate > 0.55:
                            lock_score -= 10
                            lock_factors.append({'name': '⚠️ High Hit Rate', 'score': '-10', 'desc': f'{hit_rate:.0%} over rate {rate_info} - against pick'})
            except Exception as e:
                # Log but continue - hit rate is supplementary
                print(f"[DEBUG] Fallback hit rate error: {e}")
        
        # =====================================================================
        # UNDER PICK VOLATILITY PENALTY (NEW - addresses Wemby/Collier type busts)
        # =====================================================================
        is_star_player = player_name in HIGH_VARIANCE_STARS
        is_role_expanding = player_name in ROLE_EXPANDING_PLAYERS
        
        if not is_over:  # UNDER pick
            # Calculate real 1H standard deviation from period data if available
            real_1h_std = None
            try:
                period_prefix = "1H" if period == '1h' else "1Q"
                period_file = os.path.join('data', 'period_stats', 
                    f"{player_name.replace(' ', '_')}_period_stats.json")
                if os.path.exists(period_file):
                    with open(period_file, 'r') as f:
                        pdata = json.load(f)
                    games = pdata.get('games_data', [])
                    if games:
                        if prop_type == 'pra':
                            vals = [float(g.get(f'{period_prefix}_PTS', 0)) + 
                                   float(g.get(f'{period_prefix}_REB', 0)) + 
                                   float(g.get(f'{period_prefix}_AST', 0)) for g in games]
                        else:
                            stat_key = f'{period_prefix}_{stat_col}'
                            vals = [float(g.get(stat_key, 0)) for g in games]
                        if vals and len(vals) >= 5:
                            mean_val = sum(vals) / len(vals)
                            real_1h_std = (sum((v - mean_val)**2 for v in vals) / len(vals)) ** 0.5
                            # Count how many times they EXCEEDED the line
                            times_over = sum(1 for v in vals if v > prop_line)
                            over_rate = times_over / len(vals)
                            
                            # If player goes OVER this line >40% of the time, UNDER is risky
                            if over_rate >= 0.50:
                                lock_score -= 15
                                lock_factors.append({
                                    'name': '🚨 Real Data Warns OVER',
                                    'score': '-15',
                                    'desc': f'Player exceeds {prop_line} line in {over_rate:.0%} of real {period_prefix} games!'
                                })
                            elif over_rate >= 0.35:
                                lock_score -= 8
                                lock_factors.append({
                                    'name': '⚠️ Frequent Over Risk',
                                    'score': '-8',
                                    'desc': f'Exceeds line in {over_rate:.0%} of real {period_prefix} games'
                                })
                            
                            # Check recent 5 games specifically
                            recent_vals = vals[-5:]
                            recent_over_rate = sum(1 for v in recent_vals if v > prop_line) / len(recent_vals)
                            if recent_over_rate >= 0.60:
                                lock_score -= 10
                                lock_factors.append({
                                    'name': '🔥 Recent Trend Over',
                                    'score': '-10',
                                    'desc': f'Exceeded line in {recent_over_rate:.0%} of last 5 {period_prefix} games'
                                })
            except Exception:
                pass
            
            # Star player UNDER penalty
            if is_star_player:
                penalty = UNDER_VOLATILITY_PENALTY.get('star', -15)
                lock_score += penalty  # penalty is negative
                lock_factors.append({
                    'name': '⭐ Star Player UNDER Risk',
                    'score': str(penalty),
                    'desc': f'{player_name} is a high-variance star - UNDER is inherently risky'
                })
                
                # Check if edge is large enough for a star UNDER
                min_edge_star = UNDER_MIN_EDGE.get('star', 3.5)
                if abs_diff < min_edge_star:
                    lock_score -= 10
                    lock_factors.append({
                        'name': '🚨 Edge Too Small for Star UNDER',
                        'score': '-10',
                        'desc': f'Need {min_edge_star}+ pt edge for UNDER on stars, only have {abs_diff:.1f}'
                    })
            
            # Role-expanding player UNDER penalty
            elif is_role_expanding:
                penalty = UNDER_VOLATILITY_PENALTY.get('role_expanding', -8)
                lock_score += penalty
                lock_factors.append({
                    'name': '📈 Role Expanding - UNDER Risk',
                    'score': str(penalty),
                    'desc': f'{player_name} is getting increased role - UNDER risky'
                })
                
                min_edge_role = UNDER_MIN_EDGE.get('role_expanding', 2.5)
                if abs_diff < min_edge_role:
                    lock_score -= 8
                    lock_factors.append({
                        'name': '⚠️ Edge Too Small for Expanding Role',
                        'score': '-8',
                        'desc': f'Need {min_edge_role}+ pt edge, only have {abs_diff:.1f}'
                    })
        
        # OVER pick on star player - use real hit rate from period data
        elif is_over and (is_star_player or is_role_expanding):
            try:
                period_prefix = "1H" if period == '1h' else "1Q"
                period_file = os.path.join('data', 'period_stats', 
                    f"{player_name.replace(' ', '_')}_period_stats.json")
                if os.path.exists(period_file):
                    with open(period_file, 'r') as f:
                        pdata = json.load(f)
                    games = pdata.get('games_data', [])
                    if games:
                        if prop_type == 'pra':
                            vals = [float(g.get(f'{period_prefix}_PTS', 0)) + 
                                   float(g.get(f'{period_prefix}_REB', 0)) + 
                                   float(g.get(f'{period_prefix}_AST', 0)) for g in games]
                        else:
                            stat_key = f'{period_prefix}_{stat_col}'
                            vals = [float(g.get(stat_key, 0)) for g in games]
                        if vals and len(vals) >= 5:
                            times_over = sum(1 for v in vals if v > prop_line)
                            over_rate = times_over / len(vals)
                            if over_rate >= 0.70:
                                lock_score += 8
                                lock_factors.append({
                                    'name': '📊 Strong Real Over Rate',
                                    'score': '+8',
                                    'desc': f'Exceeds line in {over_rate:.0%} of real {period_prefix} games'
                                })
            except Exception:
                pass
        
        # =====================================================================
        # TRADE DETECTION PENALTY (NEW - addresses Luka/CJ on new teams)
        # =====================================================================
        if is_recently_traded:
            lock_score -= 15
            trade_info_msg = RECENTLY_TRADED_PLAYERS.get(player_name, {})
            lock_factors.append({
                'name': '🔄 Recently Traded',
                'score': '-15',
                'desc': f'Traded from {trade_info_msg.get("old_team", "?")} to {trade_info_msg.get("new_team", "?")} - stats may not apply in new system'
            })
        
        # =====================================================================
        # ROLE CHANGE DETECTION FACTOR
        # =====================================================================
        if role_change_detected:
            if role_change_direction == 'up' and not is_over:
                # Role expanding + picking UNDER = dangerous
                lock_score -= 10
                lock_factors.append({
                    'name': '📈 Role Expanding - UNDER Danger',
                    'score': '-10',
                    'desc': f'Recent form {abs(((l5_avg - season_avg) / season_avg)):.0%} above season avg - role increasing'
                })
            elif role_change_direction == 'down' and is_over:
                # Role shrinking + picking OVER = dangerous
                lock_score -= 10
                lock_factors.append({
                    'name': '📉 Role Shrinking - OVER Danger',
                    'score': '-10',
                    'desc': f'Recent form below season avg - may be losing minutes/role'
                })
            elif role_change_direction == 'up' and is_over:
                lock_score += 5
                lock_factors.append({
                    'name': '📈 Role Expanding',
                    'score': '+5',
                    'desc': 'Player trending up - supports OVER'
                })
        
        # B2B factor (0-6 points)
        if features.get('IS_B2B'):
            b2b_impact = features.get(f'{stat_col}_b2b_impact')
            if not is_over:
                pts = 6 if b2b_impact and b2b_impact < 0.92 else 4
                lock_score += pts
                lock_factors.append({'name': '😴 B2B Fatigue', 'score': f'+{pts}', 'desc': 'B2B typically reduces output'})
            else:
                pts = 6 if b2b_impact and b2b_impact < 0.92 else 4
                lock_score -= pts
                lock_factors.append({'name': '😴 B2B Risk', 'score': f'-{pts}', 'desc': 'B2B may limit production'})
        elif features.get('IS_WELL_RESTED') and is_over:
            lock_score += 4
            lock_factors.append({'name': '💪 Well Rested', 'score': '+4', 'desc': f'{int(features.get("REST_DAYS", 3))} days rest'})
        
        # Sample size penalty (NEW)
        games_played = len(recent_games) if recent_games is not None else 0
        if games_played < 5:
            lock_score -= 20
            lock_factors.append({'name': '🚨 Limited Data', 'score': '-20', 'desc': f'Only {games_played} games - very risky'})
        elif games_played < 10:
            lock_score -= 10
            lock_factors.append({'name': '⚠️ Small Sample', 'score': '-10', 'desc': f'Only {games_played} games of data'})
        elif games_played < 15:
            lock_score -= 5
            lock_factors.append({'name': '📊 Limited Sample', 'score': '-5', 'desc': f'{games_played} games - moderate confidence'})
        
        # Regression to mean factor (NEW)
        if l5_avg and season_avg and season_avg > 0:
            deviation_pct = (ewm_5 - season_avg) / season_avg if ewm_5 else (l5_avg - season_avg) / season_avg
            
            if deviation_pct > 0.25:  # Hot streak
                if is_over:
                    lock_score -= 8
                    lock_factors.append({'name': '📉 Regression Risk', 'score': '-8', 'desc': f'Recent {deviation_pct:.0%} above avg - may cool'})
                else:
                    lock_score += 6
                    lock_factors.append({'name': '📉 Due for Regression', 'score': '+6', 'desc': f'Unsustainable {deviation_pct:.0%} above avg'})
            elif deviation_pct < -0.25:  # Cold streak
                if not is_over:
                    lock_score -= 8
                    lock_factors.append({'name': '📈 Regression Risk', 'score': '-8', 'desc': f'Recent slump may end'})
                else:
                    lock_score += 6
                    lock_factors.append({'name': '📈 Due for Bounce', 'score': '+6', 'desc': f'Due to revert from slump'})
        
        # Suspicious edge check (NEW)
        if abs_diff >= 7.0:
            lock_score -= 15
            lock_factors.append({'name': '🚨 Suspicious Edge', 'score': '-15', 'desc': f'{abs_diff:.1f} pts edge is abnormally large'})
        elif abs_diff >= 5.0:
            lock_score -= 8
            lock_factors.append({'name': '⚠️ Large Edge Warning', 'score': '-8', 'desc': f'{abs_diff:.1f} pts edge unusual'})
        
        # Variance-adjusted confidence (NEW)
        if l10_std and l10_std > 0:
            std_ratio = l10_std * ratio
            if std_ratio > 0:
                edge_in_stds = abs_diff / std_ratio
                if edge_in_stds < 0.5:
                    lock_score -= 10
                    lock_factors.append({'name': '⚠️ Edge Within Noise', 'score': '-10', 'desc': f'Edge < 0.5 std devs - coinflip'})
                elif edge_in_stds >= 1.5:
                    lock_score += 8
                    lock_factors.append({'name': '📊 Statistically Significant', 'score': '+8', 'desc': f'Edge is {edge_in_stds:.1f} std devs'})
        
        # Line proximity to average (NEW)
        if season_avg:
            season_1h = season_avg * ratio
            line_vs_avg_diff = abs(prop_line - season_1h)
            if line_vs_avg_diff < 0.5:
                lock_score -= 8
                lock_factors.append({'name': '⚖️ Tight Line', 'score': '-8', 'desc': f'Line very close to avg ({season_1h:.1f}) - coinflip'})
        
        lock_score = max(0, min(100, lock_score))
        
        # Apply 1Q-specific adjustments if this is a 1Q prediction
        if period == '1q':
            # Add games_played to features for 1Q adjustment
            recent_list = None
            if recent_games is not None:
                recent_list = recent_games.to_dict('records') if hasattr(recent_games, 'to_dict') else recent_games
                features['games_played'] = len(recent_list) if recent_list else 0
            else:
                features['games_played'] = 0
            features['_prop_type'] = prop_type  # Pass prop type for correct stat lookup
            
            lock_score, lock_factors = apply_1q_adjustments(
                lock_score, lock_factors, features, prop_line, difference, period,
                player_name=features.get('PLAYER_NAME', player_name),
                recent_games=recent_list,
                has_real_period_data=has_real_data
            )
        
        # Determine pick based on lock score - USE PERIOD-SPECIFIC THRESHOLDS
        thresholds = get_lock_thresholds(period)
        
        if lock_score >= thresholds['lock']:
            pick = "🔒 LOCK " + ("OVER" if is_over else "UNDER")
            confidence = "LOCK"
            confidence_desc = f"Extremely high confidence for {period_label}"
        elif lock_score >= thresholds['strong']:
            pick = "🔥 STRONG " + ("OVER" if is_over else "UNDER")
            confidence = "HIGH"
            confidence_desc = f"High confidence for {period_label}"
        elif lock_score >= thresholds['playable']:
            pick = "✅ " + ("OVER" if is_over else "UNDER")
            confidence = "MEDIUM"
            confidence_desc = f"Moderate confidence for {period_label}"
        elif lock_score >= thresholds['lean']:
            pick = "⚠️ LEAN " + ("OVER" if is_over else "UNDER")
            confidence = "LOW"
            confidence_desc = f"Low confidence for {period_label} - proceed with caution"
        else:
            pick = "❓ SKIP"
            confidence = "AVOID"
            confidence_desc = f"Too risky for {period_label} - insufficient edge"
        
        # Add 1Q warning if applicable
        if period == '1q' and confidence not in ['LOCK', 'AVOID']:
            confidence_desc += " (1Q requires higher standards)"
        
        # Generate detailed reasons
        reasons = []
        if ewm_1h:
            reasons.append(f"Weighted recent avg: {ewm_5:.1f} full game → {ewm_1h:.1f} {period_label}")
        if l5_1h and l5_avg != default_val:
            reasons.append(f"Last 5 games: {l5_avg:.1f} full game → {l5_1h:.1f} {period_label}")
        if season_avg and season_avg != l5_avg:
            season_1h = season_avg * ratio
            reasons.append(f"Season baseline: {season_avg:.1f} full game → {season_1h:.1f} {period_label}")
        
        reasons.append(f"📊 Model prediction: {first_half_est:.1f} {period_label} vs line {prop_line} (edge: {difference:+.1f})")
        
        if features.get('IS_B2B'):
            reasons.append("😴 Back-to-back game - fatigue factor")
        
        loc_emoji = "🏠" if features.get('IS_HOME') else "✈️"
        reasons.append(f"{loc_emoji} {'Home' if features.get('IS_HOME') else 'Road'} game vs {features.get('OPPONENT', 'OPP')}")
        
        if opp_def_tier:
            tier_desc = {1: "Elite", 2: "Good", 3: "Average", 4: "Below Avg", 5: "Poor"}
            reasons.append(f"🛡️ Opponent defense: {tier_desc.get(int(opp_def_tier), 'Unknown')} (Tier {int(opp_def_tier)})")
        
        # Add warning to reasons if prediction was capped
        if prediction_capped:
            reasons.insert(0, prediction_warning)
        
        result = {
            'pick': pick,
            'confidence': confidence,
            'confidence_desc': confidence_desc,
            'lock_score': lock_score,
            'lock_factors': lock_factors,
            'predicted_1h': round(first_half_est, 1),
            'prop_line': prop_line,
            'difference': round(difference, 1),
            'pct_difference': round(pct_diff, 1),
            'full_game_prediction': round(full_game_est, 1),
            'reasons': reasons,
            'player_name': features.get('PLAYER_NAME', player_name),
            'recent_games': recent_games.to_dict('records') if recent_games is not None and hasattr(recent_games, 'to_dict') else (recent_games if recent_games else []),
            'model_type': 'extreme_accuracy',
            'period': period,
            'period_label': period_label,
            'data_source': data_source,  # "REAL" or "ESTIMATED"
            'has_real_period_data': has_real_data,
            'player_period_ratio': round(ratio, 4) if ratio else None,
            'advanced_boost': round(advanced_boost, 3) if advanced_boost != 1.0 else None,
            'advanced_boost_info': advanced_boost_info,
            'injury_boost': round(injury_boost, 3) if injury_boost != 1.0 else None,
            'injury_info': {
                'out_teammates': injury_info.get('out_teammates', []) if injury_info else [],
                'out_opponents': injury_info.get('out_opponents', []) if injury_info else [],
                'teammate_boost': round(injury_info.get('teammate_out_boost', 1.0), 3) if injury_info else 1.0,
                'opponent_boost': round(injury_info.get('opponent_injury_boost', 1.0), 3) if injury_info else 1.0,
            } if injury_info and (injury_info.get('out_teammates') or injury_info.get('out_opponents')) else None
        }
        
        # Include warning info if prediction was capped
        if prediction_capped:
            result['prediction_warning'] = prediction_warning
            result['original_prediction'] = round(original_1h, 1)
            result['original_full_game'] = round(original_full_game, 1)
        
        return result
    
    # Use trained model - pass recent_games for accurate hit rate calculation
    recent_games_list = recent_games.to_dict('records') if recent_games is not None and hasattr(recent_games, 'to_dict') else (recent_games if recent_games else [])
    result = model.predict_over_under(features, prop_line, recent_games=recent_games_list, period=period)
    result['player_name'] = features.get('PLAYER_NAME', player_name)
    result['recent_games'] = recent_games_list
    result['model_type'] = 'ensemble'
    result['period'] = period
    result['period_label'] = "1Q" if period == '1q' else "1H"
    
    # =====================================================================
    # INJURY IMPACT ADJUSTMENT (Trained Model Path)
    # =====================================================================
    try:
        from injury_tracker import get_injury_features
        
        # Get player's team from features
        player_team = None
        matchup = features.get('MATCHUP', '')
        if matchup:
            if '@' in matchup:
                player_team = matchup.split('@')[0].strip()
            elif 'vs.' in matchup:
                player_team = matchup.split('vs.')[0].strip()
            if player_team and ' ' in player_team:
                player_team = player_team.split()[-1]
        
        if not player_team and recent_games_list:
            last_matchup = recent_games_list[-1].get('MATCHUP', '') if recent_games_list else ''
            if last_matchup:
                if '@' in last_matchup:
                    player_team = last_matchup.split('@')[0].strip()
                elif 'vs.' in last_matchup:
                    player_team = last_matchup.split('vs.')[0].strip()
                if player_team and ' ' in player_team:
                    player_team = player_team.split()[-1]
        
        if player_team and opponent:
            injury_data = get_injury_features(
                player_name=player_name,
                player_team=player_team,
                opponent_team=opponent,
                stat_type=prop_type
            )
            
            if injury_data:
                # Check if player is OUT
                if injury_data.get('player_is_out'):
                    return {
                        'error': f"{player_name} is listed as OUT",
                        'pick': None,
                        'player_name': player_name,
                        'injury_status': injury_data.get('player_status')
                    }
                
                # Apply injury boost to prediction
                combined_boost = injury_data.get('combined_injury_boost', 1.0)
                if combined_boost != 1.0:
                    # Update prediction values
                    if 'predicted_1h' in result:
                        original_pred = result['predicted_1h']
                        result['predicted_1h'] = round(original_pred * combined_boost, 1)
                    if 'full_game_prediction' in result:
                        original_full = result['full_game_prediction']
                        result['full_game_prediction'] = round(original_full * combined_boost, 1)
                    if 'difference' in result and 'prop_line' in result:
                        result['difference'] = round(result['predicted_1h'] - result['prop_line'], 1)
                    
                    result['injury_boost'] = round(combined_boost, 3)
                    result['injury_info'] = {
                        'out_teammates': injury_data.get('out_teammates', []),
                        'out_opponents': injury_data.get('out_opponents', []),
                        'teammate_boost': round(injury_data.get('teammate_out_boost', 1.0), 3),
                        'opponent_boost': round(injury_data.get('opponent_injury_boost', 1.0), 3),
                    }
                    
                    # Adjust lock_score based on injuries
                    is_over = 'OVER' in result.get('pick', '')
                    lock_score = result.get('lock_score', 50)
                    lock_factors = result.get('lock_factors', [])
                    teammate_boost = injury_data.get('teammate_out_boost', 1.0)
                    opp_boost = injury_data.get('opponent_injury_boost', 1.0)
                    out_teammates = injury_data.get('out_teammates', [])
                    out_opponents = injury_data.get('out_opponents', [])
                    
                    # Boost for OVER picks when injuries help
                    if teammate_boost > 1.05 and is_over:
                        boost_pct = (teammate_boost - 1) * 100
                        lock_score += 10
                        teammates_str = ', '.join(out_teammates[:2]) if out_teammates else 'star'
                        lock_factors.append({
                            'name': '🏥 Teammate OUT', 
                            'score': '+10', 
                            'desc': f'{teammates_str} OUT - {boost_pct:.0f}% usage boost'
                        })
                    elif teammate_boost > 1.02 and is_over:
                        lock_score += 5
                        lock_factors.append({
                            'name': '🏥 Teammate OUT', 
                            'score': '+5', 
                            'desc': 'Minor teammate absence'
                        })
                    
                    if opp_boost > 1.05 and is_over:
                        lock_score += 8
                        opps_str = ', '.join(out_opponents[:2]) if out_opponents else 'defender'
                        lock_factors.append({
                            'name': '🏥 Opponent OUT', 
                            'score': '+8', 
                            'desc': f'{opps_str} OUT - easier matchup'
                        })
                    elif opp_boost > 1.02 and is_over:
                        lock_score += 4
                        lock_factors.append({
                            'name': '🏥 Opponent OUT', 
                            'score': '+4', 
                            'desc': 'Slightly easier matchup'
                        })
                    
                    # Penalty for UNDER when injuries favor OVER
                    if (teammate_boost > 1.05 or opp_boost > 1.05) and not is_over:
                        lock_score -= 8
                        lock_factors.append({
                            'name': '⚠️ Injury Favors OVER', 
                            'score': '-8', 
                            'desc': 'Injuries suggest higher production - UNDER risky'
                        })
                    
                    result['lock_score'] = min(100, max(0, lock_score))
                    result['lock_factors'] = lock_factors
    except ImportError:
        pass  # Injury tracker not available
    except Exception:
        pass  # Any error, skip injury adjustment
    
    # =====================================================================
    # STAR PLAYER / TRADE / ROLE CHANGE ADJUSTMENTS (Trained Model Path)
    # These protect against common failure modes regardless of model path
    # =====================================================================
    try:
        is_over_trained = 'OVER' in result.get('pick', '')
        lock_score = result.get('lock_score', 50)
        lock_factors = result.get('lock_factors', [])
        predicted_1h = result.get('predicted_1h', 0)
        prop_line_val = result.get('prop_line', prop_line)
        abs_diff = abs(predicted_1h - prop_line_val)
        
        is_star = player_name in HIGH_VARIANCE_STARS
        is_expanding = player_name in ROLE_EXPANDING_PLAYERS
        is_traded = player_name in RECENTLY_TRADED_PLAYERS
        
        stat_col_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pra': 'PRA'}
        stat_col_t = stat_col_map.get(prop_type, 'PTS')
        period_prefix = "1H" if period == '1h' else "1Q"
        
        # --- UNDER PICK PROTECTIONS ---
        if not is_over_trained:
            # Read real period data for hit rate analysis
            try:
                period_file = os.path.join('data', 'period_stats', 
                    f"{player_name.replace(' ', '_')}_period_stats.json")
                if os.path.exists(period_file):
                    with open(period_file, 'r') as f:
                        pdata = json.load(f)
                    games = pdata.get('games_data', [])
                    if games:
                        if prop_type == 'pra':
                            vals = [float(g.get(f'{period_prefix}_PTS', 0)) + 
                                   float(g.get(f'{period_prefix}_REB', 0)) + 
                                   float(g.get(f'{period_prefix}_AST', 0)) for g in games]
                        else:
                            vals = [float(g.get(f'{period_prefix}_{stat_col_t}', 0)) for g in games]
                        
                        if vals and len(vals) >= 5:
                            times_over = sum(1 for v in vals if v > prop_line_val)
                            over_rate = times_over / len(vals)
                            
                            if over_rate >= 0.50:
                                lock_score -= 15
                                lock_factors.append({
                                    'name': '🚨 Real Data Warns OVER',
                                    'score': '-15',
                                    'desc': f'Player exceeds {prop_line_val} line in {over_rate:.0%} of real {period_prefix} games!'
                                })
                            elif over_rate >= 0.35:
                                lock_score -= 8
                                lock_factors.append({
                                    'name': '⚠️ Frequent Over Risk',
                                    'score': '-8',
                                    'desc': f'Exceeds line in {over_rate:.0%} of real {period_prefix} games'
                                })
                            
                            recent_vals = vals[-5:]
                            recent_over_rate = sum(1 for v in recent_vals if v > prop_line_val) / len(recent_vals)
                            if recent_over_rate >= 0.60:
                                lock_score -= 10
                                lock_factors.append({
                                    'name': '🔥 Recent Trend Over',
                                    'score': '-10',
                                    'desc': f'Exceeded line in {recent_over_rate:.0%} of last 5 {period_prefix} games'
                                })
            except Exception:
                pass
            
            # Star player UNDER penalty
            if is_star:
                penalty = UNDER_VOLATILITY_PENALTY.get('star', -15)
                lock_score += penalty
                lock_factors.append({
                    'name': '⭐ Star Player UNDER Risk',
                    'score': str(penalty),
                    'desc': f'{player_name} is a high-variance star - UNDER is inherently risky'
                })
                min_edge = UNDER_MIN_EDGE.get('star', 3.5)
                if abs_diff < min_edge:
                    lock_score -= 10
                    lock_factors.append({
                        'name': '🚨 Edge Too Small for Star UNDER',
                        'score': '-10',
                        'desc': f'Need {min_edge}+ pt edge for UNDER on stars, only have {abs_diff:.1f}'
                    })
            
            # Role-expanding player UNDER penalty
            elif is_expanding:
                penalty = UNDER_VOLATILITY_PENALTY.get('role_expanding', -8)
                lock_score += penalty
                lock_factors.append({
                    'name': '📈 Role Expanding - UNDER Risk',
                    'score': str(penalty),
                    'desc': f'{player_name} is getting increased role - UNDER risky'
                })
                min_edge = UNDER_MIN_EDGE.get('role_expanding', 2.5)
                if abs_diff < min_edge:
                    lock_score -= 8
                    lock_factors.append({
                        'name': '⚠️ Edge Too Small for Expanding Role',
                        'score': '-8',
                        'desc': f'Need {min_edge}+ pt edge, only have {abs_diff:.1f}'
                    })
        
        # --- OVER PICK: use real data hit rate for confirmation ---
        elif is_over_trained and (is_star or is_expanding):
            try:
                period_file = os.path.join('data', 'period_stats', 
                    f"{player_name.replace(' ', '_')}_period_stats.json")
                if os.path.exists(period_file):
                    with open(period_file, 'r') as f:
                        pdata = json.load(f)
                    games = pdata.get('games_data', [])
                    if games:
                        if prop_type == 'pra':
                            vals = [float(g.get(f'{period_prefix}_PTS', 0)) + 
                                   float(g.get(f'{period_prefix}_REB', 0)) + 
                                   float(g.get(f'{period_prefix}_AST', 0)) for g in games]
                        else:
                            vals = [float(g.get(f'{period_prefix}_{stat_col_t}', 0)) for g in games]
                        if vals and len(vals) >= 5:
                            times_over = sum(1 for v in vals if v > prop_line_val)
                            over_rate = times_over / len(vals)
                            if over_rate >= 0.70:
                                lock_score += 8
                                lock_factors.append({
                                    'name': '📊 Strong Real Over Rate',
                                    'score': '+8',
                                    'desc': f'Exceeds line in {over_rate:.0%} of real {period_prefix} games'
                                })
            except Exception:
                pass
        
        # --- TRADE DETECTION ---
        if is_traded:
            lock_score -= 15
            trade_data = RECENTLY_TRADED_PLAYERS.get(player_name, {})
            lock_factors.append({
                'name': '🔄 Recently Traded',
                'score': '-15',
                'desc': f'Traded from {trade_data.get("old_team", "?")} to {trade_data.get("new_team", "?")} - stats may not apply'
            })
        
        # --- ROLE CHANGE DETECTION ---
        # Check if recent form significantly deviates from season average
        try:
            l5_avg_t = features.get(f'{stat_col_t}_L5_avg')
            season_avg_t = features.get(f'{stat_col_t}_season_avg')
            if l5_avg_t and season_avg_t and season_avg_t > 0:
                deviation = (l5_avg_t - season_avg_t) / season_avg_t
                if abs(deviation) >= ROLE_CHANGE_THRESHOLD:
                    if deviation > 0 and not is_over_trained:
                        lock_score -= 10
                        lock_factors.append({
                            'name': '📈 Role Expanding - UNDER Danger',
                            'score': '-10',
                            'desc': f'Recent form {abs(deviation):.0%} above season avg - role increasing'
                        })
                    elif deviation < 0 and is_over_trained:
                        lock_score -= 10
                        lock_factors.append({
                            'name': '📉 Role Shrinking - OVER Danger',
                            'score': '-10',
                            'desc': f'Recent form below season avg - may be losing role'
                        })
                    elif deviation > 0 and is_over_trained:
                        lock_score += 5
                        lock_factors.append({
                            'name': '📈 Role Expanding',
                            'score': '+5',
                            'desc': 'Player trending up - supports OVER'
                        })
        except Exception:
            pass
        
        # Update result with adjusted scores
        result['lock_score'] = min(100, max(0, lock_score))
        result['lock_factors'] = lock_factors
        
        # Re-evaluate pick label if lock score changed significantly
        thresholds = LOCK_THRESHOLDS_1Q if period == '1q' else LOCK_THRESHOLDS
        new_score = result['lock_score']
        direction = "OVER" if is_over_trained else "UNDER"
        if new_score >= thresholds['lock']:
            result['pick'] = f"🔒 LOCK {direction}"
            result['confidence'] = "LOCK"
        elif new_score >= thresholds['strong']:
            result['pick'] = f"🔥 STRONG {direction}"
            result['confidence'] = "HIGH"
        elif new_score >= thresholds['playable']:
            result['pick'] = f"✅ {direction}"
            result['confidence'] = "MEDIUM"
        elif new_score >= thresholds['lean']:
            result['pick'] = f"⚠️ LEAN {direction}"
            result['confidence'] = "LOW"
        else:
            result['pick'] = "❓ SKIP"
            result['confidence'] = "AVOID"
    except Exception as e:
        pass  # Don't break existing functionality
    
    return result


if __name__ == "__main__":
    # Test prediction
    print("Testing prediction system...")
    
    result = make_prediction(
        player_name="LeBron James",
        opponent="GSW",
        prop_line=12.5,
        prop_type="points",
        is_home=True
    )
    
    print("\n" + "="*50)
    print(f"Player: {result.get('player_name', 'Unknown')}")
    print(f"Prop: {result.get('prop_line')} 1H Points")
    print(f"Prediction: {result.get('predicted_1h')} 1H Points")
    print(f"Pick: {result.get('pick')} ({result.get('confidence')} confidence)")
    print("="*50)
    
    print("\nReasons:")
    for reason in result.get('reasons', []):
        print(f"  - {reason}")
