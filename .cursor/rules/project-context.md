# NBA PrizePicks 1H Prop Predictor - Project Context

## What This Project Is
A PrizePicks NBA 1st Half (1H) and 1st Quarter (1Q) player prop prediction model using an ensemble of LightGBM, XGBoost, and CatBoost. Built with a Streamlit UI.

## Architecture
- `app.py` - Streamlit web app with 5 tabs (Single, Batch, Compare, Today's Games, History)
- `model.py` - Ensemble model + "extreme accuracy" estimation path (used when model isn't trained for a stat)
- `config.py` - All thresholds, team tiers, player lists, model hyperparameters
- `feature_engineering.py` - Feature creation from NBA API data
- `data_collector.py` - NBA API data fetching + caching
- `injury_tracker.py` - ESPN injury scraping, injury boost calculations
- `period_boxscore_collector.py` - Collects actual 1Q/1H box score data
- `train_model.py` - Training script for the ensemble models
- `data/period_stats/` - Real 1Q/1H player stats JSON files (gold standard data)
- `models/` - Trained model files (ensemble_points.joblib, ensemble_pra.joblib)

## Key Model Concepts
- **Lock Score (0-100)**: Composite confidence score. Factors include edge size, form, consistency, matchup history, pace, defense tier, momentum, hit rate, injuries, etc.
- **Thresholds**: LOCK (85+), STRONG (72+), PLAYABLE (55+), LEAN (40+), SKIP (<40)
- **Period Data**: We have real 1Q/1H stats for 527+ players. This is the gold standard vs estimated ratios.
- **Two model paths**: Trained model path (uses joblib models) and estimation path (uses weighted averages + period data). Both paths now have star player, trade, and role-change protections.

## Recent Improvements (Feb 5, 2026)
After analyzing busted PrizePicks bets, the following protections were added:

### 1. Star Player UNDER Penalty
- `HIGH_VARIANCE_STARS` list in config.py - stars whose UNDER picks are inherently risky
- -15 lock score penalty for UNDER on stars
- Additional -10 if edge < 3.5 pts for star UNDERs

### 2. Role-Expanding Player Detection
- `ROLE_EXPANDING_PLAYERS` list in config.py - young players getting more minutes
- -8 lock score for UNDER, +5 for OVER on expanding players
- Role change detection: if L5 avg deviates >30% from season avg, adjusts prediction

### 3. Trade Detection
- `RECENTLY_TRADED_PLAYERS` dict in config.py - players on new teams
- -15 lock score when player recently traded (stats may not apply)
- Remove players once they have 15+ games on new team

### 4. Real Data Hit Rate for UNDER Picks
- Reads actual period_stats data to check how often player exceeds the line
- -15 if player goes over the line in 50%+ of real games
- -10 if recent 5 games show 60%+ over rate

### 5. Role Change Prediction Adjustment
- When recent form deviates >30% from season avg, blends recent period data more heavily
- Known role-expanding players get 70% weight to recent 5 games

## Critical Bug Fixed (Feb 5, 2026)
- `get_player_period_average()` and `get_player_period_ratio()` in model.py had a bug where PRA predictions only used POINTS instead of PTS+REB+AST. This caused ALL PRA predictions to be drastically wrong (e.g. Wemby predicted 11.2 instead of 17.0). Fixed by adding proper PRA handling that sums all three stats.

## Known Issues / Future Work
- The `RECENTLY_TRADED_PLAYERS` list needs manual maintenance
- Ideally should auto-detect trades from roster/team data
- CJ McCollum OVER still got a high score (85) despite thin 0.9 pt edge - could add a "minimum absolute edge" requirement
- Consider retraining models with latest period stats data for better accuracy

## Betting Lessons Learned (Feb 5, 2026)
- NEVER take UNDER on star/volatile players unless edge is massive (3.5+ pts)
- Watch for role-expanding young players - their averages lag behind current form
- Recently traded players need extra caution - new system = different production
- Thin edges (<1.5 pts) are essentially coinflips regardless of model score
- For Power Plays, all legs must hit - avoid volatile players in multi-pick entries
- Flex Plays are safer for including moderate-confidence picks

## Daily Workflow
1. Run `python update_all.py` to refresh data
2. Run `streamlit run app.py` to start the UI
3. Use Batch Predictions for tonight's games
4. Filter for Score 85+ AND Edge 2.5+ for best picks
5. Update `RECENTLY_TRADED_PLAYERS` if new trades happen
