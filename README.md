# NBA 1st Half Prop Predictor

An ensemble machine learning model for predicting NBA player 1st half Points and PRA (Points + Rebounds + Assists) based on PrizePicks prop lines.

## Features

- **Ensemble Model**: Combines LightGBM (50%), XGBoost (30%), and CatBoost (20%)
- **Real-time Data**: Uses NBA API to fetch current season stats
- **Smart Features**: Rolling averages, matchup history, home/away splits, rest days
- **Streamlit UI**: Beautiful, interactive web interface
- **Detailed Analysis**: Explains reasoning behind each prediction

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App (Quick Start - No Training Required)

The app works immediately with intelligent estimation based on player stats:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 3. (Optional) Train the Full Model

For better predictions, train the ensemble model with historical data:

```bash
# Collect data for 100 players and train
python train_model.py --collect --train --players 100

# Or just collect data
python train_model.py --collect --players 200

# Or just train with existing data
python train_model.py --train --target both
```

## How to Use

1. **Enter Player Name**: Type the player's full name (e.g., "LeBron James")
2. **Select Prop Type**: Choose between 1H Points or 1H PRA
3. **Enter Prop Line**: Input the PrizePicks line you're considering
4. **Select Matchup**: Choose the opponent team and home/away
5. **Get Prediction**: Click the button to see the analysis

## Model Details

### Architecture

```
Ensemble Model
├── LightGBM (50% weight) - Primary model
│   └── 500 estimators, max_depth=8
├── XGBoost (30% weight)
│   └── 300 estimators, max_depth=6
└── CatBoost (20% weight)
    └── 300 iterations, depth=6
```

### Feature Engineering

The model uses these features:
- **Rolling Stats**: Last 3, 5, 10, 15 game averages
- **Season Averages**: Points, rebounds, assists, minutes
- **Home/Away Splits**: Performance by location
- **Rest Days**: Days between games, back-to-back indicator
- **Opponent History**: Historical performance vs specific teams
- **Variance Metrics**: Standard deviation to measure consistency

### 1st Half Estimation

Since the NBA API provides full-game stats, we estimate 1H performance using historical ratios:
- **Points**: ~48% of full-game points scored in 1H
- **Rebounds**: ~47% in 1H
- **Assists**: ~49% in 1H
- **PRA**: ~48% in 1H

## Project Structure

```
nba half/
├── app.py                 # Streamlit web application
├── model.py               # Ensemble model implementation
├── feature_engineering.py # Feature creation and processing
├── data_collector.py      # NBA API data fetching
├── config.py              # Configuration and settings
├── train_model.py         # Training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Cached data (auto-created)
│   ├── player_logs/       # Player game logs
│   └── team_stats/        # Team defensive stats
├── models/                # Saved models (auto-created)
└── cache/                 # API cache (auto-created)
```

## API Rate Limiting

The NBA API has rate limits. The code includes:
- 0.6 second delays between API calls
- Caching of responses (6-24 hours depending on data type)
- Automatic retry logic

## Prediction Output

Each prediction includes:

1. **Pick**: OVER/UNDER with strength indicator (STRONG, regular, or LEAN)
2. **Confidence**: HIGH, MEDIUM, or LOW
3. **Predicted 1H Value**: Model's estimated 1st half stat
4. **Reasoning**: Multiple factors explaining the prediction
5. **Model Breakdown**: Individual predictions from each model
6. **Recent Games**: Last 10 games with estimated 1H performance

## Example Prediction

```
Player: LeBron James
Prop: 1H Points 12.5
Opponent: GSW (Home)

PREDICTION: OVER (MEDIUM confidence)
Predicted 1H: 14.2 points
Difference: +1.7 vs line

Reasons:
- Recent form strong: averaging 27.3 points/game (est. 13.1 1H) over last 5
- Season average (26.1) suggests 1H of 12.5, at line
- Home game advantage: averages 27.8 at home vs 24.9 on road
- Historical vs opponent: 28.4 points/game (est. 13.6 1H)
- All 3 models agree on OVER
```

## Notes

- **For Entertainment Only**: This is a predictive model and does not guarantee wins
- **Gamble Responsibly**: Set limits and never bet more than you can afford to lose
- **Data Freshness**: Stats are cached, so very recent games may not be reflected immediately

## Troubleshooting

### "Could not find player"
- Use the player's full official name
- Check spelling
- The player must be active in the current NBA season

### API Errors
- The NBA API may be temporarily unavailable
- Wait a few minutes and try again
- Check your internet connection

### Model Not Found
- Run `python train_model.py --train` to train models
- Or use the app without training (uses smart estimation)

## License

MIT License - Use freely for personal purposes.
