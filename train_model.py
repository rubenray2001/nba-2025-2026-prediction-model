"""
Training Script for NBA 1H Prediction Model
Collects data and trains the ensemble model
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from data_collector import (
    get_all_active_players,
    get_player_multi_season_logs,
    collect_training_data,
    ensure_dirs
)
from feature_engineering import prepare_training_data
from model import NBA1HEnsembleModel
from config import SEASONS_TO_FETCH, DATA_DIR


def collect_player_data(num_players: int = 100, seasons: list = None):
    """
    Collect training data for specified number of players.
    
    Args:
        num_players: Number of players to collect data for
        seasons: List of seasons to fetch
    """
    if seasons is None:
        seasons = SEASONS_TO_FETCH
    
    ensure_dirs()
    
    print(f"\n{'='*50}")
    print(f"Collecting data for {num_players} players")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"{'='*50}\n")
    
    # Get active players
    print("Fetching active players list...")
    players_df = get_all_active_players()
    
    if players_df.empty:
        print("Error: Could not fetch active players")
        return None
    
    print(f"Found {len(players_df)} active players")
    
    # Limit to requested number
    player_ids = players_df['PERSON_ID'].tolist()[:num_players]
    
    # Collect data
    all_data = []
    
    for i, player_id in enumerate(tqdm(player_ids, desc="Collecting player data")):
        try:
            logs = get_player_multi_season_logs(player_id, seasons)
            if not logs.empty:
                logs['PLAYER_ID'] = player_id
                all_data.append(logs)
        except Exception as e:
            print(f"\nError collecting data for player {player_id}: {e}")
            continue
    
    if not all_data:
        print("Error: No data collected")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(DATA_DIR, f"training_data_{timestamp}.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Data collection complete!")
    print(f"Total games collected: {len(combined_df)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*50}\n")
    
    return combined_df


def train_models(data_path: str = None, target: str = 'both'):
    """
    Train the ensemble models.
    
    Args:
        data_path: Path to training data CSV (if None, uses latest)
        target: 'points', 'pra', or 'both'
    """
    ensure_dirs()
    
    # Find data file
    if data_path is None:
        # Look for latest training data
        data_files = [f for f in os.listdir(DATA_DIR) if f.startswith('training_data_') and f.endswith('.csv')]
        if not data_files:
            print("No training data found. Run with --collect first.")
            return
        
        data_path = os.path.join(DATA_DIR, sorted(data_files)[-1])
    
    print(f"\n{'='*50}")
    print(f"Loading training data from: {data_path}")
    print(f"{'='*50}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} game records")
    
    # Prepare features
    print("\nPreparing features...")
    df_processed, feature_cols = prepare_training_data(df)
    print(f"Generated {len(feature_cols)} features")
    print(f"Valid training samples: {len(df_processed)}")
    
    # Train models
    targets_to_train = ['points', 'pra'] if target == 'both' else [target]
    
    for target_type in targets_to_train:
        print(f"\n{'='*50}")
        print(f"Training {target_type.upper()} model")
        print(f"{'='*50}\n")
        
        model = NBA1HEnsembleModel(target=target_type)
        
        try:
            metrics = model.train(df_processed.copy(), feature_cols)
            
            print("\nTraining Results:")
            print(f"  Samples: {metrics['n_samples']}")
            
            for model_name, model_metrics in metrics['models'].items():
                print(f"\n  {model_name.upper()}:")
                print(f"    CV MAE: {model_metrics['cv_mae']:.2f} (+/- {model_metrics['cv_std']:.2f})")
                print(f"    Train MAE: {model_metrics['train_mae']:.2f}")
                print(f"    Train R2: {model_metrics['train_r2']:.3f}")
            
            # Show top features
            importance = model.get_feature_importance()
            print(f"\n  Top 10 Features:")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                print(f"    {i+1}. {feature}: {imp:.4f}")
            
            print(f"\n[OK] {target_type.upper()} model saved successfully")
            
        except Exception as e:
            print(f"Error training {target_type} model: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='NBA 1H Prediction Model Training')
    
    parser.add_argument(
        '--collect', 
        action='store_true',
        help='Collect training data'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train the models'
    )
    
    parser.add_argument(
        '--players', 
        type=int, 
        default=100,
        help='Number of players to collect data for (default: 100)'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '--target', 
        type=str, 
        default='both',
        choices=['points', 'pra', 'both'],
        help='Model target: points, pra, or both (default: both)'
    )
    
    args = parser.parse_args()
    
    if not args.collect and not args.train:
        print("Please specify --collect or --train (or both)")
        parser.print_help()
        return
    
    if args.collect:
        df = collect_player_data(num_players=args.players)
        if df is not None and args.train:
            # Use the freshly collected data for training
            args.data = None  # Will use latest file
    
    if args.train:
        train_models(data_path=args.data, target=args.target)


if __name__ == "__main__":
    main()
