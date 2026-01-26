"""
Script to create ensemble forecasts (both quantile and categorical).
Runs as part of GitHub Actions workflow.
"""

import pandas as pd
import numpy as np
from ensemble import create_ensemble_method1, create_categorical_ensemble_quantile
from pathlib import Path
import sys

# Define threshold map for categorical forecasts
TREND_MAP = {
    0: {'stable_count_max': 10, 'stable_rate_max': 0.3, 'large_threshold': 1.7},
    1: {'stable_count_max': 10, 'stable_rate_max': 0.5, 'large_threshold': 2.0},
    2: {'stable_count_max': 10, 'stable_rate_max': 0.7, 'large_threshold': 2.5},
    3: {'stable_count_max': 10, 'stable_rate_max': 0.9, 'large_threshold': 3.0},
}

def main():
    try:
        print("=" * 60)
        print("Creating Ensemble Forecasts")
        print("=" * 60)
        
        # Load forecast data
        print("\n1. Loading forecast data...")
        forecast_path = Path('data/all_forecasts.parquet')  # Adjust to your actual path
        
        if not forecast_path.exists():
            print(f"ERROR: Forecast file not found at {forecast_path}")
            sys.exit(1)
            
        df = pd.read_parquet(forecast_path)
        print(f"   Loaded {len(df):,} forecast rows")
        print(f"   Models: {df['model'].nunique()}")
        print(f"   Reference dates: {df['reference_date'].nunique()}")
        print(f"   Locations: {df['location'].nunique()}")
        print(f"   Horizons: {sorted(df['horizon'].unique())}")
        
        # =====================================================================
        # PART 1: Create Quantile Ensemble (Method 1)
        # =====================================================================
        print("\n" + "=" * 60)
        print("PART 1: Creating Quantile Ensemble (Median Method)")
        print("=" * 60)
        
        quantile_ensemble = create_ensemble_method1(df)
        
        if len(quantile_ensemble) == 0:
            print("ERROR: No quantile ensemble forecasts generated!")
            sys.exit(1)
        
        # Add model identifier
        quantile_ensemble['model'] = 'Median_Ensemble'
        
        print(f"   ✓ Generated {len(quantile_ensemble):,} quantile forecast rows")
        print(f"   Reference dates: {quantile_ensemble['reference_date'].nunique()}")
        print(f"   Locations: {quantile_ensemble['location'].nunique()}")
        print(f"   Quantiles: {sorted(quantile_ensemble['output_type_id'].unique())}")
        
        # Save quantile ensemble
        print("\n   Saving quantile ensemble...")
        quantile_output_path = Path('data/quantile_ensemble.pq')
        quantile_output_path.parent.mkdir(parents=True, exist_ok=True)
        quantile_ensemble.to_parquet(quantile_output_path, index=False)
        print(f"   ✓ Saved to {quantile_output_path}")
        
        # Also save as CSV
        quantile_csv_path = Path('data/quantile_ensemble.csv')
        quantile_ensemble.to_csv(quantile_csv_path, index=False)
        print(f"   ✓ Also saved as {quantile_csv_path}")
        
        # =====================================================================
        # PART 2: Create Categorical Ensemble
        # =====================================================================
        print("\n" + "=" * 60)
        print("PART 2: Creating Categorical Ensemble (Trend Predictions)")
        print("=" * 60)
        
        # Load observations
        print("\n   Loading observation data...")
        obs_url = 'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv'
        obs = pd.read_csv(obs_url)
        obs['date'] = pd.to_datetime(obs['date'])
        print(f"   Loaded {len(obs):,} observation rows")
        
        # Load locations
        print("\n   Loading location data...")
        locations_path = Path('locations.csv')  # Adjust if needed
        
        if not locations_path.exists():
            print(f"ERROR: Locations file not found at {locations_path}")
            sys.exit(1)
            
        locations = pd.read_csv(locations_path)
        print(f"   Loaded {len(locations)} locations")
        
        # Use the quantile ensemble to create categorical forecasts
        print("\n   Creating categorical ensemble from quantile ensemble...")
        print("   This may take a few minutes...")
        
        categorical_ensemble = create_categorical_ensemble_quantile(
            quantile_ensemble)
        
        if len(categorical_ensemble) == 0:
            print("WARNING: No categorical forecasts generated!")
            sys.exit(1)
        
        print(f"   ✓ Generated {len(categorical_ensemble):,} categorical forecast rows")
        
        # Validate categorical output
        print("\n   Validating categorical output...")
        required_cols = [
            'target_end_date', 'horizon', 'output_type_id', 'value', 
            'location', 'target', 'Model', 'output_type', 'reference_date'
        ]
        missing_cols = [col for col in required_cols if col not in categorical_ensemble.columns]
        
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Check for valid probabilities
        prob_check = categorical_ensemble.groupby(
            ['reference_date', 'horizon', 'location']
        )['value'].sum()
        
        invalid_probs = prob_check[(prob_check < 0.99) | (prob_check > 1.01)]
        if len(invalid_probs) > 0:
            print(f"   WARNING: {len(invalid_probs)} groups have probabilities not summing to 1")
            print(invalid_probs.head())
        else:
            print("   ✓ All probability distributions sum to 1")
        
        # Save categorical results
        print("\n   Saving categorical ensemble...")
        categorical_output_path = Path('data/categorical_ensemble.pq')
        categorical_ensemble.to_parquet(categorical_output_path, index=False)
        print(f"   ✓ Saved to {categorical_output_path}")
        
        # Also save as CSV
        categorical_csv_path = Path('data/categorical_ensemble.csv')
        categorical_ensemble.to_csv(categorical_csv_path, index=False)
        print(f"   ✓ Also saved as {categorical_csv_path}")
        
        # =====================================================================
        # PART 3: Combine Everything
        # =====================================================================
        print("\n" + "=" * 60)
        print("PART 3: Combining All Ensemble Forecasts")
        print("=" * 60)
        
        # Combine quantile and categorical ensembles
        # Make sure column names match
        if 'model' in quantile_ensemble.columns and 'Model' in categorical_ensemble.columns:
            categorical_ensemble = categorical_ensemble.rename(columns={'Model': 'model'})
        elif 'Model' in quantile_ensemble.columns and 'model' in categorical_ensemble.columns:
            quantile_ensemble = quantile_ensemble.rename(columns={'model': 'Model'})
        
        combined_ensemble = pd.concat([quantile_ensemble, categorical_ensemble], ignore_index=True)
        
        print(f"   ✓ Combined ensemble has {len(combined_ensemble):,} total rows")
        print(f"      - Quantile forecasts: {len(quantile_ensemble):,}")
        print(f"      - Categorical forecasts: {len(categorical_ensemble):,}")
        
        # Save combined
        combined_path = Path('data/ensemble_forecasts.pq')
        combined_ensemble.to_parquet(combined_path, index=False)
        print(f"   ✓ Saved combined ensemble to {combined_path}")
        
        combined_csv_path = Path('data/ensemble_forecasts.csv')
        combined_ensemble.to_csv(combined_csv_path, index=False)
        print(f"   ✓ Also saved as {combined_csv_path}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        print("\nQuantile Ensemble by Reference Date:")
        quantile_summary = quantile_ensemble.groupby('reference_date').agg({
            'location': 'nunique',
            'horizon': 'nunique',
            'output_type_id': 'nunique'
        })
        quantile_summary.columns = ['Locations', 'Horizons', 'Quantiles']
        print(quantile_summary)
        
        print("\nCategorical Ensemble by Reference Date:")
        categorical_summary = categorical_ensemble.groupby('reference_date').agg({
            'location': 'nunique',
            'horizon': 'nunique',
            'output_type_id': 'nunique'
        })
        categorical_summary.columns = ['Locations', 'Horizons', 'Categories']
        print(categorical_summary)
        
        print("\n✓ All ensemble forecasts created successfully!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()