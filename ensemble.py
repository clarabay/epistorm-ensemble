import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d
from datetime import timedelta


def create_ensemble_method1(forecast_data):
    """
    Ensemble Method 1: Create ensemble forecasts from individual model predictions
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date
        - location
        - horizon
        - target
        - target_end_date
        - output_type
        - output_type_id
        - value
        - model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble forecasts in the same format as input (without 'model' column,
        as it will be added by the calling function)
    
    uses median across models for each quantile.
    """
    
    # Filter for quantile forecasts only
    quantile_data = forecast_data[(forecast_data['output_type'] == 'quantile') & (forecast_data['target']=='wk inc flu hosp')].copy() 
    
    # Group by all relevant columns except model and value
    grouping_cols = ['reference_date', 'location', 'horizon', 'target',  'target_end_date', 'output_type', 'output_type_id']
    
    # Calculate median across models for each quantile
    ensemble = quantile_data.groupby(grouping_cols, as_index=False)['value'].median()
    
    return ensemble


def create_categorical_ensemble(forecast_data):
    """
    Create ensemble categorical forecasts by averaging probabilities across models
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date, location, horizon, target, target_end_date
        - output_type, output_type_id, value, model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble categorical forecasts with normalized probabilities
    """
    
    # Filter for categorical forecasts
    df = forecast_data[
        (forecast_data['output_type'] == 'pmf') & 
        (forecast_data['target'] == 'wk flu hosp rate change')
    ].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert date columns
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    # Group by all dimensions except model and value
    group_cols = ['reference_date', 'location', 'horizon', 'target', 
                  'target_end_date', 'output_type', 'output_type_id']
    
    # Calculate mean probability across models
    ensemble = df.groupby(group_cols)['value'].mean().reset_index()
    
    # Normalize probabilities to sum to 1 for each (reference_date, location, horizon) group
    normalize_cols = ['reference_date', 'location', 'horizon']
    
    # Calculate sum of probabilities for each group
    prob_sums = ensemble.groupby(normalize_cols)['value'].transform('sum')
    
    # Normalize (handle division by zero)
    ensemble['value'] = np.where(
        prob_sums > 0,
        ensemble['value'] / prob_sums,
        0
    )
    
    # Add model identifier
    ensemble['model'] = 'Median Epistorm Ensemble'
    
    return ensemble


def create_ensemble_method2(forecast_data: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized Ensemble Method 2: Create ensemble forecasts from individual model predictions
    
    Performance improvements:
    - Vectorized operations instead of loops where possible
    - Eliminated repeated DataFrame concatenations
    - Pre-allocated arrays for results
    - Cached repeated computations
    - Used groupby operations more efficiently
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        DataFrame containing all model forecasts with columns:
        - reference_date, location, horizon, target, target_end_date
        - output_type, output_type_id, value, model
    
    Returns:
    --------
    pd.DataFrame
        Ensemble forecasts in the same format as input
    """
    
    # Filter once at the beginning
    df = forecast_data[
        (forecast_data['output_type'] == 'quantile') & 
        (forecast_data['target'] == 'wk inc flu hosp')
    ].copy()
    
    # Convert types once
    df['output_type_id'] = df['output_type_id'].astype(float)
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    # Exclude specific locations upfront
    df = df[~df['location'].isin(['66', '72'])]
    
    # Group by date and location to process in batches
    grouped = df.groupby(['reference_date', 'location'])
    
    # Collect results in a list for efficient concatenation
    results_list = []
    
    for (date, location), group_data in grouped:
        try:
            ensemble_result = process_location_date(group_data, date, location)
            if ensemble_result is not None:
                results_list.append(ensemble_result)
        except Exception as e:
            print(f'Error processing {date}, location {location}: {e}')
            continue
    
    if not results_list:
        return pd.DataFrame()
    
    # Single concatenation at the end
    return pd.concat(results_list, ignore_index=True)


def process_location_date(group_data: pd.DataFrame, date, location) -> pd.DataFrame:
    """Process ensemble for a single location and date"""
    
    # Get unique models and horizons
    models = group_data['model'].unique()
    horizons = sorted(group_data['horizon'].unique())
    
    if len(models) == 0:
        return None
    
    # Pre-compute interpolation values for all horizons
    interp_data = {}
    
    for horizon in horizons:
        horizon_data = group_data[group_data['horizon'] == horizon]
        
        # Collect all values from all models for this horizon
        all_values = horizon_data['value'].values
        
        if len(all_values) == 0:
            continue
            
        interp_data[horizon] = all_values
    
    if not interp_data:
        return None
    
    # Process each model's quantiles
    quantile_results = []
    
    for model in models:
        model_data = group_data[group_data['model'] == model]
        
        for horizon in horizons:
            if horizon not in interp_data:
                continue
                
            horizon_model_data = model_data[model_data['horizon'] == horizon]
            
            if len(horizon_model_data) == 0:
                continue
            
            # Sort by quantile for interpolation
            horizon_model_data = horizon_model_data.sort_values('output_type_id')
            quantiles = horizon_model_data['output_type_id'].values
            values = horizon_model_data['value'].values
            
            # Interpolate
            interp_vals = interp_data[horizon]
            interp_quantiles = np.interp(interp_vals, values, quantiles)
            
            # Store results
            for val, quant in zip(interp_vals, interp_quantiles):
                quantile_results.append({
                    'horizon': horizon,
                    'xvalue': val,
                    'quantile_val': quant
                })
    
    if not quantile_results:
        return None
    
    # Convert to DataFrame and compute means
    quant_df = pd.DataFrame(quantile_results)
    avg_quantiles = quant_df.groupby(['horizon', 'xvalue'])['quantile_val'].mean().reset_index()
    
    # Get target quantiles from first model
    first_model = group_data[group_data['model'] == models[0]]
    
    # Build final results
    final_results = []
    
    for horizon in horizons:
        horizon_avg = avg_quantiles[avg_quantiles['horizon'] == horizon]
        
        if len(horizon_avg) == 0:
            continue
        
        # Get target quantiles and end date for this horizon
        horizon_first = first_model[first_model['horizon'] == horizon]
        
        if len(horizon_first) == 0:
            continue
            
        target_quantiles = sorted(horizon_first['output_type_id'].unique())
        target_end_date = horizon_first['target_end_date'].iloc[0]
        
        # Sort for interpolation
        horizon_avg = horizon_avg.sort_values('quantile_val')
        source_quantiles = horizon_avg['quantile_val'].values
        source_values = horizon_avg['xvalue'].values
        
        # Interpolate back to target quantiles
        final_values = np.interp(target_quantiles, source_quantiles, source_values)
        
        # Create result rows
        for quant, val in zip(target_quantiles, final_values):
            final_results.append({
                'output_type_id': quant,
                'value': val,
                'horizon': horizon,
                'target_end_date': target_end_date,
                'location': location,
                'reference_date': date
            })
    
    if not final_results:
        return None
    
    return pd.DataFrame(final_results)


# Additional optimization: Parallel processing option
def create_ensemble_method2_parallel(forecast_data: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Parallel version using joblib for even faster processing
    Install with: pip install joblib
    
    Parameters:
    -----------
    forecast_data : pd.DataFrame
        Input forecast data
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
    """
    from joblib import Parallel, delayed
    
    # Same preprocessing
    df = forecast_data[
        (forecast_data['output_type'] == 'quantile') & 
        (forecast_data['target'] == 'wk inc flu hosp')
    ].copy()
    
    df['output_type_id'] = df['output_type_id'].astype(float)
    df['reference_date'] = pd.to_datetime(df['reference_date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df = df[~df['location'].isin(['66', '72'])]
    
    grouped = df.groupby(['reference_date', 'location'])
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_location_date)(group_data, date, location)
        for (date, location), group_data in grouped
    )
    
    # Filter out None results and concatenate
    results = [r for r in results if r is not None]
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)



def create_categorical_ensemble_quantile(df):
    obs = pd.read_csv('https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv')
    obs['date'] = pd.to_datetime(obs['date'])
    locations = pd.read_csv('locations.csv')

    TREND_MAP = {0: {  # 1-week ahead
                "stable_rate_max": 0.3,
                "stable_count_max": 10,
                "large_threshold": 1.7, },
            1: {  # 2-week ahead
                "stable_rate_max": 0.5,
                "stable_count_max": 10,
                "large_threshold": 3.0,},
            2: {  # 3-week ahead
                "stable_rate_max": 0.7,
                "stable_count_max": 10,
                "large_threshold": 4.0,},
            3: {  # 4- & 5-week ahead
                "stable_rate_max": 1.0,
                "stable_count_max": 10,
                "large_threshold": 5.0,},}
    
    results = []
    
    # Get all unique combinations
    combinations = df[['reference_date', 'horizon', 'location']].drop_duplicates()
    
    for idx, row in combinations.iterrows():
        reference_date = row['reference_date']
        horizon = row['horizon']
        loc = row['location']
        
        try:
            # Filter data for this combination
            df_subset = df[
                (df.reference_date == reference_date) & 
                (df.horizon == horizon) & 
                (df.location == loc)
            ].sort_values(by='output_type_id')
            
            if len(df_subset) == 0:
                print(f"No data for {loc}, horizon {horizon}, reference_date {reference_date}")
                continue
            
            # Get target_end_date (should be same for all quantiles in this subset)
            target_end_date = df_subset['target_end_date'].iloc[0]
            
            # Get observed value
            last_obs = pd.to_datetime(reference_date) - timedelta(days=7)
            try:
                obs_date = last_obs.strftime('%Y-%m-%d')
                obs_vers = pd.read_csv(f'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/auxiliary-data/target-data-archive/target-hospital-admissions_{obs_date}.csv')
                obs_vers['date'] = pd.to_datetime(obs_vers['date'])
                obs_subset = obs_vers[(obs_vers.location == loc) &  (obs_vers.date == last_obs)]
            except:
                obs_subset = obs[(obs.location == loc) &  (obs.date == last_obs)]

            
            if len(obs_subset) == 0:
                print(f"No observation for {loc} on {last_obs}")
                continue
                
            val = obs_subset.value.values[0]
            
            # Calculate count differences
            df_subset = df_subset.copy()
            df_subset['count_diff'] = df_subset['value'] - val
            
            quantiles = list(df_subset['output_type_id'])
            count_changes = list(df_subset['count_diff'])
            
            # Get population
            pop_subset = locations[locations.location == loc]
            if len(pop_subset) == 0:
                print(f"No population data for {loc}")
                continue
                
            population = pop_subset.population.values[0]
            rate_changes = (np.array(count_changes) / population) * 100000
            
            # Create CDFs
            cdf_count = interp1d(
                count_changes, quantiles, 
                kind='linear', bounds_error=False, fill_value=(0, 1)
            )
            cdf_rate = interp1d(
                rate_changes, quantiles,
                kind='linear', bounds_error=False, fill_value=(0, 1)
            )
            
            # Get thresholds
            trends = TREND_MAP[horizon]
            countmap = trends['stable_count_max']
            ratemap = trends['stable_rate_max']
            largemap = trends['large_threshold']
            
            # Get CDF values at boundaries
            p_count_minus10 = float(cdf_count(-countmap))
            p_count_plus10 = float(cdf_count(countmap))
            p_rate_decrease = float(cdf_rate(-ratemap))
            p_rate_increase = float(cdf_rate(ratemap))
            p_rate_largedec = float(cdf_rate(-largemap))
            p_rate_largeinc = float(cdf_rate(largemap))
            
            # Calculate rates at count boundaries
            rate_count10 = (countmap / population) * 100000
            rate_countminus10 = (-countmap / population) * 100000
            
            # Initialize probabilities with defaults (rate-based)
            probs = {}
            probs['stable'] = p_rate_increase - p_rate_decrease
            probs['increase'] = p_rate_largeinc - p_rate_increase
            probs['large_increase'] = 1 - p_rate_largeinc
            probs['decrease'] = p_rate_decrease - p_rate_largedec
            probs['large_decrease'] = p_rate_largedec
            
            # Apply logic based on which constraints are binding
            if rate_count10 < ratemap and rate_countminus10 > -ratemap:
                # Rate is wider on both sides - keep defaults
                pass
                
            elif rate_count10 < largemap and rate_count10 >= ratemap and rate_countminus10 > -ratemap:
                probs['stable'] = p_count_plus10 - p_rate_decrease
                probs['increase'] = p_rate_largeinc - p_count_plus10
                
            elif rate_count10 >= largemap and rate_countminus10 > -ratemap:
                probs['stable'] = p_count_plus10 - p_rate_decrease
                probs['increase'] = 0
                probs['large_increase'] = 1 - p_count_plus10
                
            elif rate_count10 < ratemap and rate_countminus10 > -largemap and rate_countminus10 <= -ratemap:
                probs['stable'] = p_rate_increase - p_count_minus10
                probs['decrease'] = p_count_minus10 - p_rate_largedec
                
            elif rate_count10 < ratemap and rate_countminus10 <= -largemap:
                probs['stable'] = p_rate_increase - p_count_minus10
                probs['decrease'] = 0
                probs['large_decrease'] = p_count_minus10
                
            elif rate_count10 < largemap and rate_countminus10 > -largemap and rate_countminus10 <= -ratemap and rate_count10 >= ratemap:
                probs['stable'] = p_count_plus10 - p_count_minus10
                probs['increase'] = p_rate_largeinc - p_count_plus10
                probs['decrease'] = p_count_minus10 - p_rate_largedec
                
            elif rate_count10 >= largemap and rate_countminus10 > -largemap and rate_countminus10 <= -ratemap:
                probs['stable'] = p_count_plus10 - p_count_minus10
                probs['increase'] = 0
                probs['large_increase'] = 1 - p_count_plus10
                probs['decrease'] = p_count_minus10 - p_rate_largedec
                
            elif rate_count10 < largemap and rate_countminus10 <= -largemap and rate_count10 >= ratemap:
                probs['stable'] = p_count_plus10 - p_count_minus10
                probs['decrease'] = 0
                probs['large_decrease'] = p_count_minus10
                probs['increase'] = p_rate_largeinc - p_count_plus10
                
            elif rate_count10 >= largemap and rate_countminus10 <= -largemap:
                probs['stable'] = p_count_plus10 - p_count_minus10
                probs['decrease'] = 0
                probs['increase'] = 0
                probs['large_decrease'] = p_count_minus10
                probs['large_increase'] = 1 - p_count_plus10
                
            else:
                probs['stable'] = 1
                probs['increase'] = 0
                probs['large_increase'] = 0
                probs['decrease'] = 0
                probs['large_decrease'] = 0
            
            # Verify probabilities sum to 1
            total = sum(probs.values())
            if abs(total - 1.0) > 0.01:
                print(f"WARNING: Probabilities don't sum to 1 for {loc}, horizon {horizon}, date {reference_date}: {total:.4f}")
            
            # Create output rows
            for category, probability in probs.items():
                results.append({
                    'reference_date': reference_date,
                    'target_end_date': target_end_date,
                    'horizon': horizon,
                    'location': loc,
                    'target': 'wk flu hosp rate change',
                    'output_type': 'pmf',
                    'output_type_id': category,
                    'value': probability,
                    'Model': 'Median Epistorm Ensemble'
                })
                
        except Exception as e:
            print(f"Error processing {loc}, horizon {horizon}, reference_date {reference_date}: {str(e)}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    results_df = results_df[[
        'target_end_date', 'horizon', 'output_type_id', 'value', 
        'location', 'target', 'Model', 'output_type', 'reference_date'
    ]]
    
    return results_df
