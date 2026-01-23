"""
Script to fetch and cache forecast data
Run by GitHub Actions weekly
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from pathlib import Path
from epiweeks import Week

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS = [
   'MIGHTE-Nsemble', 
    'MIGHTE-Joint', 
    'CEPH-Rtrend_fluH', 
    'MOBS-EpyStrain_Flu', 
    'MOBS-GLEAM_RL_FLUH', 
    'NU-PGF_FLUH', 
    'NEU_ISI-FluBcast', 
    'NEU_ISI-AdaptiveEnsemble',
    'Gatech-ensemble_prob',
    'Gatech-ensemble_stat'
]

def fetch_observed_data():
    """Fetch observed hospital admissions data"""
    print("Fetching observed data...")
    url = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
    try:
        response = requests.get(url)
        data = pd.read_csv(StringIO(response.text))
        output_file = DATA_DIR / "observed_data.csv"
        data.to_csv(output_file, index=False)
        print(f"✓ Saved observed data to {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error fetching observed data: {e}")
        return False


def fetch_model_forecasts(model_name):
    """Fetch forecasts for a specific model"""
    base_url = f"https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/{model_name}"
    all_forecasts = []
    
    start_date = datetime(2025, 11, 1)
    end_date = pd.to_datetime(Week.fromdate(datetime.now()).enddate())
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_url = f"{base_url}/{date_str}-{model_name}.csv"
        
        try:
            response = requests.get(file_url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df['model'] = model_name
                
                # FIXED: Always convert location to string and zero-pad if numeric
                if 'location' in df.columns:
                    # Convert to string first
                    df['location'] = df['location'].astype(str)
                    # Zero-pad if it looks like a numeric location code
                    if df['location'].str.match(r'^\d+$').any():
                        df['location'] = df['location'].str.zfill(2)
                
                all_forecasts.append(df)
                print(f"  ✓ Fetched {date_str}")
            else:
                print(f"  - Skipped {date_str} (not found)")
        except Exception as e:
            print(f"  ✗ Error fetching {date_str}: {e}")
        
        current_date += pd.Timedelta(days=7)
    
    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    return pd.DataFrame()

def fetch_all_forecasts():
    """Fetch all model forecasts"""
    print("Fetching all model forecasts...")
    all_data = []
    
    for idx, model in enumerate(MODELS):
        print(f"[{idx+1}/{len(MODELS)}] Fetching {model}...")
        df = fetch_model_forecasts(model)
        if not df.empty:
            all_data.append(df)
            print(f"  ✓ Got {len(df)} rows")
        else:
            print(f"  ✗ No data found")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = DATA_DIR / "all_forecasts.parquet"
        combined.to_parquet(output_file, index=False)
        print(f"✓ Saved all forecasts to {output_file} ({len(combined)} rows)")
        return True
    else:
        print("✗ No forecast data collected")
        return False

def fetch_baseline_forecasts():
    """Fetch baseline model forecasts"""
    print("Fetching baseline forecasts...")
    baseline_model = 'FluSight-baseline'
    df = fetch_model_forecasts(baseline_model)
    
    if not df.empty:
        output_file = DATA_DIR / "baseline_forecasts.parquet"
        df.to_parquet(output_file, index=False)
        print(f"✓ Saved baseline forecasts to {output_file} ({len(df)} rows)")
        return True
    else:
        print("✗ No baseline data found")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Starting data fetch process...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    success_count = 0
    
    if fetch_observed_data():
        success_count += 1
    
    if fetch_all_forecasts():
        success_count += 1
    
    if fetch_baseline_forecasts():
        success_count += 1
    
    print("=" * 60)
    print(f"Completed: {success_count}/3 tasks successful")
    print("=" * 60)