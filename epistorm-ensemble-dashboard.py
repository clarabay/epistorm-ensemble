import streamlit as st
import pandas as pd
from epiweeks import Week
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np
import os
from pathlib import Path
from ensemble import create_ensemble_method1, create_ensemble_method2, create_categorical_ensemble

# Page config
#st.set_page_config(page_title="Epistorm Influenza Forecasts", layout="wide")


import plotly.io as pio

# Set Plotly to always use light theme
pio.templates.default = "plotly_white"

# Force light mode
st.set_page_config(
    page_title="Epistorm Ensemble Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Constants
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

QUANTILES = ['0.01', '0.025', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', 
             '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', 
             '0.85', '0.9', '0.95', '0.975', '0.99']

COLOR_MAP = {
    'Median Epistorm Ensemble': '#1f77b4',
    'MIGHTE-Nsemble': '#2ca02c',
    'MIGHTE-Joint': '#d62728',
    'CEPH-Rtrend_fluH': '#9467bd',
    'MOBS-EpyStrain_Flu': '#8c564b',
    'MOBS-GLEAM_RL_FLUH': '#e377c2',
    'NU-PGF_FLUH': '#7f7f7f',
    'NEU_ISI-FluBcast': '#bcbd22',
    'NEU_ISI-AdaptiveEnsemble': '#17becf',
    'Gatech-ensemble_prob': '#ff7f0e',
    'Gatech-ensemble_stat': '#aec7e8'
}

CATEGORY_COLORS = {
    'Large Decrease': '#006d77',
    'Decrease': '#83c5be',
    'Stable': '#e5e5e5',
    'Increase': '#e29578',
    'Large Increase': '#bc4749'
}

CATEGORY_ORDER = ['Large Decrease', 'Decrease', 'Stable', 'Increase', 'Large Increase']

INTERVAL_RANGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=3600)
def load_locations():
    """Load locations from local file or fallback to original source"""
    try:
        locations_df = pd.read_csv('locations.csv')
        return locations_df
    except Exception as e:
        st.error(f"Error loading locations: {e}")
        return None

@st.cache_data(ttl=3600)
def load_observed_data():
    """Load observed data from local cache or fetch if not available"""
    local_file = DATA_DIR / "observed_data.csv"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            data = pd.read_csv(local_file)
            data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            st.warning(f"Error loading cached data: {e}. Fetching fresh data...")
    
    # Fallback to fetching from URL
    url = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
    try:
        response = requests.get(url)
        data = pd.read_csv(StringIO(response.text))
        data['date'] = pd.to_datetime(data['date'])
        return data
    except Exception as e:
        st.error(f"Error loading observed data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_all_forecasts():
    """Load all forecasts from local cache or fetch if not available"""
    local_file = DATA_DIR / "all_forecasts.parquet"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            combined = pd.read_parquet(local_file)
            combined['reference_date'] = pd.to_datetime(combined['reference_date'])
            combined['target_end_date'] = pd.to_datetime(combined['target_end_date'])
            st.info(f"Loaded data (last updated: {pd.Timestamp(local_file.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})")
            return combined
        except Exception as e:
            st.warning(f"Error loading cached forecasts: {e}. Fetching fresh data...")
    
    # Fallback to fetching from URLs
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model in enumerate(MODELS):
        status_text.text(f"Loading {model}...")
        df = load_model_forecasts(model)
        if not df.empty:
            all_data.append(df)
        progress_bar.progress((idx + 1) / len(MODELS))
    
    status_text.empty()
    progress_bar.empty()
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['reference_date'] = pd.to_datetime(combined['reference_date'])
        combined['target_end_date'] = pd.to_datetime(combined['target_end_date'])
        return combined
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_baseline_forecasts():
    """Load baseline forecasts from local cache or fetch if not available"""
    local_file = DATA_DIR / "baseline_forecasts.parquet"
    
    # Try to load from local file first
    if local_file.exists():
        try:
            df = pd.read_parquet(local_file)
            df['reference_date'] = pd.to_datetime(df['reference_date'])
            df['target_end_date'] = pd.to_datetime(df['target_end_date'])
            return df
        except Exception as e:
            st.warning(f"Error loading cached baseline: {e}. Fetching fresh data...")
    
    # Fallback to fetching
    baseline_model = 'FluSight-baseline'
    df = load_model_forecasts(baseline_model)
    if not df.empty:
        df['reference_date'] = pd.to_datetime(df['reference_date'])
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    return df



@st.cache_data(ttl=3600)
def load_model_forecasts(model_name):
    base_url = f"https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output/{model_name}"
    all_forecasts = []
    start_date = datetime(2025, 11, 1)
    #end_date = datetime.now()
    end_date = pd.to_datetime(Week.fromdate(datetime.now()).enddate())
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        file_url = f"{base_url}/{date_str}-{model_name}.csv"
        try:
            response = requests.get(file_url, timeout=5)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df['model'] = model_name
                if model_name in ['Gatech-ensemble_prob', 'Gatech-ensemble_stat']:
                    if df['location'].dtype in ['int64', 'int32']:
                        df['location'] = df['location'].astype(int).astype(str).str.zfill(2)
                all_forecasts.append(df)
        except Exception as e:
            pass
        current_date += pd.Timedelta(days=7)
    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_wis_data():
    try:
        df = pd.read_parquet('./wis_ratio_epistorm_models_2526.pq')
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
        df['reference_date'] = df['target_end_date'] - pd.to_timedelta(df['horizon'] * 7, unit='D')
        df['location'] = df['location'].astype(str).str.zfill(2)
        return df
    except Exception as e:
        st.error(f"Error loading WIS data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_coverage_data():
    try:
        df = pd.read_parquet('./coverage_epistorm_models_2526.pq')
        df['target_end_date'] = pd.to_datetime(df['target_end_date'])
        df['reference_date'] = df['target_end_date'] - pd.to_timedelta(df['horizon'] * 7, unit='D')
        df['location'] = df['location'].astype(str).str.zfill(2)
        return df
    except Exception as e:
        st.error(f"Error loading coverage data: {e}")
        return None

def create_ensemble_forecasts(forecast_data):
    ensemble1 = create_ensemble_method1(forecast_data)
    ensemble1['model'] = 'Median Epistorm Ensemble'
    categorical_ensemble = create_categorical_ensemble(forecast_data)
    return pd.concat([ensemble1, categorical_ensemble], ignore_index=True)

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def format_category(category):
    category_str = str(category).strip().lower()
    word_map = {
        'large_decrease': 'Large Decrease',
        'large decrease': 'Large Decrease',
        'decrease': 'Decrease',
        'stable': 'Stable',
        'increase': 'Increase',
        'large_increase': 'Large Increase',
        'large increase': 'Large Increase'
    }
    return word_map.get(category_str, category_str.title())

def get_location_name(location_code, locations_df):
    if location_code == 'US':
        return 'United States'
    if locations_df is not None:
        loc_info = locations_df[locations_df['location'] == location_code]
        if not loc_info.empty:
            return loc_info.iloc[0]['location_name']
    return location_code

def plot_wis_boxplot_evaluation(wis_df, selected_models, locations_df):
    if wis_df is None or wis_df.empty:
        return None
    fig = go.Figure()
    models_with_data = [m for m in selected_models if m in wis_df['Model'].unique()]
    
    # Calculate median WIS ratio for each model and sort
    model_medians = {}
    for model in models_with_data:
        if model=='FluSight-ensemble':
            continue
        model_data = wis_df[wis_df['Model'] == model]
        if not model_data.empty:
            model_medians[model] = model_data['wis_ratio'].median()
    
    # Sort models by median WIS ratio (ascending - best performers first)
    sorted_models = sorted(model_medians.keys(), key=lambda m: model_medians[m])
    
    for model in sorted_models:
        model_data = wis_df[wis_df['Model'] == model]
        if not model_data.empty:
            fig.add_trace(go.Box(
                x=model_data['wis_ratio'],
                y=[model] * len(model_data),
                name=model,
                marker_color=COLOR_MAP.get(model, '#808080'),
                boxmean=False,
                orientation='h',
                hoverinfo='skip',
                boxpoints='outliers',
                marker=dict(opacity=0)  
            ))
    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="WIS ratio = 1",
        annotation_position="top"
    )

    # Calculate x-axis limits based on whiskers (1.5*IQR)
    all_whisker_mins = []
    all_whisker_maxs = []
    for model in sorted_models:
        model_data = wis_df[wis_df['Model'] == model]['wis_ratio']
        q1 = model_data.quantile(0.25)
        q3 = model_data.quantile(0.75)
        iqr = q3 - q1
        whisker_min = max(model_data.min(), q1 - 1.5 * iqr)
        whisker_max = min(model_data.max(), q3 + 1.5 * iqr)
        all_whisker_mins.append(whisker_min)
        all_whisker_maxs.append(whisker_max)

    x_min = min(all_whisker_mins) - 0.1
    x_max = max(all_whisker_maxs) + 0.1

    fig.update_layout(
        title="",
        xaxis_title="WIS Ratio (Model / FluSight Baseline)",
        yaxis_title="",
        height=max(60 * len(sorted_models) + 150, 400),
        width=1100,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50),
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(
            categoryorder='array',
            categoryarray=list(reversed(sorted_models))
        )
    )
    return fig

def plot_coverage_evaluation(coverage_df, selected_models):
    if coverage_df is None or coverage_df.empty:
        return None
    fig = go.Figure()
    intervals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
    fig.add_trace(go.Scatter(
        x=intervals,
        y=[i/100 for i in intervals],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='black', dash='dash', width=2),
        showlegend=True
    ))
    models_with_data = [m for m in selected_models if m in coverage_df['Model'].unique()]
    for model in models_with_data:
        if model=='FluSight-ensemble':
            continue
        model_data = coverage_df[coverage_df['Model'] == model]
        if not model_data.empty:
            avg_coverage = []
            for interval in intervals:
                col_name = f'{interval}_cov'
                if col_name in model_data.columns:
                    avg_coverage.append(model_data[col_name].mean())
                else:
                    avg_coverage.append(np.nan)
            fig.add_trace(go.Scatter(
                x=intervals,
                y=avg_coverage,
                mode='lines+markers',
                name=model,
                line=dict(color=COLOR_MAP.get(model, '#808080'), width=2),
                marker=dict(size=8),
                hovertemplate=f'{model}<br>Interval: %{{x}}%<br>Coverage: %{{y:.1%}}<extra></extra>'
            ))
    fig.update_layout(
        title="",
        xaxis_title="Prediction Interval (%)",
        yaxis_title="Coverage",
        height=750,
        width=1000,
        yaxis=dict(tickformat='.0%', range=[0, 1.05]),
        xaxis=dict(range=[0, 105], tickvals=intervals),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=60, r=10, t=50, b=60)
    )
    return fig

def plot_categorical_forecasts(forecast_data, selected_location, selected_date, selected_models):
    cat_data = forecast_data[
        (forecast_data['output_type'] == 'pmf') &
        (forecast_data['target'] == 'wk flu hosp rate change') &
        (forecast_data['location'] == selected_location) &
        (forecast_data['reference_date'] == selected_date) &
        (forecast_data['model'].isin(selected_models))
    ].copy()
    if cat_data.empty:
        return None, []
    cat_data['category'] = cat_data['output_type_id'].apply(format_category)
    cat_data = cat_data[cat_data['category'].isin(CATEGORY_ORDER)]
    available_models = []
    models_without_data = []
    for model in selected_models:
        model_data = cat_data[cat_data['model'] == model]
        if not model_data.empty:
            available_models.append(model)
        else:
            models_without_data.append(model)
    if not available_models:
        return None, models_without_data
    n_models = len(available_models)
    fig = make_subplots(
        rows=n_models,
        cols=1,
        subplot_titles=available_models,
        vertical_spacing=0.2 / n_models if n_models > 1 else 0.1,
        row_heights=[1] * n_models
    )
    legend_added = set()
    for idx, model in enumerate(available_models):
        model_data = cat_data[cat_data['model'] == model].copy()
        horizons = [0, 1, 2, 3]
        for horizon in horizons:
            horizon_data = model_data[model_data['horizon'] == horizon]
            cumulative = 0
            for category in CATEGORY_ORDER:
                cat_rows = horizon_data[horizon_data['category'] == category]
                if not cat_rows.empty:
                    prob = cat_rows['value'].values[0]
                else:
                    prob = 0
                show_in_legend = category not in legend_added
                if show_in_legend:
                    legend_added.add(category)
                fig.add_trace(
                    go.Bar(
                        x=[horizon],
                        y=[prob],
                        name=category,
                        marker_color=CATEGORY_COLORS[category],
                        legendgroup=category,
                        showlegend=show_in_legend,
                        hovertemplate=f'{category}: {prob:.1%}<extra></extra>' if prob > 0 else None,
                        base=cumulative,
                        width=0.6
                    ),
                    row=idx + 1,
                    col=1
                )
                cumulative += prob
        fig.update_xaxes(
            title_text="Horizon" if idx == n_models - 1 else "",
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['0', '1', '2', '3'],
            range=[-0.5, 3.5],
            row=idx + 1,
            col=1
        )
        fig.update_yaxes(
            title_text="Probability",
            tickformat='.0%',
            range=[0, 1.0],
            row=idx + 1,
            col=1
        )
    fig.update_layout(
        barmode='stack',
        height=max(300 * n_models, 400),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        margin=dict(t=120, b=50, l=80, r=50)
    )
    return fig, models_without_data

def plot_forecasts(observed_data, forecast_data, selected_location, selected_date, selected_models, available_dates, start_date, end_date):
    fig = go.Figure()
    obs_filtered = observed_data[
        (observed_data['location'] == selected_location) &
        (observed_data['date'] >= start_date) &
        (observed_data['date'] <= end_date)
    ].copy()
    obs_filtered = obs_filtered.sort_values('date')
    if not obs_filtered.empty:
        fig.add_trace(go.Scatter(
            x=obs_filtered['date'],
            y=obs_filtered['value'],
            mode='lines+markers',
            name='Observed',
            line=dict(color='black', width=2),
            marker=dict(size=4),
            hovertemplate='Value: %{y:,.0f}<extra></extra>'
        ))
    forecast_filtered = forecast_data[
        (forecast_data['location'] == selected_location) &
        (forecast_data['reference_date'] == selected_date) &
        (forecast_data['model'].isin(selected_models))
    ].copy()
    max_forecast_date = end_date
    for model in selected_models:
        model_data = forecast_filtered[forecast_filtered['model'] == model]
        if model_data.empty:
            continue
        color = COLOR_MAP.get(model, '#808080')
        quantiles_df = model_data[model_data['output_type'] == 'quantile'].copy()
        quantiles_df['output_type_id'] = quantiles_df['output_type_id'].astype(float)
        pivot_df = quantiles_df.pivot_table(
            index='target_end_date',
            columns='output_type_id',
            values='value'
        ).reset_index()
        pivot_df = pivot_df.sort_values('target_end_date')
        if not pivot_df.empty:
            max_forecast_date = max(max_forecast_date, pivot_df['target_end_date'].max())
        if 0.025 in pivot_df.columns and 0.975 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.975],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=model
            ))
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.025],
                mode='lines',
                line=dict(width=0),
                fillcolor=hex_to_rgba(color, 0.2),
                fill='tonexty',
                name=f'{model} (95% PI)',
                showlegend=True,
                customdata=pivot_df[[0.025, 0.975]],
                hovertemplate='95% PI: (%{customdata[0]:,.0f} - %{customdata[1]:,.0f}<extra></extra>)',
                legendgroup=model
            ))
        if 0.25 in pivot_df.columns and 0.75 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.75],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=model
            ))
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.25],
                mode='lines',
                line=dict(width=0),
                fillcolor=hex_to_rgba(color, 0.4),
                fill='tonexty',
                name=f'{model} (50% PI)',
                showlegend=True,
                customdata=pivot_df[[0.25, 0.75]],
                hovertemplate='50% PI: (%{customdata[0]:,.0f} - %{customdata[1]:,.0f}<extra></extra>)',
                legendgroup=model
            ))
        if 0.5 in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['target_end_date'],
                y=pivot_df[0.5],
                mode='lines+markers',
                name=f'{model} (median)',
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=pivot_df[[0.5]],
                hovertemplate='Median: %{customdata[0]:,.0f}<extra></extra>',
                legendgroup=model
            ))
    for ref_date in available_dates:
        fig.add_shape(
            type="line",
            x0=ref_date, x1=ref_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.3
        )
    fig.add_shape(
        type="line",
        x0=selected_date, x1=selected_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color='red', width=2, dash='dash')
    )
    fig.add_annotation(
        x=selected_date,
        y=1,
        yref="paper",
        text=f"Forecast Date: {selected_date.strftime('%Y-%m-%d')}",
        showarrow=False,
        yshift=10
    )
    location_name = "US" if selected_location == "US" else selected_location
    if selected_location != "US":
        locations_df = load_locations()
        if locations_df is not None:
            loc_info = locations_df[locations_df['location'] == selected_location]
            if not loc_info.empty:
                location_name = loc_info.iloc[0]['location_name']
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Weekly Incident Flu Hospitalizations",
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(
            range=[start_date, max_forecast_date]
        )
    )
    return fig, location_name

# Main app
st.markdown("<h1 style='color: #415584;'>Epistorm Influenza Forecasts 2025-26</h1>", unsafe_allow_html=True)
st.markdown('<p style="font-size: 20px; color: black;">Interactive visualization of influenza hospitalization forecasts from models in the Epistorm consortium for the 2025-26 season.</p>',
    unsafe_allow_html=True)

# View selection in sidebar
tab_selection = st.sidebar.selectbox(
    "Select View",
    ["Forecasts", "Evaluation"],
    key="tab_selection"
)

st.sidebar.markdown("---")

# Load common data
with st.spinner("Loading data..."):
    locations_df = load_locations()
    observed_data = load_observed_data()
    forecast_data = load_all_forecasts()
    baseline_data = load_baseline_forecasts()

if forecast_data.empty:
    st.error("No forecast data could be loaded. Please check your internet connection and try again.")
    st.stop()

if observed_data is None or observed_data.empty:
    st.warning("Could not load observed data. Continuing with forecasts only.")
    observed_data = pd.DataFrame(columns=['date', 'location', 'value'])

with st.spinner("Creating ensemble forecasts..."):
    try:
        if 'Median Epistorm Ensemble' not in forecast_data['model'].values:
            ensemble_data = create_ensemble_forecasts(forecast_data)
            forecast_data = pd.concat([forecast_data, ensemble_data], ignore_index=True)
        forecast_data = forecast_data[forecast_data['horizon'] >= 0]
    except Exception as e:
        st.warning(f"Could not create ensemble forecasts: {e}")
        import traceback
        st.error(traceback.format_exc())

# ============== SIDEBAR CONTROLS ==============
if tab_selection == "Forecasts":
    st.sidebar.header("Forecast Controls")
    
    if locations_df is not None:
        state_locations_df = locations_df[locations_df['location'] != 'US']
        location_options = ['US'] + state_locations_df['location'].tolist()
        location_names = ['United States'] + state_locations_df['location_name'].tolist()
        location_dict = dict(zip(location_names, location_options))
        selected_location_name = st.sidebar.selectbox(
            "Select Location",
            location_names,
            index=0,
            key="forecast_location"
        )
        selected_location = location_dict[selected_location_name]
    else:
        selected_location = st.sidebar.text_input("Enter Location Code", value="US", key="forecast_location_text")

    available_dates = sorted([pd.Timestamp(d).to_pydatetime() for d in forecast_data['reference_date'].unique()], reverse=True)

    if available_dates:
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = available_dates[0]
        selected_date = st.sidebar.selectbox(
            "Select Forecast Date",
            available_dates,
            format_func=lambda x: x.strftime('%Y-%m-%d'),
            index=available_dates.index(st.session_state.selected_date) if st.session_state.selected_date in available_dates else 0,
            key="forecast_date"
        )
        st.session_state.selected_date = selected_date
    else:
        st.error("No forecast dates available")
        st.stop()

    st.sidebar.markdown("### Historical Data Range")

    if not observed_data.empty:
        min_obs_date = observed_data['date'].min().date()
        max_obs_date = observed_data['date'].max().date()
    else:
        min_obs_date = datetime(2024, 1, 1).date()
        max_obs_date = datetime.now().date()

    earliest_forecast_date = None
    latest_forecast_date = None
    if not forecast_data.empty and 'target_end_date' in forecast_data.columns:
        selected_forecast_data = forecast_data[forecast_data['reference_date'] == selected_date]
        if not selected_forecast_data.empty:
            earliest_forecast_date = selected_forecast_data['target_end_date'].min()
            latest_forecast_date = selected_forecast_data['target_end_date'].max()
            earliest_forecast_date = pd.Timestamp(earliest_forecast_date).date()
            latest_forecast_date = pd.Timestamp(latest_forecast_date).date()

    min_end_date = earliest_forecast_date if earliest_forecast_date else min_obs_date
    if latest_forecast_date:
        max_end_date = max(max_obs_date, latest_forecast_date)
    else:
        max_end_date = max_obs_date

    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime(2025, 10, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_end_date

    if st.session_state.end_date < min_end_date:
        st.session_state.end_date = min_end_date
    if st.session_state.end_date > max_end_date:
        st.session_state.end_date = max_end_date

    start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.start_date,
        min_value=min_obs_date,
        max_value=max_end_date,
        key="forecast_start_date"
    )
    st.session_state.start_date = start_date

    end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state.end_date,
        min_value=min_end_date,
        max_value=max_end_date,
        key="forecast_end_date"
    )
    st.session_state.end_date = end_date

    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)

    all_models = sorted(forecast_data['model'].unique())
    ensemble_models = [m for m in all_models if m=='Median Epistorm Ensemble']
    individual_models = [m for m in all_models if not m=='Median Epistorm Ensemble']

    st.sidebar.markdown("### Select Models to Display")

    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['Median Epistorm Ensemble']

    selected_models = []

    if ensemble_models:
        st.sidebar.markdown("**Ensemble Models:**")
        for model in ensemble_models:
            default_value = model in st.session_state.selected_models
            if st.sidebar.checkbox(model, value=default_value, key=f"forecast_model_{model}"):
                selected_models.append(model)

    if individual_models:
        st.sidebar.markdown("**Individual Models:**")
        for model in individual_models:
            default_value = model in st.session_state.selected_models
            if st.sidebar.checkbox(model, value=default_value, key=f"forecast_model_{model}"):
                selected_models.append(model)

    st.session_state.selected_models = selected_models

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Summary")
    st.sidebar.markdown(f"**Forecast dates available:** {len(available_dates)}")
    st.sidebar.markdown(f"**Models loaded:** {len(all_models)}")
    st.sidebar.markdown(f"**Selected forecast date:** {selected_date.strftime('%Y-%m-%d')}")

if tab_selection == "Evaluation":
    st.sidebar.header("Evaluation Controls")
    
    # Load evaluation data
    wis_data = load_wis_data()
    coverage_data = load_coverage_data()
    
    if wis_data is None and coverage_data is None:
        st.error("Could not load evaluation data. Please ensure the parquet files exist.")
    else:
        # Get available models from evaluation data
        eval_models = []
        if wis_data is not None:
            eval_models = sorted(wis_data['Model'].unique().tolist())
        elif coverage_data is not None:
            eval_models = sorted(coverage_data['Model'].unique().tolist())
        
        st.sidebar.markdown("### Select Models to Display")
        
        eval_ensemble_models = [m for m in eval_models if m == 'Median Epistorm Ensemble']
        eval_individual_models = [m for m in eval_models if m != 'Median Epistorm Ensemble' and m != 'FluSight-ensemble']
        
        if 'selected_eval_models' not in st.session_state:
            st.session_state.selected_eval_models = eval_models.copy()
        
        selected_eval_models = []
        
        if eval_ensemble_models:
            st.sidebar.markdown("**Ensemble Models:**")
            for model in eval_ensemble_models:
                default_value = model in st.session_state.selected_eval_models
                if st.sidebar.checkbox(model, value=default_value, key=f"eval_model_{model}"):
                    selected_eval_models.append(model)
        
        if eval_individual_models:
            st.sidebar.markdown("**Individual Models:**")
            for model in eval_individual_models:
                default_value = model in st.session_state.selected_eval_models
                if st.sidebar.checkbox(model, value=default_value, key=f"eval_model_{model}"):
                    selected_eval_models.append(model)
        
        st.session_state.selected_eval_models = selected_eval_models
        
        eval_location_map = {'All Locations': 'all'}
        if locations_df is not None:
            eval_location_codes = []
            if wis_data is not None:
                eval_location_codes = sorted(wis_data['location'].unique().tolist())
            elif coverage_data is not None:
                eval_location_codes = sorted(coverage_data['location'].unique().tolist())
            
            eval_location_names = ['All Locations']
            for loc_code in eval_location_codes:
                loc_name = get_location_name(loc_code, locations_df)
                eval_location_names.append(loc_name)
                eval_location_map[loc_name] = loc_code
            
            selected_eval_location_name = st.sidebar.selectbox(
                "Select Location",
                eval_location_names,
                index=0,
                key="eval_location"
            )
            selected_eval_location = eval_location_map[selected_eval_location_name]
        else:
            selected_eval_location = st.sidebar.text_input("Enter Location Code (or 'all')", value="all", key="eval_location_text")
        
        horizon_options = ['All Horizons', 0, 1, 2, 3]
        selected_horizon = st.sidebar.selectbox(
            "Select Horizon",
            horizon_options,
            index=0,
            key="eval_horizon"
        )
        
        st.sidebar.markdown("### Date Filters")
        
        if wis_data is not None:
            min_ref_date = wis_data['reference_date'].min().date()
            max_ref_date = wis_data['reference_date'].max().date()
            min_target_date = wis_data['target_end_date'].min().date()
            max_target_date = wis_data['target_end_date'].max().date()
        else:
            min_ref_date = datetime(2024, 1, 1).date()
            max_ref_date = datetime.now().date()
            min_target_date = min_ref_date
            max_target_date = max_ref_date
        
        ref_date_range = st.sidebar.date_input(
            "Reference Date Range",
            value=(min_ref_date, max_ref_date),
            min_value=min_ref_date,
            max_value=max_ref_date,
            key="eval_ref_date_range"
        )
        
        target_date_range = st.sidebar.date_input(
            "Target End Date Range",
            value=(min_target_date, max_target_date),
            min_value=min_target_date,
            max_value=max_target_date,
            key="eval_target_date_range"
        )
        
        if isinstance(ref_date_range, tuple) and len(ref_date_range) == 2:
            ref_start, ref_end = ref_date_range
        else:
            ref_start = ref_end = ref_date_range
        
        if isinstance(target_date_range, tuple) and len(target_date_range) == 2:
            target_start, target_end = target_date_range
        else:
            target_start = target_end = target_date_range
        
        # Filter data
        filtered_wis = wis_data.copy() if wis_data is not None else None
        filtered_coverage = coverage_data.copy() if coverage_data is not None else None
        
        if filtered_wis is not None:
            if selected_eval_models:
                filtered_wis = filtered_wis[filtered_wis['Model'].isin(selected_eval_models)]
            if selected_eval_location != 'all':
                filtered_wis = filtered_wis[filtered_wis['location'] == selected_eval_location]
            if selected_horizon != 'All Horizons':
                filtered_wis = filtered_wis[filtered_wis['horizon'] == selected_horizon]
            filtered_wis = filtered_wis[
                (filtered_wis['reference_date'].dt.date >= ref_start) &
                (filtered_wis['reference_date'].dt.date <= ref_end) &
                (filtered_wis['target_end_date'].dt.date >= target_start) &
                (filtered_wis['target_end_date'].dt.date <= target_end)
            ]
        
        if filtered_coverage is not None:
            if selected_eval_models:
                filtered_coverage = filtered_coverage[filtered_coverage['Model'].isin(selected_eval_models)]
            if selected_eval_location != 'all':
                filtered_coverage = filtered_coverage[filtered_coverage['location'] == selected_eval_location]
            if selected_horizon != 'All Horizons':
                filtered_coverage = filtered_coverage[filtered_coverage['horizon'] == selected_horizon]
            filtered_coverage = filtered_coverage[
                (filtered_coverage['reference_date'].dt.date >= ref_start) &
                (filtered_coverage['reference_date'].dt.date <= ref_end) &
                (filtered_coverage['target_end_date'].dt.date >= target_start) &
                (filtered_coverage['target_end_date'].dt.date <= target_end)
            ]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Evaluation Data Summary")
        if filtered_wis is not None:
            st.sidebar.markdown(f"**WIS observations:** {len(filtered_wis):,}")
        if filtered_coverage is not None:
            st.sidebar.markdown(f"**Coverage observations:** {len(filtered_coverage):,}")
        st.sidebar.markdown(f"**Models selected:** {len(selected_eval_models)}")

# ============== MAIN CONTENT ==============
if tab_selection == "Forecasts":
    if selected_models:
        fig, location_name = plot_forecasts(observed_data, forecast_data, selected_location, selected_date, selected_models, available_dates, start_date_ts, end_date_ts)
        
        st.markdown(f"<h2 style='color: #518fb0;'>Flu Hospitalization Forecasts - {location_name}</h2>", unsafe_allow_html=True)

        ensemble_forecast_data = forecast_data[
            (forecast_data['model'] == 'Median Epistorm Ensemble') &
            (forecast_data['location'] == selected_location) &
            (forecast_data['reference_date'] == selected_date) &
            (forecast_data['horizon'] == 3) &
            (forecast_data['output_type'] == 'quantile')
        ].copy()
        
        if not ensemble_forecast_data.empty:
            ensemble_forecast_data['output_type_id'] = ensemble_forecast_data['output_type_id'].astype(float)
            median_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.5]['value'].values
            lower_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.025]['value'].values
            upper_val = ensemble_forecast_data[ensemble_forecast_data['output_type_id'] == 0.975]['value'].values
            
            if len(median_val) > 0 and len(lower_val) > 0 and len(upper_val) > 0:
                median = int(round(median_val[0]))
                lower = int(round(lower_val[0]))
                upper = int(round(upper_val[0]))
                
                population = None
                if locations_df is not None:
                    if selected_location == 'US':
                        pop_row = locations_df[locations_df['location'] == 'US']
                    else:
                        pop_row = locations_df[locations_df['location'] == selected_location]
                    if not pop_row.empty and 'population' in pop_row.columns:
                        population = pop_row.iloc[0]['population']
                
                end_date_4_weeks = ensemble_forecast_data['target_end_date'].iloc[0].strftime('%B %d, %Y')
                heading = f"As of {selected_date.strftime('%B %d, %Y')}, the Epistorm Ensemble forecasts <b>{median:,}</b> influenza hospital admissions"
                
                if population and population > 0:
                    per_capita_median = int(round((median / population) * 100000))
                    per_capita_lower = int(round((lower / population) * 100000))
                    per_capita_upper = int(round((upper / population) * 100000))
                    heading += f" (<b>{per_capita_median:,}</b> per 100,000 people)"
                
                heading += f" by {end_date_4_weeks} for {location_name}."
                pi_text = f"95% prediction interval: <b>{lower:,}–{upper:,}</b> hospital admissions"
                
                if population and population > 0:
                    pi_text += f" (<b>{per_capita_lower:,}–{per_capita_upper:,}</b> per 100,000 people)"
                pi_text += "."
                
                st.markdown(f"<p style='font-size: 22px;'>{heading}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 18px;'>{pi_text}</p>", unsafe_allow_html=True)
        
        event = st.plotly_chart(fig, key="forecast_plot", on_select="rerun", use_container_width=True)
        
        if event and hasattr(event, 'selection') and event.selection:
            try:
                if 'points' in event.selection and len(event.selection['points']) > 0:
                    clicked_x = event.selection['points'][0]['x']
                    clicked_date = pd.to_datetime(clicked_x)
                    closest_date = min(available_dates, key=lambda d: abs((d - clicked_date).total_seconds()))
                    if closest_date != st.session_state.selected_date:
                        st.session_state.selected_date = closest_date
                        st.rerun()
            except:
                pass
        
        st.markdown("💡 **Tip:** Click on any date on the plot to update the forecast reference date.")
        st.markdown("---")
        
        st.markdown(
            """
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 650px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -150px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 18px;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
            <h3 class="tooltip" style="color: #518fb0;">Categorical Forecasts: Weekly Hospitalization Rate Change
                <span class="tooltiptext">For the categorical forecasts, teams predict the trend of weekly hospitalization rates over 1-4 weeks, with predictions submitted as probabilities. These plots show the probability of each trend category (increasing, decreasing, or stable) for each week ahead. The ensemble is computed by taking the mean of the individual model predictions and ensuring the probabilities sum to 1.</span>
            </h3>
            """,
            unsafe_allow_html=True
        )

        ensemble_cat_data = forecast_data[
            (forecast_data['model'] == 'Median Epistorm Ensemble') &
            (forecast_data['location'] == selected_location) &
            (forecast_data['reference_date'] == selected_date) &
            (forecast_data['horizon'] == 3) &
            (forecast_data['output_type'] == 'pmf') &
            (forecast_data['target'] == 'wk flu hosp rate change')
        ].copy()

        if not ensemble_cat_data.empty:
            ensemble_cat_data['category'] = ensemble_cat_data['output_type_id'].apply(format_category)
            ensemble_cat_data = ensemble_cat_data[ensemble_cat_data['category'].isin(CATEGORY_ORDER)]
            
            if not ensemble_cat_data.empty:
                sorted_cats = ensemble_cat_data.sort_values('value', ascending=False)
                
                if len(sorted_cats) >= 1:
                    max_prob = sorted_cats.iloc[0]['value']
                    highest_cats = sorted_cats[sorted_cats['value'] == max_prob]
                    
                    if len(highest_cats) == 1:
                        cat_name = highest_cats.iloc[0]['category']
                        cat_color = CATEGORY_COLORS.get(cat_name, '#000000')
                        if cat_name == 'Stable':
                            cat_color = 'dimgray'
                        prob_pct = max_prob * 100
                        heading = f"As of {selected_date.strftime('%B %d, %Y')}, the Epistorm Ensemble forecasts a trend of: <b><span style='color:{cat_color}'>{cat_name}</span></b> with <b>{prob_pct:.1f}%</b> probability by {end_date_4_weeks} for {location_name}."
                    else:
                        cat_texts = []
                        for _, row in highest_cats.iterrows():
                            cat_name = row['category']
                            cat_color = CATEGORY_COLORS.get(cat_name, '#000000')
                            cat_texts.append(f"<b><span style='color:{cat_color}'>{cat_name}</span></b>")
                        cat_list = " and ".join(cat_texts) if len(cat_texts) == 2 else ", ".join(cat_texts[:-1]) + f", and {cat_texts[-1]}"
                        prob_pct = max_prob * 100
                        heading = f"As of {selected_date.strftime('%B %d, %Y')}, the Epistorm Ensemble forecasts {cat_list} (tied) with <b>{prob_pct:.1f}%</b> probability each by {end_date_4_weeks} for {location_name}."
                    
                    st.markdown(f"<p style='font-size: 22px;'>{heading}</p>", unsafe_allow_html=True)
                    
                    remaining_cats = sorted_cats[sorted_cats['value'] < max_prob]
                    if len(remaining_cats) >= 1:
                        second_max_prob = remaining_cats.iloc[0]['value']
                        second_highest_cats = remaining_cats[remaining_cats['value'] == second_max_prob]
                        
                        if len(second_highest_cats) == 1:
                            cat_name = second_highest_cats.iloc[0]['category']
                            cat_color = CATEGORY_COLORS.get(cat_name, '#000000')
                            if cat_name == 'Stable':
                                cat_color = 'dimgray'
                            prob_pct = second_max_prob * 100
                            secondary_text = f"A rate trend of <b><span style='color:{cat_color}'>{cat_name}</span></b> is the next most likely with <b>{prob_pct:.1f}%</b> probability."
                        else:
                            cat_texts = []
                            for _, row in second_highest_cats.iterrows():
                                cat_name = row['category']
                                cat_color = CATEGORY_COLORS.get(cat_name, '#000000')
                                cat_texts.append(f"<b><span style='color:{cat_color}'>{cat_name}</span></b>")
                            cat_list = " and ".join(cat_texts) if len(cat_texts) == 2 else ", ".join(cat_texts[:-1]) + f", and {cat_texts[-1]}"
                            prob_pct = second_max_prob * 100
                            secondary_text = f"Second most likely (tied): {cat_list} with <b>{prob_pct:.1f}%</b> probability each."
                        
                        st.markdown(f"<p style='font-size: 18px;'>{secondary_text}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 22px;'>No categorical forecast available for the Epistorm Ensemble for this forecast date.</p>", unsafe_allow_html=True)

        cat_fig, models_without_data = plot_categorical_forecasts(
            forecast_data, selected_location, selected_date, selected_models
        )
        
        if cat_fig is not None:
            st.plotly_chart(cat_fig, use_container_width=True)
            if models_without_data:
                st.info(f"ℹ️ The following models do not have categorical forecast data for this forecast date: {', '.join(models_without_data)}")
        else:
            st.warning("⚠️ No categorical forecast data available for the selected models, location, and date.")
            if models_without_data:
                st.info(f"Models without categorical data: {', '.join(models_without_data)}")
    else:
        st.warning("Please select at least one model to display")

if tab_selection == "Evaluation":

    # Check if evaluation data was loaded
    wis_data = load_wis_data()
    coverage_data = load_coverage_data()
    
    if wis_data is None and coverage_data is None:
        st.error("Could not load evaluation data. Please ensure the parquet files exist.")
    else:
        # Get filter values from sidebar (already set above)
        selected_eval_models = st.session_state.get('selected_eval_models', [])
        selected_eval_location = st.session_state.get('eval_location', 'All Locations')

        st.markdown(f"<h2 style='color: #518fb0;'>Forecast Evaluation - {selected_eval_location}</h2>", unsafe_allow_html=True)
        st.markdown("Evaluate forecast performance using the Weighted Interval Score (WIS) and prediction interval coverage metrics.")
    

        if selected_eval_location != 'All Locations':
            # Convert location name back to code if needed
            if locations_df is not None:
                eval_location_codes = sorted(wis_data['location'].unique().tolist()) if wis_data is not None else []
                eval_location_map = {'All Locations': 'all'}
                for loc_code in eval_location_codes:
                    loc_name = get_location_name(loc_code, locations_df)
                    eval_location_map[loc_name] = loc_code
                selected_eval_location = eval_location_map.get(selected_eval_location, 'all')
        else:
            selected_eval_location = 'all'
        
        selected_horizon = st.session_state.get('eval_horizon', 'All Horizons')
        
        ref_date_range = st.session_state.get('eval_ref_date_range', None)
        target_date_range = st.session_state.get('eval_target_date_range', None)
        
        if wis_data is not None:
            min_ref_date = wis_data['reference_date'].min().date()
            max_ref_date = wis_data['reference_date'].max().date()
            min_target_date = wis_data['target_end_date'].min().date()
            max_target_date = wis_data['target_end_date'].max().date()
        else:
            min_ref_date = datetime(2024, 1, 1).date()
            max_ref_date = datetime.now().date()
            min_target_date = min_ref_date
            max_target_date = max_ref_date
        
        if ref_date_range is not None:
            if isinstance(ref_date_range, tuple) and len(ref_date_range) == 2:
                ref_start, ref_end = ref_date_range
            else:
                ref_start = ref_end = ref_date_range
        else:
            ref_start, ref_end = min_ref_date, max_ref_date
        
        if target_date_range is not None:
            if isinstance(target_date_range, tuple) and len(target_date_range) == 2:
                target_start, target_end = target_date_range
            else:
                target_start = target_end = target_date_range
        else:
            target_start, target_end = min_target_date, max_target_date
        
        # Filter data
        filtered_wis = wis_data.copy() if wis_data is not None else None
        filtered_coverage = coverage_data.copy() if coverage_data is not None else None
        
        if filtered_wis is not None:
            if selected_eval_models:
                filtered_wis = filtered_wis[filtered_wis['Model'].isin(selected_eval_models)]
            if selected_eval_location != 'all':
                filtered_wis = filtered_wis[filtered_wis['location'] == selected_eval_location]
            if selected_horizon != 'All Horizons':
                filtered_wis = filtered_wis[filtered_wis['horizon'] == selected_horizon]
            filtered_wis = filtered_wis[
                (filtered_wis['reference_date'].dt.date >= ref_start) &
                (filtered_wis['reference_date'].dt.date <= ref_end) &
                (filtered_wis['target_end_date'].dt.date >= target_start) &
                (filtered_wis['target_end_date'].dt.date <= target_end)
            ]
        
        if filtered_coverage is not None:
            if selected_eval_models:
                filtered_coverage = filtered_coverage[filtered_coverage['Model'].isin(selected_eval_models)]
            if selected_eval_location != 'all':
                filtered_coverage = filtered_coverage[filtered_coverage['location'] == selected_eval_location]
            if selected_horizon != 'All Horizons':
                filtered_coverage = filtered_coverage[filtered_coverage['horizon'] == selected_horizon]
            filtered_coverage = filtered_coverage[
                (filtered_coverage['reference_date'].dt.date >= ref_start) &
                (filtered_coverage['reference_date'].dt.date <= ref_end) &
                (filtered_coverage['target_end_date'].dt.date >= target_start) &
                (filtered_coverage['target_end_date'].dt.date <= target_end)
            ]
        
        # Display WIS boxplot
        st.markdown("### Weighted Interval Score (WIS) Ratio")
        st.markdown("""
        The WIS ratio compares each model's Weighted Interval Score to the FluSight-baseline model. 
        A ratio less than 1 indicates better performance than baseline, while a ratio greater than 1 indicates worse performance.
        """)
        
        if filtered_wis is not None and not filtered_wis.empty:
            wis_fig = plot_wis_boxplot_evaluation(filtered_wis, selected_eval_models, locations_df)
            if wis_fig:
                st.plotly_chart(wis_fig, use_container_width=False)
            else:
                st.warning("No WIS data available for the selected filters.")
        else:
            st.warning("No WIS data available for the selected filters.")
        
        st.markdown("---")
        
        # Display coverage plot
        st.markdown("### Prediction Interval Coverage")
        st.markdown("""
        Coverage measures the proportion of observed values that fall within each prediction interval. 
        A well-calibrated model should have coverage close to the diagonal line (e.g., 50% of observations within the 50% prediction interval).
        """)
        
        if filtered_coverage is not None and not filtered_coverage.empty:
            cov_fig = plot_coverage_evaluation(filtered_coverage, selected_eval_models)
            if cov_fig:
                st.plotly_chart(cov_fig, use_container_width=False)
            else:
                st.warning("No coverage data available for the selected filters.")
        else:
            st.warning("No coverage data available for the selected filters.")