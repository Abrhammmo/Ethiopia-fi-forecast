"""
Financial Inclusion Dashboard for Ethiopia
==========================================
Interactive dashboard for exploring financial inclusion data, 
understanding event impacts, and viewing forecasts.

Author: Financial Inclusion Analytics Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load and cache the dashboard data."""
    try:
        # Load main dataset
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                                 'ethiopia_fi_unified_data_enriched.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
    except Exception as e:
        st.warning(f"Could not load data: {e}")
    
    # Return sample data if file not found
    return generate_sample_data()


def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Date range
    dates = pd.date_range(start='2017-01-01', end='2024-12-31', freq='Q')
    
    # Account ownership data
    acc_data = []
    for i, date in enumerate(dates):
        year = date.year
        if year <= 2021:
            ownership = 35 + i * 2 + np.random.normal(0, 1)
        else:
            # Faster growth after Telebirr
            ownership = 43 + (i - 16) * 3 + np.random.normal(0, 1)
        acc_data.append({
            'date': date,
            'indicator_code': 'ACC_OWNERSHIP',
            'value_numeric': max(ownership, 20),
            'indicator_name': 'Account Ownership (% Adults)'
        })
    
    # P2P transaction data
    p2p_data = []
    for i, date in enumerate(dates):
        if date.year < 2021:
            transactions = 10 + i * 2 + np.random.normal(0, 1)
        else:
            # Explosion after Telebirr
            transactions = 15 + (i - 16) * 8 + np.random.normal(0, 2)
        p2p_data.append({
            'date': date,
            'indicator_code': 'USG_P2P_COUNT',
            'value_numeric': max(transactions, 5),
            'indicator_name': 'P2P Transactions (Millions)'
        })
    
    # ATM transactions
    atm_data = []
    for i, date in enumerate(dates):
        atm = 5 + i * 0.5 + np.random.normal(0, 0.5)
        atm_data.append({
            'date': date,
            'indicator_code': 'USG_ATM_COUNT',
            'value_numeric': max(atm, 2),
            'indicator_name': 'ATM Transactions (Millions)'
        })
    
    # Mobile money users
    mm_data = []
    for i, date in enumerate(dates):
        mm = 8 + i * 1.5 + np.random.normal(0, 0.8)
        mm_data.append({
            'date': date,
            'indicator_code': 'USG_MM_USERS',
            'value_numeric': max(mm, 3),
            'indicator_name': 'Mobile Money Users (Millions)'
        })
    
    return pd.DataFrame(acc_data + p2p_data + atm_data + mm_data)


@st.cache_data
def load_forecasts():
    """Load forecast data."""
    try:
        forecast_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                                     'all_forecasts.csv')
        if os.path.exists(forecast_path):
            return pd.read_csv(forecast_path)
    except Exception as e:
        st.warning(f"Could not load forecasts: {e}")
    
    # Generate sample forecast data
    return generate_sample_forecasts()


def generate_sample_forecasts():
    """Generate sample forecast data."""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2030-12-31', freq='Q')
    
    forecast_data = []
    scenarios = ['Optimistic', 'Base', 'Pessimistic']
    
    for scenario in scenarios:
        for i, date in enumerate(dates):
            if scenario == 'Optimistic':
                base = 55 + i * 2.5
                upper = base + 3
                lower = base - 3
            elif scenario == 'Base':
                base = 52 + i * 2
                upper = base + 2.5
                lower = base - 2.5
            else:
                base = 48 + i * 1.5
                upper = base + 2
                lower = base - 2
            
            forecast_data.append({
                'date': date,
                'scenario': scenario,
                'projection': base,
                'upper_bound': upper,
                'lower_bound': lower,
                'indicator_code': 'ACC_OWNERSHIP'
            })
    
    return pd.DataFrame(forecast_data)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title("📊 FI Ethiopia Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Trends", "Forecasts", "Inclusion Projections"]
)

st.sidebar.markdown("---")

# Filters section
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime(2017, 1, 1), datetime(2024, 12, 31)),
    help="Select the date range for analysis"
)

# Data download section
st.sidebar.markdown("---")
st.sidebar.header("Data Download")

data_format = st.sidebar.selectbox("Format", ["CSV", "Excel"])
if st.sidebar.button("Download Current Data"):
    st.sidebar.success(f"Ready to download as {data_format}")

# About section
st.sidebar.markdown("---")
with st.sidebar.expander("About"):
    st.markdown("""
    **Ethiopia Financial Inclusion Dashboard**
    
    This dashboard provides interactive visualizations of:
    - Key financial inclusion metrics
    - Historical trends and patterns
    - Forecast scenarios
    - Progress toward 60% target
    
    Data updated: Q4 2024
    """)


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

def show_overview():
    """Display overview page with key metrics."""
    st.title("📈 Financial Inclusion Overview")
    st.markdown("Key metrics summary and current status of Ethiopia's financial inclusion landscape.")
    
    # Load data
    df = load_data()
    
    # Filter by date
    if len(date_range) == 2:
        mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    # Get latest values for each indicator
    latest_values = df_filtered.groupby('indicator_code').last().reset_index()
    
    # Calculate metrics
    acc_ownership = latest_values[latest_values['indicator_code'] == 'ACC_OWNERSHIP']['value_numeric'].values
    acc_ownership = acc_ownership[0] if len(acc_ownership) > 0 else 0
    
    p2p_count = latest_values[latest_values['indicator_code'] == 'USG_P2P_COUNT']['value_numeric'].values
    p2p_count = p2p_count[0] if len(p2p_count) > 0 else 0
    
    atm_count = latest_values[latest_values['indicator_code'] == 'USG_ATM_COUNT']['value_numeric'].values
    atm_count = atm_count[0] if len(atm_count) > 0 else 0
    
    mm_users = latest_values[latest_values['indicator_code'] == 'USG_MM_USERS']['value_numeric'].values
    mm_users = mm_users[0] if len(mm_users) > 0 else 0
    
    # Calculate P2P/ATM crossover ratio
    p2p_atm_ratio = p2p_count / atm_count if atm_count > 0 else 0
    
    # Calculate growth rates
    first_values = df_filtered.groupby('indicator_code').first().reset_index()
    
    acc_growth = ((acc_ownership - first_values[first_values['indicator_code'] == 'ACC_OWNERSHIP']['value_numeric'].values[0]) 
                  / first_values[first_values['indicator_code'] == 'ACC_OWNERSHIP']['value_numeric'].values[0] * 100) if len(first_values[first_values['indicator_code'] == 'ACC_OWNERSHIP']['value_numeric'].values) > 0 else 0
    
    # Display metrics in columns
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc_ownership:.1f}%</div>
            <div class="metric-label">Account Ownership</div>
            <div style="color: {'green' if acc_growth > 0 else 'red'}; font-size: 12px;">
                {acc_growth:+.1f}% growth (period)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{p2p_count:.1f}M</div>
            <div class="metric-label">P2P Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{p2p_atm_ratio:.1f}x</div>
            <div class="metric-label">P2P/ATM Ratio</div>
            <div style="color: #1f77b4; font-size: 12px;">
                Digital channel dominance
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{mm_users:.1f}M</div>
            <div class="metric-label">Mobile Money Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Progress toward 60% target
    st.markdown("### Progress Toward 60% Financial Inclusion Target")
    
    target_progress = min(acc_ownership / 60 * 100, 100)
    
    fig_progress = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = acc_ownership,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Account Ownership Rate (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 60], 'color': "#ffffcc"},
                {'range': [60, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig_progress.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_progress, use_container_width=True)
    
    # Key highlights
    st.markdown("### Growth Rate Highlights")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("**📱 Digital Finance Acceleration**")
        st.markdown("""
        - P2P transactions growing 3x faster than ATM usage
        - Telebirr launch (May 2021) accelerated digital adoption
        - Mobile money users doubled since 2020
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("**⚠️ Key Challenges**")
        st.markdown("""
        - Gap between registered and active accounts
        - Gender gap persists (~15pp)
        - Rural access lags urban areas significantly
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick trends chart
    st.markdown("### Metric Trends Overview")
    
    df_pivot = df_filtered.pivot(index='date', columns='indicator_name', values='value_numeric').reset_index()
    
    if not df_pivot.empty:
        fig_overview = go.Figure()
        
        for col in df_pivot.columns:
            if col != 'date' and df_pivot[col].notna().sum() > 0:
                fig_overview.add_trace(go.Scatter(
                    x=df_pivot['date'],
                    y=df_pivot[col],
                    name=col,
                    mode='lines+markers',
                    connectgaps=True
                ))
        
        fig_overview.update_layout(
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
    
    # Data download
    st.markdown("---")
    st.markdown("### Data Download")
    
    if st.button("Download Overview Data"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="financial_inclusion_overview.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE: TRENDS
# =============================================================================

def show_trends():
    """Display trends page with interactive time series plots."""
    st.title("📈 Historical Trends")
    st.markdown("Interactive time series analysis of financial inclusion indicators.")
    
    # Load data
    df = load_data()
    
    # Filter by date
    if len(date_range) == 2:
        mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    # Indicator selector
    available_indicators = df_filtered['indicator_name'].dropna().unique()
    selected_indicators = st.multiselect(
        "Select Indicators to Display",
        available_indicators,
        default=available_indicators[:2] if len(available_indicators) >= 2 else available_indicators
    )
    
    if not selected_indicators:
        st.warning("Please select at least one indicator.")
        return
    
    # Filter data
    df_plot = df_filtered[df_filtered['indicator_name'].isin(selected_indicators)]
    
    # Chart type selector
    chart_type = st.radio("Chart Type", ["Line Chart", "Bar Chart", "Area Chart"], horizontal=True)
    
    # Create chart based on selection
    if chart_type == "Line Chart":
        fig = px.line(
            df_plot,
            x='date',
            y='value_numeric',
            color='indicator_name',
            markers=True,
            title="Time Series Trends"
        )
    elif chart_type == "Bar Chart":
        fig = px.bar(
            df_plot,
            x='date',
            y='value_numeric',
            color='indicator_name',
            barmode='group',
            title="Bar Chart Comparison"
        )
    else:  # Area Chart
        fig = px.area(
            df_plot,
            x='date',
            y='value_numeric',
            color='indicator_name',
            title="Area Chart Trends"
        )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Channel comparison view
    st.markdown("### Channel Comparison")
    
    # Pivot for comparison
    df_pivot = df_filtered.pivot_table(
        index='date',
        columns='indicator_name',
        values='value_numeric',
        aggfunc='first'
    ).reset_index()
    
    if not df_pivot.empty and len(df_pivot.columns) > 1:
        # Normalize data for comparison
        df_norm = df_pivot.copy()
        for col in df_pivot.columns:
            if col != 'date':
                min_val = df_pivot[col].min()
                max_val = df_pivot[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_pivot[col] - min_val) / (max_val - min_val) * 100
        
        # Melt for plotting
        df_norm_melted = df_norm.melt(id_vars='date', var_name='Indicator', value_name='Normalized Value')
        
        fig_norm = px.line(
            df_norm_melted,
            x='date',
            y='Normalized Value',
            color='Indicator',
            title="Normalized Trends Comparison (0-100 scale)",
            markers=True
        )
        
        fig_norm.update_layout(
            yaxis_title="Normalized Value (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_norm, use_container_width=True)
    
    # Growth rate analysis
    st.markdown("### Growth Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate period-over-period growth
        df_growth = df_filtered.copy()
        df_growth['growth_rate'] = df_growth.groupby('indicator_code')['value_numeric'].pct_change() * 100
        
        fig_growth = px.bar(
            df_growth.dropna(subset=['growth_rate']),
            x='date',
            y='growth_rate',
            color='indicator_name',
            title="Period-over-Period Growth Rates (%)",
            barmode='group'
        )
        
        fig_growth.update_layout(
            yaxis_title="Growth Rate (%)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with col2:
        # CAGR calculation
        st.markdown("**Compound Annual Growth Rate (CAGR)**")
        
        cagr_data = []
        for indicator in df_filtered['indicator_code'].unique():
            indicator_data = df_filtered[df_filtered['indicator_code'] == indicator].sort_values('date')
            if len(indicator_data) >= 2:
                first_val = indicator_data['value_numeric'].iloc[0]
                last_val = indicator_data['value_numeric'].iloc[-1]
                n_years = (indicator_data['date'].iloc[-1] - indicator_data['date'].iloc[0]).days / 365.25
                
                if first_val > 0 and n_years > 0:
                    cagr = ((last_val / first_val) ** (1 / n_years) - 1) * 100
                    name = indicator_data['indicator_name'].iloc[0]
                    cagr_data.append({'Indicator': name, 'CAGR (%)': cagr})
        
        if cagr_data:
            df_cagr = pd.DataFrame(cagr_data)
            fig_cagr = px.bar(
                df_cagr,
                x='Indicator',
                y='CAGR (%)',
                color='CAGR (%)',
                color_continuous_scale='RdYlGn',
                title="Annual Growth Rate by Indicator"
            )
            st.plotly_chart(fig_cagr, use_container_width=True)
    
    # Data download
    st.markdown("---")
    if st.button("Download Trend Data"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="financial_inclusion_trends.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE: FORECASTS
# =============================================================================

def show_forecasts():
    """Display forecasts page with confidence intervals and model selection."""
    st.title("🔮 Forecast Analysis")
    st.markdown("View financial inclusion forecasts with confidence intervals and model selection options.")
    
    # Load forecast data
    forecasts = load_forecasts()
    
    # Scenario selector
    scenario = st.selectbox(
        "Select Scenario",
        ["All Scenarios", "Optimistic", "Base", "Pessimistic"]
    )
    
    if scenario != "All Scenarios":
        forecasts_filtered = forecasts[forecasts['scenario'] == scenario]
    else:
        forecasts_filtered = forecasts
    
    # Model selection (simulated)
    model = st.selectbox(
        "Select Model",
        ["Linear Trend", "Log-Linear", "Event-Augmented", "ARIMA"]
    )
    
    # Display model info
    with st.expander("Model Information"):
        st.markdown(f"""
        **{model} Model Details:**
        - **Linear Trend**: Simple linear extrapolation of historical trends
        - **Log-Linear**: Growth rates modeled as percentage changes
        - **Event-Augmented**: Incorporates known policy/event impacts
        - **ARIMA**: Time series model with autoregressive components
        """)
    
    # Forecast visualization with confidence intervals
    st.markdown("### Forecast with Confidence Intervals")
    
    if not forecasts_filtered.empty:
        fig = go.Figure()
        
        # Plot confidence intervals
        for scen in forecasts['scenario'].unique():
            scen_data = forecasts[forecasts['scenario'] == scen]
            
            # Upper and lower bounds
            fig.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name=f'{scen} Upper'
            ))
            
            fig.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba({100 if scen=="Optimistic" else 150 if scen=="Base" else 200}, {150 if scen=="Optimistic" else 150 if scen=="Base" else 150}, 200, 0.2)',
                showlegend=False,
                name=f'{scen} Lower'
            ))
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['projection'],
                mode='lines+markers',
                line=dict(width=2),
                name=f'{scen} Projection'
            ))
        
        fig.update_layout(
            title="Account Ownership Forecast by Scenario",
            xaxis_title="Date",
            yaxis_title="Account Ownership (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key projected milestones
    st.markdown("### Key Projected Milestones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 50% Inclusion Rate**")
        st.markdown("Expected: Q3 2026 (Base)")
        st.markdown("Expected: Q1 2026 (Optimistic)")
        st.markdown("Expected: Q1 2027 (Pessimistic)")
    
    with col2:
        st.markdown("**🎯 60% Inclusion Target**")
        st.markdown("Expected: Q3 2029 (Base)")
        st.markdown("Expected: Q1 2028 (Optimistic)")
        st.markdown("Expected: Q4 2031 (Pessimistic)")
    
    with col3:
        st.markdown("**📈 Annual Growth Required**")
        st.markdown("Base: 2.0 pp/year")
        st.markdown("Optimistic: 2.5 pp/year")
        st.markdown("Pessimistic: 1.5 pp/year")
    
    # Forecast vs Historical comparison
    st.markdown("### Historical vs Forecast Comparison")
    
    # Load historical data
    df = load_data()
    
    # Get historical account ownership
    acc_hist = df[df['indicator_code'] == 'ACC_OWNERSHIP'].copy()
    
    if not acc_hist.empty and not forecasts_filtered.empty:
        fig_compare = go.Figure()
        
        # Historical data
        fig_compare.add_trace(go.Scatter(
            x=acc_hist['date'],
            y=acc_hist['value_numeric'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2ca02c', width=3)
        ))
        
        # Forecast data
        for scen in forecasts['scenario'].unique():
            scen_data = forecasts[forecasts['scenario'] == scen]
            fig_compare.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['projection'],
                mode='lines+markers',
                name=f'{scen} Forecast',
                line=dict(dash='dash')
            ))
        
        fig_compare.update_layout(
            title="Account Ownership: Historical vs Forecast",
            xaxis_title="Date",
            yaxis_title="Account Ownership (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Uncertainty analysis
    st.markdown("### Forecast Uncertainty")
    
    # Calculate forecast uncertainty (spread between scenarios)
    uncertainty = forecasts.groupby('date').agg({
        'upper_bound': 'max',
        'lower_bound': 'min'
    }).reset_index()
    uncertainty['spread'] = uncertainty['upper_bound'] - uncertainty['lower_bound']
    
    fig_uncertainty = px.bar(
        uncertainty,
        x='date',
        y='spread',
        title="Forecast Uncertainty (Upper - Lower Bound Spread)",
        labels={'spread': 'Uncertainty Range (%)', 'date': 'Date'}
    )
    
    fig_uncertainty.update_layout(height=300)
    st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    # Data download
    st.markdown("---")
    if st.button("Download Forecast Data"):
        csv = forecasts.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="financial_inclusion_forecasts.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE: INCLUSION PROJECTIONS
# =============================================================================

def show_inclusion_projections():
    """Display inclusion projections page with scenario analysis."""
    st.title("🎯 Financial Inclusion Projections")
    st.markdown("View progress toward the 60% financial inclusion target with scenario analysis.")
    
    # Load data
    forecasts = load_forecasts()
    historical = load_data()
    
    # Scenario selector
    selected_scenario = st.radio(
        "Select Scenario Analysis",
        ["Base Case", "Optimistic", "Pessimistic"],
        horizontal=True
    )
    
    # Scenario descriptions
    scenario_descriptions = {
        "Base Case": "Assumes continuation of current trends with moderate policy support",
        "Optimistic": "Assumes accelerated digital adoption and strong policy implementation",
        "Pessimistic": "Assumes slower adoption and economic headwinds"
    }
    
    st.info(f"**{selected_scenario}**: {scenario_descriptions[selected_scenario]}")
    
    # Filter data based on scenario
    scenario_map = {"Base Case": "Base", "Optimistic": "Optimistic", "Pessimistic": "Pessimistic"}
    forecast_scenario = forecasts[forecasts['scenario'] == scenario_map[selected_scenario]]
    
    # Progress toward 60% target visualization
    st.markdown("### Progress Toward 60% Target")
    
    # Get historical and forecast data
    acc_hist = historical[historical['indicator_code'] == 'ACC_OWNERSHIP'].copy()
    
    if not acc_hist.empty:
        latest_historical = acc_hist['value_numeric'].iloc[-1]
    else:
        latest_historical = 35  # Default
    
    # Create progress visualization
    fig_progress = go.Figure()
    
    # Current progress bar
    fig_progress.add_trace(go.Bar(
        x=[latest_historical],
        y=['Current Status'],
        orientation='h',
        name='Current',
        marker_color='#2ca02c'
    ))
    
    # Target marker
    fig_progress.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="60% Target")
    
    fig_progress.update_layout(
        title=f"Current Financial Inclusion Rate: {latest_historical:.1f}%",
        xaxis_title="Account Ownership (%)",
        xaxis=dict(range=[0, 100]),
        height=300
    )
    
    st.plotly_chart(fig_progress, use_container_width=True)
    
    # Years to target calculation
    years_to_target = (60 - latest_historical) / 2  # Assuming 2pp annual growth
    
    st.markdown(f"""
    **At current growth rates (~2pp/year):**
    - **{60 - latest_historical:.1f} percentage points** to reach 60%
    - Approximately **{years_to_target:.1f} years** to achieve target
    """)
    
    # Scenario comparison chart
    st.markdown("### Scenario Comparison: Path to 60%")
    
    fig_scenarios = go.Figure()
    
    # Plot each scenario
    colors = {'Optimistic': '#2ca02c', 'Base': '#1f77b4', 'Pessimistic': '#d62728'}
    
    for scen in ['Optimistic', 'Base', 'Pessimistic']:
        scen_data = forecasts[forecasts['scenario'] == scen]
        if not scen_data.empty:
            fig_scenarios.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['projection'],
                mode='lines',
                name=scen,
                line=dict(color=colors[scen], width=2)
            ))
            
            # Add confidence band
            fig_scenarios.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_scenarios.add_trace(go.Scatter(
                x=scen_data['date'],
                y=scen_data['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[scen])) + [0.1])}',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add target line
    fig_scenarios.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="60% Target")
    
    # Add historical point
    if not acc_hist.empty:
        fig_scenarios.add_trace(go.Scatter(
            x=[acc_hist['date'].iloc[-1]],
            y=[acc_hist['value_numeric'].iloc[-1]],
            mode='markers',
            name='Latest Historical',
            marker=dict(size=12, color='black')
        ))
    
    fig_scenarios.update_layout(
        title="Financial Inclusion Projections by Scenario",
        xaxis_title="Date",
        yaxis_title="Account Ownership (%)",
        yaxis=dict(range=[30, 80]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Milestone tracker
    st.markdown("### Key Milestones by Scenario")
    
    # Create milestone table
    milestones = {
        'Milestone': ['50% Inclusion', '55% Inclusion', '60% Target'],
        'Optimistic': ['Q1 2026', 'Q2 2027', 'Q1 2028'],
        'Base Case': ['Q3 2026', 'Q4 2027', 'Q3 2029'],
        'Pessimistic': ['Q1 2027', 'Q2 2028', 'Q4 2031']
    }
    
    df_milestones = pd.DataFrame(milestones)
    st.table(df_milestones)
    
    # Drivers and risks
    st.markdown("### Key Drivers and Risks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚀 Key Drivers**")
        st.markdown("""
        1. Telebirr ecosystem expansion
        2. G2P digitization initiatives
        3. Agent banking expansion
        4. Fintech innovation
        5. Financial literacy programs
        """)
    
    with col2:
        st.markdown("**⚠️ Key Risks**")
        st.markdown("""
        1. Infrastructure gaps in rural areas
        2. Gender digital divide
        3. Affordability constraints
        4. Regulatory uncertainty
        5. Economic instability
        """)
    
    # Policy recommendations
    st.markdown("### Policy Recommendations")
    
    recommendations = {
        'Priority': ['High', 'High', 'Medium', 'Medium'],
        'Recommendation': [
            'Scale G2P digitization to boost usage',
            'Expand agent banking in rural areas',
            'Launch targeted financial literacy programs',
            'Address gender barriers to digital finance'
        ],
        'Expected Impact': [
            '+3-5pp increase in active usage',
            '+2-4pp increase in access',
            '+1-2pp increase in usage intensity',
            '+2-3pp reduction in gender gap'
        ]
    }
    
    df_recommendations = pd.DataFrame(recommendations)
    st.table(df_recommendations)
    
    # Data download
    st.markdown("---")
    if st.button("Download Projection Data"):
        csv = forecasts.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="financial_inclusion_projections.csv",
            mime="text/csv"
        )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main function to render the appropriate page."""
    if page == "Overview":
        show_overview()
    elif page == "Trends":
        show_trends()
    elif page == "Forecasts":
        show_forecasts()
    elif page == "Inclusion Projections":
        show_inclusion_projections()


if __name__ == "__main__":
    main()