"""
Scenario Analysis Module

Provides scenario-based forecasting with optimistic, base, and pessimistic
scenarios, including confidence intervals and visualization support.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pickle
import json


@dataclass
class ScenarioForecast:
    """Container for scenario forecast results."""
    scenario_name: str
    years: List[int]
    values: Dict[str, np.ndarray]  # 'baseline', 'optimistic', 'pessimistic'
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (lower, upper)
    event_impacts: Dict[str, float]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        records = []
        for year in self.years:
            row = {'year': year}
            for scenario in ['baseline', 'optimistic', 'pessimistic']:
                row[f'{scenario}_value'] = self.values.get(scenario, {}).get(year, np.nan)
                lower, upper = self.confidence_intervals.get(scenario, (np.nan, np.nan))
                row[f'{scenario}_lower'] = lower.get(year, np.nan) if isinstance(lower, dict) else np.nan
                row[f'{scenario}_upper'] = upper.get(year, np.nan) if isinstance(upper, dict) else np.nan
            records.append(row)
        return pd.DataFrame(records)
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'scenario_name': self.scenario_name,
            'years': self.years,
            'final_baseline': self.values.get('baseline', {}).get(self.years[-1], np.nan),
            'final_optimistic': self.values.get('optimistic', {}).get(self.years[-1], np.nan),
            'final_pessimistic': self.values.get('pessimistic', {}).get(self.years[-1], np.nan),
            'total_event_impact': sum(self.event_impacts.values()),
            'key_drivers': sorted(self.event_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        }


def generate_scenario_forecasts(
    historical_values: np.ndarray,
    years: List[int],
    trend_model_params: Dict,
    event_impacts: Dict[str, Dict],
    base_scenario_name: str = 'baseline'
) -> ScenarioForecast:
    """
    Generate forecasts for multiple scenarios.
    
    Args:
        historical_values: Historical indicator values
        years: Years to forecast
        trend_model_params: Trend parameters (slope, intercept)
        event_impacts: Event impacts by scenario
        base_scenario_name: Name of baseline scenario
        
    Returns:
        ScenarioForecast object with all scenarios
    """
    n_years = len(years)
    t = np.arange(len(historical_values), len(historical_values) + n_years)
    
    # Base trend
    base_trend = trend_model_params['slope'] * t + trend_model_params['intercept']
    
    # Calculate event impacts for each scenario
    scenarios = {}
    
    for scenario_name, impacts in event_impacts.items():
        total_impact = np.zeros(n_years)
        
        for event_id, params in impacts.items():
            effect_magnitude = params.get('effect_magnitude', 0)
            lag = params.get('lag', 0)
            ramp = params.get('ramp', 12)
            
            # Apply impact with lag and ramp
            for i in range(n_years):
                if i >= lag:
                    adjusted = i - lag
                    if adjusted < ramp:
                        total_impact[i] += effect_magnitude * (adjusted / ramp)
                    else:
                        total_impact[i] += effect_magnitude
        
        scenarios[scenario_name] = base_trend + total_impact
    
    # Calculate confidence intervals (wider for pessimistic)
    confidence_intervals = {}
    base_std = np.std(historical_values[-5:]) if len(historical_values) >= 5 else np.std(historical_values)
    
    for scenario_name, values in scenarios.items():
        if scenario_name == 'optimistic':
            multiplier = 0.8
        elif scenario_name == 'pessimistic':
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        margin = base_std * multiplier * np.sqrt(np.arange(1, n_years + 1))
        confidence_intervals[scenario_name] = (values - margin, values + margin)
    
    # Prepare output
    values_dict = {name: dict(zip(years, vals)) for name, vals in scenarios.items()}
    ci_dict = {name: (dict(zip(years, lower)), dict(zip(years, upper))) 
               for name, (lower, upper) in confidence_intervals.items()}
    
    # Calculate total event impact by event
    total_event_impact = {}
    for scenario_name, impacts in event_impacts.items():
        for event_id, params in impacts.items():
            if event_id not in total_event_impact:
                total_event_impact[event_id] = 0
            total_event_impact[event_id] += params.get('effect_magnitude', 0)
    
    return ScenarioForecast(
        scenario_name=base_scenario_name,
        years=years,
        values=values_dict,
        confidence_intervals=ci_dict,
        event_impacts=total_event_impact
    )


def create_confidence_intervals(
    forecast_values: np.ndarray,
    historical_std: float,
    confidence: float = 0.95,
    time_varying: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create confidence intervals for forecasts.
    
    Args:
        forecast_values: Point forecasts
        historical_std: Standard deviation of historical data
        confidence: Confidence level (0.95 = 95%)
        time_varying: Whether uncertainty increases over time
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    from scipy import stats
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    if time_varying:
        # Uncertainty increases with forecast horizon
        horizon_factor = np.sqrt(np.arange(1, len(forecast_values) + 1))
        margin = z_score * historical_std * horizon_factor * 0.5
    else:
        margin = z_score * historical_std
    
    lower = forecast_values - margin
    upper = forecast_values + margin
    
    # Ensure non-negative for percentage indicators
    lower = np.maximum(lower, 0)
    
    return lower, upper


def calculate_scenario_ranges(
    scenario_forecast: ScenarioForecast,
    metric: str = 'final_value'
) -> Dict:
    """
    Calculate ranges for scenario comparison.
    
    Args:
        scenario_forecast: ScenarioForecast object
        metric: Metric to extract ('final_value', 'total_change', 'avg_growth')
        
    Returns:
        Dictionary with range statistics
    """
    values = scenario_forecast.values
    years = scenario_forecast.years
    
    results = {}
    
    for scenario_name, year_values in values.items():
        values_array = np.array(list(year_values.values()))
        
        if metric == 'final_value':
            results[scenario_name] = values_array[-1]
        elif metric == 'total_change':
            results[scenario_name] = values_array[-1] - values_array[0]
        elif metric == 'avg_growth':
            if values_array[0] > 0:
                results[scenario_name] = ((values_array[-1] / values_array[0]) ** (1/len(years)) - 1) * 100
            else:
                results[scenario_name] = np.nan
    
    # Calculate range
    valid_values = [v for v in results.values() if not np.isnan(v)]
    if valid_values:
        results['range'] = max(valid_values) - min(valid_values)
        results['spread_pct'] = (results['range'] / min(valid_values) * 100) if min(valid_values) > 0 else np.nan
    
    return results


def create_scenario_summary(
    indicator_name: str,
    forecasts: Dict[str, ScenarioForecast],
    target_values: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Create summary table for multiple scenario forecasts.
    
    Args:
        indicator_name: Name of the indicator
        forecasts: Dictionary of scenario_name -> ScenarioForecast
        target_values: Optional target values by year
        
    Returns:
        Summary DataFrame
    """
    records = []
    
    for scenario_name, forecast in forecasts.items():
        for year in forecast.years:
            record = {
                'indicator': indicator_name,
                'scenario': scenario_name,
                'year': year,
                'value': forecast.values.get(scenario_name, {}).get(year, np.nan),
                'lower_ci': forecast.confidence_intervals.get(scenario_name, (np.nan, np.nan))[0].get(year, np.nan),
                'upper_ci': forecast.confidence_intervals.get(scenario_name, (np.nan, np.nan))[1].get(year, np.nan)
            }
            
            if target_values and year in target_values:
                record['target'] = target_values[year]
                record['gap_to_target'] = target_values[year] - record['value']
            
            records.append(record)
    
    return pd.DataFrame(records)


def export_scenario_forecast(
    scenario_forecast: ScenarioForecast,
    filepath: str,
    format: str = 'csv'
) -> None:
    """
    Export scenario forecast to file.
    
    Args:
        scenario_forecast: ScenarioForecast object
        filepath: Output file path
        format: 'csv' or 'json'
    """
    df = scenario_forecast.to_dataframe()
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
        print(f"Scenario forecast exported to {filepath}")
    elif format == 'json':
        # Serialize to dict
        data = {
            'scenario_name': scenario_forecast.scenario_name,
            'years': scenario_forecast.years,
            'values': scenario_forecast.values,
            'event_impacts': scenario_forecast.event_impacts
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Scenario forecast exported to {filepath}")
    else:
        raise ValueError(f"Unknown format: {format}")


def get_scenario_parameters(
    scenario: str,
    base_impact: float,
    event_confidence: str = 'medium'
) -> Dict[str, Dict]:
    """
    Get scenario-specific parameters for event impacts.
    
    Args:
        scenario: 'optimistic', 'base', or 'pessimistic'
        base_impact: Base impact estimate
        event_confidence: 'high', 'medium', or 'low'
        
    Returns:
        Dictionary of event impact parameters
    """
    # Adjust impact based on scenario and confidence
    scenario_multipliers = {
        'optimistic': 1.3,
        'base': 1.0,
        'pessimistic': 0.6
    }
    
    confidence_adjustments = {
        'high': 1.0,
        'medium': 0.8,
        'low': 0.5
    }
    
    multiplier = scenario_multipliers.get(scenario, 1.0) * confidence_adjustments.get(event_confidence, 0.8)
    
    return {'event': {'effect_magnitude': base_impact * multiplier}}
