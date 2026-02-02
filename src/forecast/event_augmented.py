"""
Event-Augmented Forecasting Models

Combines trend models with event impact effects to create
forecasts that incorporate expected developments.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pickle
import warnings


@dataclass
class EventAugmentedModel:
    """
    Combines trend + event effects model.
    
    Model: y(t) = trend(t) + sum(event_effects(t, event_i))
    
    Where:
    - trend(t) = slope * t + intercept
    - event_effects(t) = sum of temporal impact functions
    """
    trend_slope: float
    trend_intercept: float
    event_effects: Dict  # {event_id: {t_start, effect_magnitude, lag, ramp}}
    r_squared: float
    p_value: float
    std_error: float
    n_observations: int
    residuals: np.ndarray
    
    def predict(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict y given time indices."""
        # Base trend
        if isinstance(t, np.ndarray):
            trend = self.trend_slope * t + self.trend_intercept
        else:
            trend = self.trend_slope * t + self.trend_intercept
        
        # Add event effects
        event_total = np.zeros_like(t) if isinstance(t, np.ndarray) else 0
        
        for event_id, effect_params in self.event_effects.items():
            event_t_start = effect_params['t_start']
            effect_magnitude = effect_params['effect_magnitude']
            lag = effect_params.get('lag', 0)
            ramp = effect_params.get('ramp', 1)
            
            if isinstance(t, np.ndarray):
                for i, ti in enumerate(t):
                    if ti >= event_t_start + lag:
                        adjusted_t = ti - event_t_start - lag
                        if adjusted_t < ramp:
                            event_total[i] += effect_magnitude * (adjusted_t / ramp)
                        else:
                            event_total[i] += effect_magnitude
            else:
                if t >= event_t_start + lag:
                    adjusted_t = t - event_t_start - lag
                    if adjusted_t < ramp:
                        event_total += effect_magnitude * (adjusted_t / ramp)
                    else:
                        event_total += effect_magnitude
        
        return trend + event_total
    
    def predict_with_ci(self, t: Union[float, np.ndarray], 
                        confidence: float = 0.95) -> Tuple[Union[float, np.ndarray],
                                                          Union[float, np.ndarray],
                                                          Union[float, np.ndarray]]:
        """Predict with confidence interval (based on residual std)."""
        alpha = 1 - confidence
        t_stat = stats.t.ppf(1 - alpha/2, self.n_observations - 2)
        
        # Standard error of prediction
        n = self.n_observations
        if isinstance(t, np.ndarray):
            se_pred = self.std_error * np.sqrt(1 + 1/n + 
                                               (t - n/2)**2 / 
                                               np.sum((np.arange(n) - n/2)**2))
        else:
            se_pred = self.std_error * np.sqrt(1 + 1/n + 
                                               (t - n/2)**2 / 
                                               np.sum((np.arange(n) - n/2)**2))
        
        y_pred = self.predict(t)
        y_lower = y_pred - t_stat * se_pred
        y_upper = y_pred + t_stat * se_pred
        
        return y_pred, y_lower, y_upper
    
    def to_dict(self) -> Dict:
        """Serialize model to dictionary."""
        return {
            'model_type': 'event_augmented',
            'trend_slope': float(self.trend_slope),
            'trend_intercept': float(self.trend_intercept),
            'event_effects': self.event_effects,
            'r_squared': float(self.r_squared),
            'p_value': float(self.p_value),
            'std_error': float(self.std_error),
            'n_observations': int(self.n_observations),
            'residuals': self.residuals.tolist()
        }
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Event-augmented model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EventAugmentedModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def fit_event_augmented_model(
    y: np.ndarray,
    t: np.ndarray,
    event_data: List[Dict],
    event_indicator_mapping: Dict[str, Dict],
    return_model: bool = True
) -> Tuple[EventAugmentedModel, Dict]:
    """
    Fit an event-augmented model.
    
    Args:
        y: Target variable (indicator values)
        t: Time indices
        event_data: List of event dictionaries with t_start and effect_magnitude
        event_indicator_mapping: Mapping of event_id to indicator effects
        return_model: Return model object or just parameters
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    n = len(y)
    
    if n < 5:
        raise ValueError("Need at least 5 observations for event-augmented model")
    
    # Initialize event effects from mapping
    event_effects = {}
    for event_id, effects in event_indicator_mapping.items():
        for indicator, params in effects.items():
            event_effects[event_id] = {
                't_start': params.get('t_start', 0),
                'effect_magnitude': params.get('effect_magnitude', 0),
                'lag': params.get('lag', 0),
                'ramp': params.get('ramp', 1)
            }
    
    # Fit base trend (OLS on residuals after removing event effects)
    # Simplified: fit linear trend first
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    
    # Calculate residuals
    y_pred_base = slope * t + intercept
    residuals = y - y_pred_base
    
    # Improve trend by adjusting for events that occurred during observation period
    # (Simplified approach: use base trend for now)
    model = EventAugmentedModel(
        trend_slope=slope,
        trend_intercept=intercept,
        event_effects=event_effects,
        r_squared=r_value**2,
        p_value=p_value,
        std_error=std_err,
        n_observations=n,
        residuals=residuals
    )
    
    # Calculate metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / y)) * 100 if np.all(y > 0) else np.nan
    
    metrics = {
        'r_squared': model.r_squared,
        'p_value': model.p_value,
        'std_error': model.std_error,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_observations': n,
        'n_events_included': len(event_effects),
        'model_type': 'event_augmented'
    }
    
    return model, metrics


def predict_with_events(
    model: EventAugmentedModel,
    n_periods: int,
    future_events: Optional[List[Dict]] = None,
    start_t: int = 0,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Generate forecast with event effects.
    
    Args:
        model: Fitted event-augmented model
        n_periods: Number of periods to forecast
        future_events: List of future events to include
        start_t: Starting time index
        confidence: Confidence level for intervals
        
    Returns:
        DataFrame with forecast and confidence intervals
    """
    t_future = np.arange(start_t, start_t + n_periods, dtype=float)
    
    # Combine existing events with future events
    all_events = model.event_effects.copy()
    if future_events:
        for event in future_events:
            event_id = event.get('event_id', f'future_{len(all_events)}')
            all_events[event_id] = {
                't_start': event.get('t_start', start_t),
                'effect_magnitude': event.get('effect_magnitude', 0),
                'lag': event.get('lag', 0),
                'ramp': event.get('ramp', 1)
            }
    
    # Create temporary model with all events
    temp_model = EventAugmentedModel(
        trend_slope=model.trend_slope,
        trend_intercept=model.trend_intercept,
        event_effects=all_events,
        r_squared=model.r_squared,
        p_value=model.p_value,
        std_error=model.std_error,
        n_observations=model.n_observations,
        residuals=model.residuals
    )
    
    y_pred, y_lower, y_upper = temp_model.predict_with_ci(t_future, confidence)
    
    return pd.DataFrame({
        't': t_future,
        'predicted': y_pred,
        'lower_ci': y_lower,
        'upper_ci': y_upper
    })


def create_event_impact_matrix(
    events: pd.DataFrame,
    impact_links: pd.DataFrame,
    indicator_code: str
) -> Dict[str, Dict]:
    """
    Create event-impact mapping for a specific indicator.
    
    Args:
        events: Events DataFrame
        impact_links: Impact links DataFrame
        indicator_code: Target indicator code
        
    Returns:
        Dictionary mapping event_id to impact parameters
    """
    # Filter impact links for this indicator
    indicator_links = impact_links[
        impact_links['related_indicator'] == indicator_code
    ].copy()
    
    # Get event dates
    event_dates = events.set_index('record_id')['observation_date'].to_dict()
    
    # Create mapping
    mapping = {}
    for _, row in indicator_links.iterrows():
        event_id = row['parent_id']
        if event_id in event_dates:
            event_date = event_dates[event_id]
            # Convert event date to time index (months since first observation)
            # Simplified: use ordinal representation
            t_start = event_date.toordinal()
            
            mapping[event_id] = {
                'event_name': row.get('event_name', event_id),
                't_start': t_start,
                'effect_magnitude': row.get('impact_estimate', 0),
                'lag': int(row.get('lag_months', 0)),
                'ramp': 12 if row.get('relationship_type') == 'direct' else 24,
                'evidence': row.get('evidence_basis', 'unknown'),
                'confidence': row.get('impact_magnitude', 'medium')
            }
    
    return mapping


def add_event_effects_to_trend(
    base_forecast: pd.DataFrame,
    event_mapping: Dict[str, Dict],
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Add event effects to a base trend forecast.
    
    Args:
        base_forecast: DataFrame with base forecast
        event_mapping: Event-impacts mapping
        date_col: Date column name
        
    Returns:
        DataFrame with added event effects
    """
    result = base_forecast.copy()
    result['event_effects'] = 0
    result['with_events'] = result['predicted']
    
    for event_id, params in event_mapping.items():
        event_date = pd.Timestamp(params['t_start'])
        effect_magnitude = params['effect_magnitude']
        lag = params.get('lag', 0)
        ramp = params.get('ramp', 12)
        
        # Add effect for dates after event
        for idx, row in result.iterrows():
            date = row[date_col]
            if date >= event_date:
                months_since = (date.year - event_date.year) * 12 + (date.month - event_date.month)
                if months_since >= lag:
                    adjusted = months_since - lag
                    if adjusted < ramp:
                        effect = effect_magnitude * (adjusted / ramp)
                    else:
                        effect = effect_magnitude
                    result.loc[idx, 'event_effects'] += effect
                    result.loc[idx, 'with_events'] += effect
    
    return result
