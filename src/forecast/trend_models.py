"""
Trend Models for Financial Inclusion Forecasting

Provides linear and log-linear trend models with confidence intervals,
validation metrics, and model persistence.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pickle
import json
import os
import warnings


@dataclass
class LinearTrendModel:
    """Linear trend model: y = slope * t + intercept"""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    n_observations: int
    residuals: np.ndarray
    
    def predict(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict y given t (time index)."""
        return self.slope * t + self.intercept
    
    def predict_with_ci(self, t: Union[float, np.ndarray], 
                        confidence: float = 0.95) -> Tuple[Union[float, np.ndarray], 
                                                          Union[float, np.ndarray],
                                                          Union[float, np.ndarray]]:
        """Predict with confidence interval."""
        # Calculate prediction interval
        alpha = 1 - confidence
        t_stat = stats.t.ppf(1 - alpha/2, self.n_observations - 2)
        
        # Standard error of prediction
        se_pred = self.std_error * np.sqrt(1 + 1/self.n_observations + 
                                           (t - self.n_observations/2)**2 / 
                                           np.sum((np.arange(self.n_observations) - 
                                                  self.n_observations/2)**2))
        
        y_pred = self.predict(t)
        y_lower = y_pred - t_stat * se_pred
        y_upper = y_pred + t_stat * se_pred
        
        return y_pred, y_lower, y_upper
    
    def to_dict(self) -> Dict:
        """Serialize model to dictionary."""
        return {
            'model_type': 'linear',
            'slope': float(self.slope),
            'intercept': float(self.intercept),
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
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LinearTrendModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


@dataclass
class LogLinearTrendModel:
    """Log-linear trend model: log(y) = slope * t + intercept -> y = exp(slope * t + intercept)"""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    n_observations: int
    residuals: np.ndarray
    
    def predict(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict y given t (time index)."""
        return np.exp(self.slope * t + self.intercept)
    
    def predict_with_ci(self, t: Union[float, np.ndarray], 
                        confidence: float = 0.95) -> Tuple[Union[float, np.ndarray],
                                                          Union[float, np.ndarray],
                                                          Union[float, np.ndarray]]:
        """Predict with confidence interval (on log scale, then transformed)."""
        # Calculate on log scale
        alpha = 1 - confidence
        t_stat = stats.t.ppf(1 - alpha/2, self.n_observations - 2)
        
        # Standard error of prediction on log scale
        se_pred = self.std_error * np.sqrt(1 + 1/self.n_observations + 
                                           (t - self.n_observations/2)**2 / 
                                           np.sum((np.arange(self.n_observations) - 
                                                  self.n_observations/2)**2))
        
        log_y_pred = self.slope * t + self.intercept
        log_y_lower = log_y_pred - t_stat * se_pred
        log_y_upper = log_y_pred + t_stat * se_pred
        
        # Transform back to original scale
        y_pred = np.exp(log_y_pred)
        y_lower = np.exp(log_y_lower)
        y_upper = np.exp(log_y_upper)
        
        return y_pred, y_lower, y_upper
    
    def growth_rate(self) -> float:
        """Return annualized growth rate (as percentage)."""
        return (np.exp(self.slope) - 1) * 100
    
    def doubling_time(self) -> Optional[float]:
        """Return doubling time in years (if growing)."""
        if self.slope <= 0:
            return None
        return np.log(2) / self.slope / 12  # Convert to years
    
    def to_dict(self) -> Dict:
        """Serialize model to dictionary."""
        return {
            'model_type': 'log_linear',
            'slope': float(self.slope),
            'intercept': float(self.intercept),
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
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LogLinearTrendModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def fit_trend_model(y: np.ndarray, 
                    t: Optional[np.ndarray] = None,
                    model_type: str = 'linear',
                    return_model: bool = True) -> Tuple[Union[LinearTrendModel, LogLinearTrendModel], Dict]:
    """
    Fit a trend model to the data.
    
    Args:
        y: Target variable (indicator values)
        t: Time indices (auto-generated if None)
        model_type: 'linear' or 'log_linear'
        return_model: Return model object or just parameters
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    # Validate input
    if len(y) < 3:
        raise ValueError("Need at least 3 observations to fit trend model")
    
    if np.any(y <= 0) and model_type == 'log_linear':
        raise ValueError("Log-linear model requires positive y values")
    
    # Generate time indices if not provided
    if t is None:
        t = np.arange(len(y), dtype=float)
    
    # Fit model based on type
    if model_type == 'linear':
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        residuals = y - (slope * t + intercept)
        
        model = LinearTrendModel(
            slope=slope,
            intercept=intercept,
            r_squared=r_value**2,
            p_value=p_value,
            std_error=std_err,
            n_observations=len(y),
            residuals=residuals
        )
    elif model_type == 'log_linear':
        log_y = np.log(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_y)
        residuals = log_y - (slope * t + intercept)
        
        model = LogLinearTrendModel(
            slope=slope,
            intercept=intercept,
            r_squared=r_value**2,
            p_value=p_value,
            std_error=std_err,
            n_observations=len(y),
            residuals=residuals
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate additional metrics
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
        'n_observations': len(y),
        'model_type': model_type
    }
    
    if model_type == 'log_linear':
        metrics['annual_growth_rate'] = model.growth_rate()
        metrics['doubling_time_years'] = model.doubling_time()
    
    return model, metrics


def predict_trend(model: Union[LinearTrendModel, LogLinearTrendModel],
                  n_periods: int,
                  start_t: int = 0,
                  confidence: float = 0.95) -> pd.DataFrame:
    """
    Generate forecast using trend model.
    
    Args:
        model: Fitted trend model
        n_periods: Number of periods to forecast
        start_t: Starting time index
        confidence: Confidence level for intervals
        
    Returns:
        DataFrame with forecast and confidence intervals
    """
    t_future = np.arange(start_t, start_t + n_periods, dtype=float)
    y_pred, y_lower, y_upper = model.predict_with_ci(t_future, confidence)
    
    return pd.DataFrame({
        't': t_future,
        'predicted': y_pred,
        'lower_ci': y_lower,
        'upper_ci': y_upper
    })


def compare_models(y: np.ndarray, 
                   t: Optional[np.ndarray] = None) -> Dict:
    """
    Compare linear and log-linear models.
    
    Args:
        y: Target variable
        t: Time indices
        
    Returns:
        Dictionary with comparison metrics
    """
    linear_model, linear_metrics = fit_trend_model(y, t, 'linear')
    log_model, log_metrics = fit_trend_model(y, t, 'log_linear')
    
    comparison = {
        'linear': linear_metrics,
        'log_linear': log_metrics,
        'best_model': 'log_linear' if log_metrics['r_squared'] > linear_metrics['r_squared'] else 'linear',
        'r_squared_improvement': log_metrics['r_squared'] - linear_metrics['r_squared']
    }
    
    return comparison


def select_best_model(y: np.ndarray,
                      t: Optional[np.ndarray] = None,
                      metrics: List[str] = ['r_squared', 'mape']) -> Tuple[str, Dict]:
    """
    Select the best model based on multiple metrics.
    
    Args:
        y: Target variable
        t: Time indices
        metrics: Metrics to consider for selection
        
    Returns:
        Tuple of (best_model_type, all_metrics)
    """
    comparison = compare_models(y, t)
    
    # Score models (lower is better for error metrics, higher for r_squared)
    scores = {'linear': 0, 'log_linear': 0}
    
    for metric in metrics:
        if metric == 'r_squared':
            if comparison['linear'][metric] > comparison['log_linear'][metric]:
                scores['linear'] += 1
            else:
                scores['log_linear'] += 1
        else:  # error metrics (lower is better)
            linear_val = comparison['linear'].get(metric, float('inf'))
            log_val = comparison['log_linear'].get(metric, float('inf'))
            if linear_val < log_val:
                scores['linear'] += 1
            else:
                scores['log_linear'] += 1
    
    best = 'log_linear' if scores['log_linear'] > scores['linear'] else 'linear'
    
    return best, comparison


def add_forecast_columns(df: pd.DataFrame,
                         indicator_col: str,
                         date_col: str = 'observation_date',
                         forecast_years: int = 3) -> pd.DataFrame:
    """
    Add forecast columns to a DataFrame.
    
    Args:
        df: DataFrame with historical_col: Column name data
        indicator for indicator values
        date_col: Column name for dates
        forecast_years: Number of years to forecast
        
    Returns:
        DataFrame with added forecast columns
    """
    result = df.copy()
    
    # Fit model on historical data
    historical = result[result[indicator_col].notna()].copy()
    y = historical[indicator_col].values
    t = np.arange(len(y))
    
    # Select best model
    best_model_type, _ = select_best_model(y, t)
    model, _ = fit_trend_model(y, t, best_model_type)
    
    # Generate forecast
    n_periods = forecast_years * 12  # Monthly forecast
    forecast = predict_trend(model, n_periods, start_t=len(y))
    
    # Add to result
    result['is_forecast'] = False
    
    # Create forecast rows
    last_date = result[date_col].max()
    forecast_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq='M')[1:]
    
    forecast_df = pd.DataFrame({
        date_col: forecast_dates,
        indicator_col: forecast['predicted'].values,
        'is_forecast': True,
        'lower_ci': forecast['lower_ci'].values,
        'upper_ci': forecast['upper_ci'].values,
        'trend_model': best_model_type
    })
    
    result = pd.concat([result, forecast_df], ignore_index=True)
    result = result.sort_values(date_col)
    
    return result
