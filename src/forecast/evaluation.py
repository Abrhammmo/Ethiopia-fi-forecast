"""
Model Evaluation Module

Provides functions for evaluating forecast models including
cross-validation, backtesting, and metric calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
import warnings


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    rmse: float
    mape: float
    r_squared: float
    bias: float
    coverage: float  # Percentage of actuals within prediction intervals
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r_squared': self.r_squared,
            'bias': self.bias,
            'coverage': self.coverage
        }
    
    def summary(self) -> str:
        """Get summary string."""
        return (f"MAE: {self.mae:.3f}, RMSE: {self.rmse:.3f}, MAPE: {self.mape:.1f}%, "
                f"R²: {self.r_squared:.3f}, Coverage: {self.coverage:.1%}")


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_ci: Optional[np.ndarray] = None,
    upper_ci: Optional[np.ndarray] = None
) -> EvaluationMetrics:
    """
    Calculate evaluation metrics for forecasts.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        lower_ci: Lower confidence interval bounds (optional)
        upper_ci: Upper confidence interval bounds (optional)
        
    Returns:
        EvaluationMetrics object
    """
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        raise ValueError("No valid data points for evaluation")
    
    # Calculate error metrics
    errors = actual - predicted
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / actual) * 100 if np.all(actual > 0) else np.full(len(errors), np.nan)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.nanmean(pct_errors)
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Bias (mean error)
    bias = np.mean(errors)
    
    # Coverage (if CI provided)
    coverage = np.nan
    if lower_ci is not None and upper_ci is not None:
        lower_ci = lower_ci[mask]
        upper_ci = upper_ci[mask]
        within_interval = (actual >= lower_ci) & (actual <= upper_ci)
        coverage = np.mean(within_interval)
    
    return EvaluationMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        r_squared=r_squared,
        bias=bias,
        coverage=coverage
    )


def cross_validate_model(
    y: np.ndarray,
    model_type: str = 'linear',
    n_splits: int = 2,
    min_train_size: int = 3
) -> Dict:
    """
    Perform time series cross-validation.
    
    Args:
        y: Target variable
        model_type: 'linear' or 'log_linear'
        n_splits: Number of CV splits
        min_train_size: Minimum training size
        
    Returns:
        Dictionary with CV results
    """
    from .trend_models import fit_trend_model
    
    if len(y) < min_train_size + n_splits:
        warnings.warn("Not enough data for specified cross-validation")
        return {'error': 'Insufficient data'}
    
    t = np.arange(len(y))
    
    # Manual time series split (sklearn < 1.3 doesn't have min_train_size)
    n_samples = len(y)
    test_size = max(1, (n_samples - min_train_size) // (n_splits + 1))
    
    mae_scores = []
    rmse_scores = []
    mape_scores = []
    
    split_count = 0
    for test_start in range(min_train_size, n_samples - 1, test_size):
        if split_count >= n_splits:
            break
            
        test_end = min(test_start + test_size, n_samples)
        train_end = test_start
        
        if train_end < min_train_size:
            continue
            
        train_idx = slice(0, train_end)
        test_idx = slice(test_start, test_end)
        
        y_train, y_test = y[train_idx], y[test_idx]
        t_train = t[train_idx]
        
        try:
            model, _ = fit_trend_model(y_train, t_train, model_type)
            y_pred = model.predict(t[test_idx])
            
            metrics = calculate_metrics(y_test, y_pred)
            mae_scores.append(metrics.mae)
            rmse_scores.append(metrics.rmse)
            mape_scores.append(metrics.mape)
            split_count += 1
        except Exception as e:
            warnings.warn(f"CV fold failed: {e}")
            continue
    
    return {
        'model_type': model_type,
        'n_splits': len(mae_scores),
        'mae_mean': np.mean(mae_scores) if mae_scores else np.nan,
        'mae_std': np.std(mae_scores) if mae_scores else np.nan,
        'rmse_mean': np.mean(rmse_scores) if rmse_scores else np.nan,
        'mape_mean': np.mean(mape_scores) if mape_scores else np.nan,
        'all_mae_scores': mae_scores,
        'all_rmse_scores': rmse_scores,
        'all_mape_scores': mape_scores
    }


def backtest_forecasts(
    historical_dates: np.ndarray,
    historical_values: np.ndarray,
    forecast_horizons: List[int] = [1, 2, 3],
    model_type: str = 'linear'
) -> pd.DataFrame:
    """
    Backtest forecasts against historical data.
    
    Simulates making forecasts at various points in the past and
    evaluating against actual observed values.
    
    Args:
        historical_dates: Array of dates
        historical_values: Array of actual values
        forecast_horizons: Years ahead to forecast
        model_type: 'linear' or 'log_linear'
        
    Returns:
        DataFrame with backtest results
    """
    from .trend_models import fit_trend_model, predict_trend
    
    results = []
    n = len(historical_values)
    
    # Start from at least 5 years of data
    min_history = 5
    
    for i in range(min_history, n - max(forecast_horizons)):
        # Training data up to current point
        train_values = historical_values[:i]
        train_dates = historical_dates[:i]
        
        # Fit model
        t_train = np.arange(len(train_values))
        try:
            model, _ = fit_trend_model(train_values, t_train, model_type)
        except Exception as e:
            warnings.warn(f"Backtest iteration {i} failed: {e}")
            continue
        
        # Make forecasts for each horizon
        for horizon in forecast_horizons:
            if i + horizon >= n:
                continue
            
            # Forecast
            forecast = predict_trend(model, horizon, start_t=len(train_values))
            predicted = forecast['predicted'].iloc[-1]
            
            # Actual value at that horizon
            actual = historical_values[i + horizon]
            
            # Store results
            results.append({
                'forecast_date': train_dates[i-1] if i > 0 else train_dates[0],
                'horizon_years': horizon,
                'actual': actual,
                'predicted': predicted,
                'error': actual - predicted,
                'pct_error': (actual - predicted) / actual * 100 if actual > 0 else np.nan
            })
    
    return pd.DataFrame(results)


def evaluate_scenario_differences(
    actual: np.ndarray,
    baseline_forecast: np.ndarray,
    event_forecast: np.ndarray
) -> Dict:
    """
    Evaluate the additional impact of events vs baseline.
    
    Args:
        actual: Actual observed values
        baseline_forecast: Forecast without events
        event_forecast: Forecast with events
        
    Returns:
        Dictionary with impact evaluation metrics
    """
    # Calculate differences
    event_effect = event_forecast - baseline_forecast
    baseline_error = baseline_forecast - actual
    event_error = event_forecast - actual
    
    return {
        'mean_event_effect': np.mean(event_effect),
        'max_event_effect': np.max(event_effect),
        'baseline_mae': np.mean(np.abs(baseline_error)),
        'event_mae': np.mean(np.abs(event_error)),
        'improvement': np.mean(np.abs(baseline_error)) - np.mean(np.abs(event_error)),
        'improvement_pct': ((np.mean(np.abs(baseline_error)) - np.mean(np.abs(event_error))) / 
                           np.mean(np.abs(baseline_error)) * 100) if np.mean(np.abs(baseline_error)) > 0 else 0
    }


def print_evaluation_report(
    metrics: EvaluationMetrics,
    model_name: str = "Model"
) -> str:
    """Generate a formatted evaluation report."""
    report = f"""
{'='*60}
{model_name} Evaluation Report
{'='*60}
Mean Absolute Error (MAE):        {metrics.mae:.3f}
Root Mean Square Error (RMSE):    {metrics.rmse:.3f}
Mean Absolute % Error (MAPE):     {metrics.mape:.1f}%
R-squared:                        {metrics.r_squared:.3f}
Bias (Mean Error):                {metrics.bias:.3f}
Coverage (within CI):             {metrics.coverage:.1%} {'✓' if metrics.coverage >= 0.9 else '⚠'}
{'='*60}
"""
    return report


def compare_forecast_models(
    actual: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    ci_bounds: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
) -> pd.DataFrame:
    """
    Compare multiple forecast models.
    
    Args:
        actual: Actual values
        forecasts: Dictionary of model_name -> predicted values
        ci_bounds: Optional confidence interval bounds
        
    Returns:
        DataFrame with comparison metrics
    """
    records = []
    
    for model_name, predicted in forecasts.items():
        lower = None
        upper = None
        if ci_bounds and model_name in ci_bounds:
            lower, upper = ci_bounds[model_name]
        
        metrics = calculate_metrics(actual, predicted, lower, upper)
        
        record = {
            'model': model_name,
            'MAE': metrics.mae,
            'RMSE': metrics.rmse,
            'MAPE': metrics.mape,
            'R²': metrics.r_squared,
            'Bias': metrics.bias,
            'Coverage': metrics.coverage
        }
        records.append(record)
    
    return pd.DataFrame(records).sort_values('RMSE')
