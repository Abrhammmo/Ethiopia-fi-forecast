"""
Forecasting Module for Ethiopia Financial Inclusion Forecast

This package provides forecasting functions for financial inclusion indicators
with trend analysis, event-augmented models, and scenario analysis.

Modules:
- trend_models: Linear and log-linear trend models
- event_augmented: Trend + event effects models
- scenarios: Scenario analysis with confidence intervals
"""

from .trend_models import (
    LinearTrendModel,
    LogLinearTrendModel,
    fit_trend_model,
    predict_trend
)

from .event_augmented import (
    EventAugmentedModel,
    fit_event_augmented_model,
    predict_with_events
)

from .scenarios import (
    ScenarioForecast,
    generate_scenario_forecasts,
    create_confidence_intervals
)

from .evaluation import (
    calculate_metrics,
    cross_validate_model,
    backtest_forecasts
)

__version__ = '1.0.0'
__author__ = 'Ethiopia FI Forecast Team'

__all__ = [
    'LinearTrendModel',
    'LogLinearTrendModel',
    'fit_trend_model',
    'predict_trend',
    'EventAugmentedModel',
    'fit_event_augmented_model',
    'predict_with_events',
    'ScenarioForecast',
    'generate_scenario_forecasts',
    'create_confidence_intervals',
    'calculate_metrics',
    'cross_validate_model',
    'backtest_forecasts'
]
