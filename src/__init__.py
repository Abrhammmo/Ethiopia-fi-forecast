"""
Ethiopia Financial Inclusion Forecast - Impact Modeling Module

This package provides reusable functions for modeling how events (policies, 
product launches, infrastructure investments) affect financial inclusion indicators.

Modules:
- impact_modeling: Core impact modeling functions and classes

Key Components:
- Event-Indicator Association Matrix
- Temporal impact functions (immediate, gradual, sustained)
- Impact aggregation from multiple events
- Model validation against historical data
"""

from .impact_modeling import (
    # Data Classes
    ImpactLink,
    Event,
    TemporalImpactFunction,
    
    # Data Loading Functions
    load_unified_data,
    load_impact_links,
    extract_events,
    extract_observations,
    extract_impact_links,
    join_events_with_impacts,
    
    # Association Matrix Functions
    create_association_matrix,
    create_detailed_association_matrix,
    create_impact_summary,
    
    # Temporal Impact Modeling
    build_temporal_impact_function,
    calculate_aggregated_impact,
    simulate_indicator_trajectory,
    
    # Model Validation
    validate_against_historical,
    validate_telebirr_impact,
    
    # Helper Functions
    get_key_indicators,
    get_key_events,
    estimate_impact_with_comparable_country,
    
    # Error Handling
    validate_impact_link,
    clean_impact_data,
    
    # Export Functions
    export_association_matrix,
    export_impact_summary
)

__version__ = '1.0.0'
__author__ = 'Ethiopia FI Forecast Team'

__all__ = [
    'ImpactLink',
    'Event', 
    'TemporalImpactFunction',
    'load_unified_data',
    'load_impact_links',
    'extract_events',
    'extract_observations',
    'extract_impact_links',
    'join_events_with_impacts',
    'create_association_matrix',
    'create_detailed_association_matrix',
    'create_impact_summary',
    'build_temporal_impact_function',
    'calculate_aggregated_impact',
    'simulate_indicator_trajectory',
    'validate_against_historical',
    'validate_telebirr_impact',
    'get_key_indicators',
    'get_key_events',
    'estimate_impact_with_comparable_country',
    'validate_impact_link',
    'clean_impact_data',
    'export_association_matrix',
    'export_impact_summary'
]
