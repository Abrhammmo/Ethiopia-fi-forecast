"""
Event Impact Modeling Module for Ethiopia Financial Inclusion Forecast

This module provides reusable functions for modeling how events (policies, 
product launches, infrastructure investments) affect financial inclusion indicators.

Key Components:
- Event-Indicator Association Matrix
- Temporal impact functions (immediate, gradual, sustained)
- Impact aggregation from multiple events
- Model validation against historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


# ============================================================================
# Data Classes for Impact Modeling
# ============================================================================

@dataclass
class ImpactLink:
    """Represents a single impact link between an event and an indicator."""
    record_id: str
    parent_id: str  # Event ID
    event_name: str
    indicator_code: str
    indicator_name: str
    pillar: str
    impact_estimate: float  # Magnitude of effect
    impact_direction: str  # 'increase' or 'decrease'
    relationship_type: str  # 'direct', 'indirect', 'enabling'
    impact_magnitude: str  # 'high', 'medium', 'low'
    lag_months: int  # Time delay before effect materializes
    evidence_basis: str  # 'empirical', 'literature', 'theoretical'
    comparable_country: Optional[str]
    observation_date: str
    unit: str
    
    @property
    def is_positive(self) -> bool:
        """Returns True if impact increases the indicator."""
        return self.impact_direction == 'increase'
    
    @property
    def confidence_score(self) -> float:
        """Returns a numeric confidence score based on evidence basis."""
        confidence_map = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        return confidence_map.get(self.impact_magnitude, 0.5)
    
    @property
    def absolute_impact(self) -> float:
        """Returns the absolute value of impact."""
        return abs(self.impact_estimate) if not pd.isna(self.impact_estimate) else 0.0


@dataclass
class Event:
    """Represents an event that can impact financial inclusion indicators."""
    record_id: str
    event_name: str
    event_type: str  # product_launch, market_entry, policy, infrastructure, etc.
    observation_date: str
    category: str
    description: str
    
    def to_date(self) -> datetime:
        """Convert observation_date to datetime."""
        try:
            return pd.to_datetime(self.observation_date)
        except:
            return pd.NaT


@dataclass
class TemporalImpactFunction:
    """Defines how an event's effect unfolds over time."""
    event_id: str
    indicator_code: str
    total_effect: float  # Total effect at maturity
    lag_months: int  # Delay before effect starts
    ramp_months: int  # Months to reach full effect (0 = immediate)
    decay_rate: Optional[float] = None  # For effects that fade over time
    peak_months: Optional[int] = None  # When effect peaks (if not sustained)
    
    def effect_at_month(self, months_since_event: int) -> float:
        """
        Calculate the effect at a given month since the event.
        
        Args:
            months_since_event: Number of months since the event occurred
            
        Returns:
            Effect magnitude at that time point
        """
        if months_since_event < self.lag_months:
            return 0.0
        
        adjusted_time = months_since_event - self.lag_months
        
        # Immediate effect
        if self.ramp_months == 0:
            return self.total_effect
        
        # Gradual ramp-up
        if self.peak_months is None or adjusted_time <= self.peak_months:
            ramp_factor = min(adjusted_time / self.ramp_months, 1.0)
            return self.total_effect * ramp_factor
        
        # Peak and decay
        if self.decay_rate is not None:
            decay_time = adjusted_time - self.peak_months
            peak_effect = self.total_effect * (self.peak_months / self.ramp_months)
            return peak_effect * np.exp(-self.decay_rate * decay_time)
        
        # Sustained at peak
        return self.total_effect


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_unified_data(filepath: str) -> pd.DataFrame:
    """
    Load the unified Ethiopia financial inclusion dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with all records
    """
    try:
        df = pd.read_csv(filepath)
        df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
        return df
    except Exception as e:
        raise ValueError(f"Failed to load unified data from {filepath}: {e}")


def load_impact_links(filepath: str) -> pd.DataFrame:
    """
    Load impact links data from CSV.
    
    Args:
        filepath: Path to the impact links CSV file
        
    Returns:
        DataFrame with impact link records
    """
    try:
        df = pd.read_csv(filepath)
        df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
        return df
    except Exception as e:
        raise ValueError(f"Failed to load impact links from {filepath}: {e}")


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract event records from unified data.
    
    Args:
        df: Unified DataFrame
        
    Returns:
        DataFrame containing only event records
    """
    events = df[df['record_type'] == 'event'].copy()
    events['observation_date'] = pd.to_datetime(events['observation_date'], errors='coerce')
    return events


def extract_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract observation records from unified data.
    
    Args:
        df: Unified DataFrame
        
    Returns:
        DataFrame containing only observation records
    """
    observations = df[df['record_type'] == 'observation'].copy()
    observations['observation_date'] = pd.to_datetime(observations['observation_date'], errors='coerce')
    return observations


def extract_impact_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract impact link records from unified data.
    
    Args:
        df: Unified DataFrame
        
    Returns:
        DataFrame containing only impact link records
    """
    impact_links = df[df['record_type'] == 'impact_link'].copy()
    impact_links['observation_date'] = pd.to_datetime(impact_links['observation_date'], errors='coerce')
    return impact_links


def join_events_with_impacts(events: pd.DataFrame, impact_links: pd.DataFrame) -> pd.DataFrame:
    """
    Join events with their impact links using parent_id.
    
    Args:
        events: DataFrame of events
        impact_links: DataFrame of impact links
        
    Returns:
        Merged DataFrame with event details and impact information
    """
    if 'parent_id' not in impact_links.columns:
        impact_links = impact_links.rename(columns={'record_id': 'parent_id'})
    
    # Get event details for joining
    event_details = events[['record_id', 'indicator', 'record_type', 'value_text']].copy()
    event_details = event_details.rename(columns={
        'record_id': 'parent_id',
        'indicator': 'event_name',
        'value_text': 'event_value'
    })
    
    merged = impact_links.merge(
        event_details,
        on='parent_id',
        how='left'
    )
    
    return merged


# ============================================================================
# Event-Indicator Association Matrix
# ============================================================================

def create_association_matrix(impact_links: pd.DataFrame, 
                               indicator_codes: Optional[List[str]] = None,
                               event_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create an event-indicator association matrix.
    
    Args:
        impact_links: DataFrame of impact links
        indicator_codes: List of indicator codes to include (all if None)
        event_ids: List of event IDs to include (all if None)
        
    Returns:
        Matrix with events as rows, indicators as columns, and impact estimates as values
    """
    # Filter if specified
    if event_ids:
        impact_links = impact_links[impact_links['parent_id'].isin(event_ids)]
    if indicator_codes:
        impact_links = impact_links[impact_links['related_indicator'].isin(indicator_codes)]
    
    if impact_links.empty:
        warnings.warn("No impact links found after filtering")
        return pd.DataFrame()
    
    # Create the matrix
    # Use pivot_table to handle potential duplicates
    matrix = impact_links.pivot_table(
        index='parent_id',
        columns='related_indicator',
        values='impact_estimate',
        aggfunc='mean'  # Average if duplicates exist
    )
    
    # Fill NaN with 0 (no direct impact)
    matrix = matrix.fillna(0)
    
    return matrix


def create_detailed_association_matrix(impact_links: pd.DataFrame) -> pd.DataFrame:
    """
    Create a detailed association matrix with metadata about each impact.
    
    Returns:
        DataFrame with columns: event_id, event_name, indicator_code, indicator_name,
        impact_estimate, lag_months, relationship_type, confidence, etc.
    """
    if impact_links.empty:
        return pd.DataFrame()
    
    detailed = impact_links[[
        'parent_id', 'related_indicator', 'indicator', 'pillar',
        'impact_estimate', 'impact_direction', 'relationship_type',
        'impact_magnitude', 'lag_months', 'evidence_basis', 'comparable_country'
    ]].copy()
    
    detailed = detailed.rename(columns={
        'parent_id': 'event_id',
        'related_indicator': 'indicator_code',
        'indicator': 'indicator_name',
        'impact_magnitude': 'confidence'
    })
    
    # Add numeric confidence
    confidence_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
    detailed['confidence_score'] = detailed['confidence'].map(confidence_map)
    
    return detailed


def create_impact_summary(impact_links: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary showing total impact per event and indicator.
    
    Args:
        impact_links: DataFrame of impact links
        
    Returns:
        Summary DataFrame with aggregated impacts
    """
    if impact_links.empty:
        return pd.DataFrame()
    
    summary = impact_links.groupby(['parent_id', 'related_indicator']).agg({
        'impact_estimate': ['sum', 'mean', 'count'],
        'lag_months': 'mean',
        'confidence': lambda x: x.mode().iloc[0] if len(x) > 0 else 'medium'
    }).reset_index()
    
    summary.columns = [
        'event_id', 'indicator_code',
        'total_impact', 'avg_impact', 'impact_count',
        'avg_lag_months', 'confidence_level'
    ]
    
    return summary


# ============================================================================
# Temporal Impact Modeling
# ============================================================================

def build_temporal_impact_function(impact_link: pd.Series,
                                    ramp_months: int = 12,
                                    decay_rate: Optional[float] = None,
                                    peak_months: Optional[int] = None) -> TemporalImpactFunction:
    """
    Build a temporal impact function from an impact link.
    
    Args:
        impact_link: Row from impact links DataFrame
        ramp_months: Months to reach full effect (default 12)
        decay_rate: Rate of decay for effects that fade (optional)
        peak_months: When effect peaks if not sustained (optional)
        
    Returns:
        TemporalImpactFunction object
    """
    # Determine total effect with sign
    base_impact = impact_link.get('impact_estimate', 0)
    direction = impact_link.get('impact_direction', 'increase')
    
    if pd.isna(base_impact):
        total_effect = 0
    else:
        total_effect = base_impact if direction == 'increase' else -base_impact
    
    lag = int(impact_link.get('lag_months', 0)) if not pd.isna(impact_link.get('lag_months')) else 0
    
    return TemporalImpactFunction(
        event_id=impact_link.get('parent_id', ''),
        indicator_code=impact_link.get('related_indicator', ''),
        total_effect=total_effect,
        lag_months=lag,
        ramp_months=ramp_months,
        decay_rate=decay_rate,
        peak_months=peak_months
    )


def calculate_aggregated_impact(impact_functions: List[TemporalImpactFunction],
                                 months_since_event: int) -> float:
    """
    Calculate the aggregated impact from multiple events at a given time point.
    
    Args:
        impact_functions: List of TemporalImpactFunction objects
        months_since_event: Number of months since the events occurred
        
    Returns:
        Aggregated impact magnitude
    """
    total_impact = 0.0
    for func in impact_functions:
        total_impact += func.effect_at_month(months_since_event)
    return total_impact


def simulate_indicator_trajectory(
    baseline_value: float,
    impact_functions: List[TemporalImpactFunction],
    start_date: datetime,
    months: int = 60,
    seasonal_factor: float = 1.0
) -> pd.DataFrame:
    """
    Simulate an indicator's trajectory over time with event impacts.
    
    Args:
        baseline_value: Starting value of the indicator
        impact_functions: List of impact functions to apply
        start_date: Simulation start date
        months: Number of months to simulate
        seasonal_factor: Optional seasonal adjustment (1.0 = no adjustment)
        
    Returns:
        DataFrame with monthly projections
    """
    dates = pd.date_range(start=start_date, periods=months, freq='M')
    values = []
    cumulative_impact = []
    
    for date in dates:
        months_since_ref = (date.year - start_date.year) * 12 + (date.month - start_date.month)
        
        # Calculate aggregated impact from all events
        impact = calculate_aggregated_impact(impact_functions, months_since_ref)
        
        # Apply seasonal factor
        adjusted_value = baseline_value * (1 + impact / 100) * seasonal_factor
        values.append(adjusted_value)
        cumulative_impact.append(impact)
    
    return pd.DataFrame({
        'date': dates,
        'projected_value': values,
        'cumulative_impact': cumulative_impact
    })


# ============================================================================
# Model Validation
# ============================================================================

def validate_against_historical(predicted: pd.DataFrame,
                                 observed: pd.DataFrame,
                                 indicator_code: str,
                                 tolerance: float = 0.1) -> Dict:
    """
    Validate model predictions against historical observations.
    
    Args:
        predicted: DataFrame with predicted values
        observed: DataFrame with observed values
        indicator_code: The indicator code to validate
        tolerance: Acceptable error tolerance (default 10%)
        
    Returns:
        Dictionary with validation metrics
    """
    # Filter for the specific indicator
    obs_filtered = observed[observed['indicator_code'] == indicator_code].copy()
    
    if obs_filtered.empty:
        return {'error': f'No observations found for indicator: {indicator_code}'}
    
    # Merge predicted and observed by nearest date
    obs_filtered = obs_filtered.set_index('observation_date').sort_index()
    predicted = predicted.set_index('date').sort_index()
    
    # Calculate metrics
    errors = []
    for obs_date, obs_row in obs_filtered.iterrows():
        # Find nearest predicted date
        idx = predicted.index.get_indexer([obs_date], method='nearest')[0]
        pred_value = predicted.iloc[idx]['projected_value']
        actual_value = obs_row['value_numeric']
        
        if actual_value > 0:
            relative_error = abs(pred_value - actual_value) / actual_value
            errors.append(relative_error)
    
    if not errors:
        return {'error': 'No matching dates found'}
    
    avg_error = np.mean(errors)
    within_tolerance = sum(1 for e in errors if e <= tolerance) / len(errors)
    
    return {
        'indicator': indicator_code,
        'mean_relative_error': avg_error,
        'within_tolerance_pct': within_tolerance * 100,
        'observations_compared': len(errors),
        'predictions_within_tolerance': within_tolerance >= 0.7,
        'validation_passed': avg_error < tolerance
    }


def validate_telebirr_impact(impact_links: pd.DataFrame,
                              observations: pd.DataFrame) -> Dict:
    """
    Specifically validate Telebirr impact model against historical data.
    
    Telebirr launched in May 2021:
    - Mobile money accounts went from 4.7% (2021) to 9.45% (2024)
    - This represents ~4.75pp increase over ~3 years
    
    Args:
        impact_links: Impact links DataFrame
        observations: Observations DataFrame
        
    Returns:
        Validation results dictionary
    """
    # Get Telebirr impact on mobile money accounts
    telebirr_link = impact_links[
        (impact_links['parent_id'] == 'EVT_0001') &
        (impact_links['related_indicator'] == 'ACC_MM_ACCOUNT')
    ]
    
    if telebirr_link.empty:
        return {'error': 'No Telebirr impact link found for mobile money accounts'}
    
    # Get actual observations
    mm_account_obs = observations[
        observations['indicator_code'] == 'ACC_MM_ACCOUNT'
    ].sort_values('observation_date')
    
    if mm_account_obs.empty:
        return {'error': 'No mobile money account observations found'}
    
    # Calculate actual change
    latest = mm_account_obs.iloc[-1]['value_numeric']
    earliest = mm_account_obs.iloc[0]['value_numeric']
    actual_change = latest - earliest
    
    # Get estimated impact
    estimated_impact = telebirr_link.iloc[0]['impact_estimate']
    
    # Calculate validation metrics
    if estimated_impact > 0:
        estimated_total_effect = estimated_impact  # 5pp increase
    else:
        estimated_total_effect = 0
    
    relative_error = abs(actual_change - estimated_total_effect) / actual_change if actual_change > 0 else float('inf')
    
    return {
        'event': 'Telebirr Launch',
        'indicator': 'Mobile Money Account Rate',
        'launch_date': '2021-05-17',
        'observed_change_2021_to_2024': actual_change,
        'estimated_impact': estimated_impact,
        'estimated_total_effect': estimated_total_effect,
        'relative_error': relative_error,
        'validation_passed': relative_error < 0.3,  # 30% tolerance
        'notes': 'Telebirr showed ~4.75pp increase, model estimates 5pp'
    }


# ============================================================================
# Impact Estimation Helpers
# ============================================================================

def get_key_indicators() -> Dict[str, str]:
    """
    Return the key indicators for financial inclusion modeling.
    
    Returns:
        Dictionary mapping indicator codes to names
    """
    return {
        'ACC_OWNERSHIP': 'Account Ownership Rate',
        'ACC_MM_ACCOUNT': 'Mobile Money Account Rate',
        'ACC_4G_COV': '4G Population Coverage',
        'USG_TELEBIRR_USERS': 'Telebirr Registered Users',
        'USG_MPESA_USERS': 'M-Pesa Registered Users',
        'USG_P2P_COUNT': 'P2P Transaction Count',
        'USG_DIGITAL_PAYMENT': 'Digital Payment Usage',
        'AFF_DATA_INCOME': 'Data Affordability Index',
        'GEN_GAP_ACC': 'Account Ownership Gender Gap',
        'GEN_MM_SHARE': 'Female Mobile Money Account Share'
    }


def get_key_events() -> Dict[str, Dict]:
    """
    Return key events with their details for modeling.
    
    Returns:
        Dictionary mapping event IDs to event details
    """
    return {
        'EVT_0001': {
            'name': 'Telebirr Launch',
            'type': 'product_launch',
            'date': '2021-05-17',
            'description': 'First major mobile money service in Ethiopia'
        },
        'EVT_0002': {
            'name': 'Safaricom Ethiopia Launch',
            'type': 'market_entry',
            'date': '2022-08-01',
            'description': 'End of state telecom monopoly, competition introduced'
        },
        'EVT_0003': {
            'name': 'M-Pesa Ethiopia Launch',
            'type': 'product_launch',
            'date': '2023-08-01',
            'description': 'Second mobile money entrant'
        },
        'EVT_0004': {
            'name': 'Fayda Digital ID Rollout',
            'type': 'infrastructure',
            'date': '2024-01-01',
            'description': 'National biometric digital ID system'
        },
        'EVT_0005': {
            'name': 'FX Liberalization',
            'type': 'policy',
            'date': '2024-07-29',
            'description': 'Foreign exchange market liberalization'
        },
        'EVT_0009': {
            'name': 'NFIS-II Strategy',
            'type': 'policy',
            'date': '2021-09-01',
            'description': '5-year national financial inclusion strategy'
        },
        'EVT_0011': {
            'name': 'Wage Digitization',
            'type': 'policy',
            'date': '2022-01-01',
            'description': 'Government payroll digitization initiative'
        }
    }


def estimate_impact_with_comparable_country(
    base_impact: float,
    comparable_country: str,
    adjustment_factor: float = 1.0
) -> float:
    """
    Adjust impact estimates based on comparable country evidence.
    
    Args:
        base_impact: Base impact estimate from literature
        comparable_country: Country name for comparison
        adjustment_factor: Adjustment for Ethiopia context (0.5-1.5 typical)
        
    Returns:
        Adjusted impact estimate
    """
    # Context-specific adjustments based on known evidence
    country_adjustments = {
        'Kenya': 1.0,  # Similar mobile money adoption trajectory
        'India': 0.7,  # Different context (larger, more mature market)
        'Rwanda': 0.8,  # Similar stage of development
        'Tanzania': 0.9,  # Similar regional context
        None: 1.0  # No adjustment if no comparable country
    }
    
    country_factor = country_adjustments.get(comparable_country, 0.8)
    return base_impact * adjustment_factor * country_factor


# ============================================================================
# Error Handling and Validation
# ============================================================================

def validate_impact_link(row: pd.Series) -> Tuple[bool, List[str]]:
    """
    Validate a single impact link record.
    
    Args:
        row: Impact link row
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['parent_id', 'related_indicator', 'impact_estimate']
    for field in required_fields:
        if field not in row or pd.isna(row.get(field)):
            errors.append(f"Missing required field: {field}")
    
    # Validate impact direction
    valid_directions = ['increase', 'decrease']
    if row.get('impact_direction') not in valid_directions:
        errors.append(f"Invalid impact_direction: {row.get('impact_direction')}")
    
    # Validate impact magnitude
    valid_magnitudes = ['high', 'medium', 'low']
    if row.get('impact_magnitude') not in valid_magnitudes:
        errors.append(f"Invalid impact_magnitude: {row.get('impact_magnitude')}")
    
    return len(errors) == 0, errors


def clean_impact_data(impact_links: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate impact link data.
    
    Args:
        impact_links: Raw impact links DataFrame
        
    Returns:
        Cleaned DataFrame with validation issues addressed
    """
    cleaned = impact_links.copy()
    
    # Drop rows with missing essential fields
    essential_fields = ['parent_id', 'related_indicator']
    cleaned = cleaned.dropna(subset=essential_fields)
    
    # Validate and log issues
    issues = []
    for idx, row in cleaned.iterrows():
        is_valid, errors = validate_impact_link(row)
        if not is_valid:
            issues.append({'row': idx, 'errors': errors})
    
    if issues:
        warnings.warn(f"Found {len(issues)} rows with validation issues")
        for issue in issues[:5]:  # Show first 5
            warnings.warn(f"  Row {issue['row']}: {issue['errors']}")
    
    return cleaned


# ============================================================================
# Export Functions
# ============================================================================

def export_association_matrix(matrix: pd.DataFrame, filepath: str) -> None:
    """
    Export association matrix to CSV.
    
    Args:
        matrix: Association matrix DataFrame
        filepath: Output file path
    """
    matrix.to_csv(filepath)
    print(f"Association matrix exported to {filepath}")


def export_impact_summary(summary: pd.DataFrame, filepath: str) -> None:
    """
    Export impact summary to CSV.
    
    Args:
        summary: Summary DataFrame
        filepath: Output file path
    """
    summary.to_csv(filepath, index=False)
    print(f"Impact summary exported to {filepath}")
