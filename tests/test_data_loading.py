"""
Tests for Task-1 Data Enrichment - Data Loading and Validation
"""
import pytest
import pandas as pd
import os


class TestDataLoading:
    """Test data file loading and basic structure"""
    
    @pytest.fixture
    def raw_data_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    @pytest.fixture
    def processed_data_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    def test_raw_data_exists(self, raw_data_path):
        """Test that raw data files exist"""
        assert os.path.exists(raw_data_path), "Raw data directory should exist"
        assert os.path.exists(os.path.join(raw_data_path, 'ethiopia_fi_unified_data.csv'))
    
    def test_processed_data_exists(self, processed_data_path):
        """Test that processed data files exist"""
        assert os.path.exists(processed_data_path), "Processed data directory should exist"
        assert os.path.exists(os.path.join(processed_data_path, 'ethiopia_fi_unified_data_enriched.csv'))
        assert os.path.exists(os.path.join(processed_data_path, 'impact_links_enriched.csv'))
    
    def test_load_unified_data(self, processed_data_path):
        """Test loading unified dataset"""
        df = pd.read_csv(os.path.join(processed_data_path, 'ethiopia_fi_unified_data_enriched.csv'))
        assert len(df) > 0, "Dataset should not be empty"
        assert 'record_id' in df.columns
        assert 'record_type' in df.columns
    
    def test_load_impact_links(self, processed_data_path):
        """Test loading impact links dataset"""
        impact_df = pd.read_csv(os.path.join(processed_data_path, 'impact_links_enriched.csv'))
        assert len(impact_df) > 0, "Impact links should not be empty"
        assert 'parent_id' in impact_df.columns
        assert 'related_indicator' in impact_df.columns


class TestUnifiedSchema:
    """Test unified schema validation"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    @pytest.fixture
    def impact_df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'impact_links_enriched.csv')
        return pd.read_csv(path)
    
    def test_record_types_exist(self, df):
        """Test that all required record types exist"""
        record_types = df['record_type'].unique()
        assert 'observation' in record_types
        assert 'event' in record_types
        assert 'target' in record_types or 'impact_link' in record_types
    
    def test_pillars_exist(self, df):
        """Test that all required pillars exist in observations"""
        obs_df = df[df['record_type'] == 'observation']
        pillars = obs_df['pillar'].dropna().unique()
        assert 'ACCESS' in pillars
        assert 'USAGE' in pillars
        assert 'GENDER' in pillars
        assert 'AFFORDABILITY' in pillars
    
    def test_impact_links_have_required_fields(self, impact_df):
        """Test impact links have required fields"""
        required_fields = ['parent_id', 'related_indicator', 'impact_direction', 
                          'impact_magnitude', 'lag_months', 'evidence_basis']
        for field in required_fields:
            assert field in impact_df.columns, f"Impact links should have {field} field"
    
    def test_impact_magnitude_values(self, impact_df):
        """Test impact magnitude has valid values"""
        valid_magnitudes = ['low', 'medium', 'high']
        magnitudes = impact_df['impact_magnitude'].dropna().unique()
        for m in magnitudes:
            assert m in valid_magnitudes, f"Invalid impact magnitude: {m}"
    
    def test_evidence_basis_values(self, impact_df):
        """Test evidence basis has valid values"""
        valid_basis = ['literature', 'empirical', 'theoretical', 'expert']
        basis = impact_df['evidence_basis'].dropna().unique()
        for b in basis:
            assert b in valid_basis, f"Invalid evidence basis: {b}"


class TestDataQuality:
    """Test data quality metrics"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    def test_confidence_levels(self, df):
        """Test confidence level values are valid"""
        valid_confidence = ['high', 'medium', 'low']
        confidence = df['confidence'].dropna().unique()
        for c in confidence:
            assert c in valid_confidence, f"Invalid confidence level: {c}"
    
    def test_high_confidence_majority(self, df):
        """Test that majority of records have high confidence"""
        high_conf = (df['confidence'] == 'high').sum()
        total = df['confidence'].notna().sum()
        ratio = high_conf / total if total > 0 else 0
        assert ratio >= 0.7, "At least 70% of records should have high confidence"
    
    def test_no_duplicate_record_ids(self, df):
        """Test no duplicate record IDs"""
        assert df['record_id'].nunique() == len(df), "Record IDs should be unique"
    
    def test_required_fields_not_null(self, df):
        """Test required fields are not null"""
        required_fields = ['record_id', 'record_type', 'indicator', 'indicator_code']
        for field in required_fields:
            null_count = df[field].isna().sum()
            assert null_count == 0, f"{field} should not have null values"
    
    def test_dates_are_valid(self, df):
        """Test observation dates are valid"""
        df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
        assert df['observation_date'].notna().any(), "Should have valid dates"
        assert (df['observation_date'].dt.year >= 2014).all(), "Dates should be from 2014 or later"
