"""
Tests for Task-2 Exploratory Data Analysis - Summary Statistics and Insights
"""
import pytest
import pandas as pd
import numpy as np
import os


class TestEDASummaryStatistics:
    """Test EDA summary statistics calculations"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    @pytest.fixture
    def obs_df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        df = pd.read_csv(path)
        return df[df['record_type'] == 'observation']
    
    def test_record_type_counts(self, df):
        """Test record type distribution"""
        record_types = df['record_type'].value_counts()
        assert 'observation' in record_types.index
        assert 'event' in record_types.index
        assert record_types['observation'] >= 30, "Should have at least 30 observations"
        assert record_types['event'] >= 10, "Should have at least 10 events"
    
    def test_pillar_distribution(self, obs_df):
        """Test pillar distribution in observations"""
        pillars = obs_df['pillar'].value_counts()
        assert 'ACCESS' in pillars.index
        assert 'USAGE' in pillars.index
        assert 'GENDER' in pillars.index
        assert 'AFFORDABILITY' in pillars.index
        # ACCESS and USAGE should be largest pillars
        assert pillars['ACCESS'] >= pillars['GENDER']
        assert pillars['USAGE'] >= pillars['GENDER']
    
    def test_source_type_distribution(self, df):
        """Test source type distribution"""
        source_types = df['source_type'].value_counts()
        assert 'survey' in source_types.index
        assert 'operator' in source_types.index
        assert 'regulator' in source_types.index
    
    def test_temporal_range(self, df):
        """Test temporal range of data"""
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        min_date = df['observation_date'].min()
        max_date = df['observation_date'].max()
        assert min_date.year <= 2025, "Data should start by 2025"
        assert max_date.year >= 2024, "Data should include 2024 or later"
    
    def test_confidence_distribution(self, df):
        """Test confidence level distribution"""
        confidence = df['confidence'].value_counts()
        assert 'high' in confidence.index
        high_ratio = confidence.get('high', 0) / len(df)
        assert high_ratio >= 0.7, "At least 70% should be high confidence"


class TestAccessAnalysis:
    """Test access-related statistics"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    def test_account_ownership_values(self, df):
        """Test account ownership values are reasonable"""
        acc = df[(df['indicator_code'] == 'ACC_OWNERSHIP') & 
                 (df['gender'] == 'all')]
        values = acc['value_numeric'].dropna().sort_values()
        assert len(values) >= 3, "Should have at least 3 account ownership observations"
        # Values should be increasing over time
        assert values.iloc[0] < values.iloc[-1], "Account ownership should increase over time"
        # Values should be between 0 and 100
        assert all((values >= 0) & (values <= 100)), "Account ownership should be 0-100%"
    
    def test_account_ownership_growth(self, df):
        """Test account ownership growth calculation"""
        acc = df[(df['indicator_code'] == 'ACC_OWNERSHIP') & 
                 (df['gender'] == 'all')].copy()
        acc = acc.sort_values('observation_date')
        values = acc['value_numeric'].values
        # Calculate growth periods
        growth_2014_2017 = values[1] - values[0] if len(values) > 1 else 0
        growth_2017_2021 = values[2] - values[1] if len(values) > 2 else 0
        growth_2021_2024 = values[3] - values[2] if len(values) > 3 else 0
        # Growth should be positive
        assert growth_2014_2017 > 0, "2014-2017 growth should be positive"
        assert growth_2017_2021 > 0, "2017-2021 growth should be positive"
        assert growth_2021_2024 > 0, "2021-2024 growth should be positive"
    
    def test_gender_gap_exists(self, df):
        """Test gender gap data exists"""
        gender_acc = df[(df['indicator_code'] == 'ACC_OWNERSHIP') & 
                        (df['gender'].isin(['male', 'female']))]
        assert len(gender_acc) >= 2, "Should have gender-disaggregated data"
        male_vals = gender_acc[gender_acc['gender'] == 'male']['value_numeric']
        female_vals = gender_acc[gender_acc['gender'] == 'female']['value_numeric']
        # Male should have higher ownership than female
        assert male_vals.mean() > female_vals.mean(), "Male ownership should exceed female"
    
    def test_mobile_money_account_growth(self, df):
        """Test mobile money account growth"""
        mm = df[df['indicator_code'] == 'ACC_MM_ACCOUNT'].copy()
        mm = mm.sort_values('observation_date')
        values = mm['value_numeric'].values
        assert len(values) >= 2, "Should have multiple mobile money observations"
        assert values[-1] > values[0], "Mobile money accounts should grow"


class TestUsageAnalysis:
    """Test usage-related statistics"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    def test_telebirr_users_exist(self, df):
        """Test Telebirr users data exists"""
        telebirr = df[df['indicator_code'] == 'USG_TELEBIRR_USERS']
        assert len(telebirr) >= 1, "Should have Telebirr users data"
        users = telebirr['value_numeric'].max()
        assert users > 1e7, "Telebirr should have over 10M users"
    
    def test_mpesa_data_exists(self, df):
        """Test M-Pesa data exists"""
        mpesa_users = df[df['indicator_code'] == 'USG_MPESA_USERS']
        mpesa_active = df[df['indicator_code'] == 'USG_MPESA_ACTIVE']
        assert len(mpesa_users) >= 1, "Should have M-Pesa users data"
        assert len(mpesa_active) >= 1, "Should have M-Pesa active data"
        # Active should be less than registered
        registered = mpesa_users['value_numeric'].values[0]
        active = mpesa_active['value_numeric'].values[0]
        assert active < registered, "Active users should be less than registered"
    
    def test_p2p_transactions_exist(self, df):
        """Test P2P transaction data exists"""
        p2p = df[df['indicator_code'] == 'USG_P2P_COUNT']
        assert len(p2p) >= 1, "Should have P2P transaction data"
        transactions = p2p['value_numeric'].max()
        assert transactions > 1e7, "Should have over 10M P2P transactions"
    
    def test_p2p_atm_crossover(self, df):
        """Test P2P > ATM crossover indicator exists"""
        crossover = df[df['indicator_code'] == 'USG_CROSSOVER']
        assert len(crossover) >= 1, "Should have crossover indicator"
        ratio = crossover['value_numeric'].values[0]
        assert ratio > 1.0, "P2P should exceed ATM (ratio > 1)"


class TestInfrastructureAnalysis:
    """Test infrastructure-related statistics"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    def test_4g_coverage_exists(self, df):
        """Test 4G coverage data exists"""
        coverage = df[df['indicator_code'] == 'ACC_4G_COV']
        assert len(coverage) >= 1, "Should have 4G coverage data"
        values = coverage['value_numeric'].values
        assert all((values >= 0) & (values <= 100)), "4G coverage should be 0-100%"
    
    def test_mobile_penetration_exists(self, df):
        """Test mobile penetration data exists"""
        mobile = df[df['indicator_code'] == 'ACC_MOBILE_PEN']
        assert len(mobile) >= 1, "Should have mobile penetration data"
        value = mobile['value_numeric'].values[0]
        assert value > 0, "Mobile penetration should be positive"
    
    def test_fayda_id_exists(self, df):
        """Test Fayda Digital ID data exists"""
        fayda = df[df['indicator_code'] == 'ACC_FAYDA']
        assert len(fayda) >= 1, "Should have Fayda ID data"
        enrolled = fayda['value_numeric'].max()
        assert enrolled > 1e6, "Should have over 1M Fayda enrollments"


class TestKeyInsightsValidation:
    """Test key insights assertions"""
    
    @pytest.fixture
    def df(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                           'ethiopia_fi_unified_data_enriched.csv')
        return pd.read_csv(path)
    
    def test_slowdown_insight(self, df):
        """Test the 2021-2024 slowdown insight"""
        acc = df[(df['indicator_code'] == 'ACC_OWNERSHIP') & 
                 (df['gender'] == 'all')].copy()
        acc = acc.sort_values('observation_date')
        values = acc['value_numeric'].values
        if len(values) >= 4:
            growth_2017_2021 = values[2] - values[1]
            growth_2021_2024 = values[3] - values[2]
            # 2021-2024 growth should be less than 2017-2021
            assert growth_2021_2024 < growth_2017_2021, "Should confirm slowdown"
    
    def test_usage_growth_insight(self, df):
        """Test that usage grows faster than access"""
        p2p = df[df['indicator_code'] == 'USG_P2P_COUNT'].sort_values('observation_date')
        acc = df[(df['indicator_code'] == 'ACC_OWNERSHIP') & 
                 (df['gender'] == 'all')].sort_values('observation_date')
        if len(p2p) >= 2 and len(acc) >= 4:
            p2p_growth = (p2p['value_numeric'].iloc[-1] / p2p['value_numeric'].iloc[0] - 1)
            acc_growth = (acc['value_numeric'].iloc[-1] / acc['value_numeric'].iloc[0] - 1)
            assert p2p_growth > acc_growth, "P2P growth should exceed access growth"
    
    def test_g2p_opportunity_exists(self, df):
        """Test G2P digitization data exists"""
        g2p = df[df['indicator_code'] == 'USG_G2P_DIGITIZED']
        if len(g2p) > 0:
            value = g2p['value_numeric'].values[0]
            assert value < 50, "G2P should be under 50% (confirming opportunity)"
    
    def test_gender_gap_insight(self, df):
        """Test gender gap is positive but narrowing"""
        gender = df[(df['indicator_code'] == 'GEN_GAP_ACC')].sort_values('observation_date')
        if len(gender) >= 2:
            gap_early = gender['value_numeric'].iloc[0]
            gap_late = gender['value_numeric'].iloc[-1]
            assert gap_early > 0, "Gender gap should be positive"
            assert gap_late < gap_early or gap_late == gap_early, "Gender gap should not widen"
    
    def test_activity_rate_reasonable(self, df):
        """Test activity rate is reasonable (0-100%)"""
        activity = df[df['indicator_code'] == 'USG_ACTIVE_RATE']
        if len(activity) > 0:
            rate = activity['value_numeric'].values[0]
            assert 0 <= rate <= 100, "Activity rate should be 0-100%"
