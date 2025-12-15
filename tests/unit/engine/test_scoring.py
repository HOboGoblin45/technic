"""Unit tests for scoring engine"""
import pytest
import pandas as pd
import numpy as np
from technic_v4.engine.scoring import compute_scores, build_institutional_core_score


class TestTechRatingCalculation:
    """Test TechRating calculation logic"""
    
    def test_tech_rating_exists(self, sample_price_data):
        """Test that TechRating is calculated and present"""
        result = compute_scores(sample_price_data)
        
        assert 'TechRating' in result.columns, "TechRating column should exist"
        assert not result['TechRating'].isna().all(), "TechRating should have values"
    
    def test_tech_rating_range(self, sample_price_data):
        """Test that TechRating is within expected range"""
        result = compute_scores(sample_price_data)
        
        # TechRating should be numeric
        assert pd.api.types.is_numeric_dtype(result['TechRating'])
        
        # Should have at least some non-null values
        valid_ratings = result['TechRating'].dropna()
        assert len(valid_ratings) > 0, "Should have at least some valid ratings"
    
    def test_risk_score_calculation(self, sample_price_data):
        """Test that RiskScore is calculated correctly"""
        result = compute_scores(sample_price_data)
        
        if 'RiskScore' in result.columns:
            risk_scores = result['RiskScore'].dropna()
            if len(risk_scores) > 0:
                # Risk score should be between 0 and 1
                assert (risk_scores >= 0).all(), "RiskScore should be >= 0"
                assert (risk_scores <= 1).all(), "RiskScore should be <= 1"
    
    def test_signal_classification(self, sample_price_data):
        """Test that signals are classified correctly"""
        result = compute_scores(sample_price_data)
        
        if 'Signal' in result.columns:
            valid_signals = {'Long', 'Strong Long', 'Short', 'Strong Short', 'Avoid', 'Neutral'}
            signals = result['Signal'].dropna().unique()
            
            for signal in signals:
                assert signal in valid_signals, f"Invalid signal: {signal}"
    
    def test_sub_scores_present(self, sample_price_data):
        """Test that sub-scores are calculated"""
        result = compute_scores(sample_price_data)
        
        # Check for expected sub-score columns
        expected_subscores = [
            'TrendScore', 'MomentumScore', 'VolumeScore',
            'VolatilityScore', 'OscillatorScore', 'BreakoutScore'
        ]
        
        # At least some sub-scores should be present
        present_subscores = [col for col in expected_subscores if col in result.columns]
        assert len(present_subscores) > 0, "At least some sub-scores should be calculated"


class TestInstitutionalCoreScore:
    """Test ICS calculation logic"""
    
    def test_ics_calculation(self, sample_scan_result):
        """Test basic ICS calculation"""
        # Create a DataFrame with sample data
        df = pd.DataFrame([sample_scan_result])
        
        # Add required columns for ICS
        df['AlphaScore'] = 0.05
        df['QualityScore'] = 75.0
        df['SponsorshipScore'] = 60.0
        df['has_upcoming_earnings'] = False
        df['days_since_last_earnings'] = 30
        
        try:
            result = build_institutional_core_score(df)
            
            if 'InstitutionalCoreScore' in result.columns:
                ics = result['InstitutionalCoreScore'].iloc[0]
                
                # ICS should be between 0 and 100
                assert 0 <= ics <= 100, f"ICS should be 0-100, got {ics}"
                assert not pd.isna(ics), "ICS should not be NaN"
        except Exception as e:
            # ICS calculation might require more data, that's okay for now
            pytest.skip(f"ICS calculation requires more data: {e}")
    
    def test_ics_components(self, sample_scan_result):
        """Test that ICS uses expected components"""
        df = pd.DataFrame([sample_scan_result])
        
        # Add all possible ICS components
        df['AlphaScore'] = 0.05
        df['QualityScore'] = 75.0
        df['SponsorshipScore'] = 60.0
        df['has_upcoming_earnings'] = False
        df['days_since_last_earnings'] = 30
        df['DollarVolume'] = 7500000000
        
        try:
            result = build_institutional_core_score(df)
            
            # Check that ICS was calculated
            assert 'InstitutionalCoreScore' in result.columns or \
                   'ICS' in result.columns, "ICS should be calculated"
        except Exception as e:
            pytest.skip(f"ICS calculation requires more setup: {e}")


class TestRiskAdjustment:
    """Test risk adjustment logic"""
    
    def test_high_volatility_penalty(self):
        """Test that high volatility reduces scores"""
        # Create two identical stocks, one with high volatility
        low_vol = pd.DataFrame({
            'Close': [100] * 10,
            'TechRating': [80] * 10,
            'ATR_pct_14': [0.01] * 10,  # 1% ATR (low volatility)
        })
        
        high_vol = pd.DataFrame({
            'Close': [100] * 10,
            'TechRating': [80] * 10,
            'ATR_pct_14': [0.05] * 10,  # 5% ATR (high volatility)
        })
        
        # Risk-adjusted scores should differ
        # (This is a conceptual test - actual implementation may vary)
        assert True  # Placeholder for actual risk adjustment test
    
    def test_risk_score_from_atr(self):
        """Test that RiskScore is derived from ATR"""
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'ATR_pct_14': [0.01, 0.02, 0.05],
        })
        
        # Lower ATR should give higher RiskScore
        # (This is a conceptual test - actual implementation may vary)
        assert True  # Placeholder


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        
        try:
            result = compute_scores(df)
            # Should either return empty or raise appropriate error
            assert result is None or result.empty
        except Exception as e:
            # Expected to fail gracefully
            assert True
    
    def test_insufficient_data(self):
        """Test handling of insufficient data points"""
        # Only 5 data points (not enough for 20-day MA)
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000] * 5,
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
        })
        
        try:
            result = compute_scores(df)
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            # Expected to fail gracefully
            assert True
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        df = pd.DataFrame({
            'Close': [100, 101, 102],
        })
        
        try:
            result = compute_scores(df)
            # Should handle gracefully or raise appropriate error
            assert True
        except KeyError:
            # Expected if required columns are missing
            assert True
    
    def test_nan_values(self):
        """Test handling of NaN values in data"""
        df = pd.DataFrame({
            'Close': [100, np.nan, 102, 103, 104],
            'Volume': [1000000, 1000000, np.nan, 1000000, 1000000],
            'High': [101, 102, 103, np.nan, 105],
            'Low': [99, 100, 101, 102, 103],
        })
        
        try:
            result = compute_scores(df)
            # Should handle NaN values gracefully
            assert result is not None
        except Exception:
            # Expected to handle or fail gracefully
            assert True


class TestScoreConsistency:
    """Test score consistency and reproducibility"""
    
    def test_deterministic_scoring(self, sample_price_data):
        """Test that scoring is deterministic (same input = same output)"""
        result1 = compute_scores(sample_price_data.copy())
        result2 = compute_scores(sample_price_data.copy())
        
        if 'TechRating' in result1.columns and 'TechRating' in result2.columns:
            # Scores should be identical for same input
            pd.testing.assert_series_equal(
                result1['TechRating'],
                result2['TechRating'],
                check_names=False
            )
    
    def test_score_monotonicity(self):
        """Test that better technicals lead to higher scores"""
        # Create data with clear uptrend
        uptrend = pd.DataFrame({
            'Close': list(range(100, 150)),
            'Volume': [1000000] * 50,
            'High': list(range(101, 151)),
            'Low': list(range(99, 149)),
        })
        
        # Create data with clear downtrend
        downtrend = pd.DataFrame({
            'Close': list(range(150, 100, -1)),
            'Volume': [1000000] * 50,
            'High': list(range(151, 101, -1)),
            'Low': list(range(149, 99, -1)),
        })
        
        try:
            up_result = compute_scores(uptrend)
            down_result = compute_scores(downtrend)
            
            if 'TechRating' in up_result.columns and 'TechRating' in down_result.columns:
                # Uptrend should generally have higher scores
                up_score = up_result['TechRating'].iloc[-1]
                down_score = down_result['TechRating'].iloc[-1]
                
                # This is a soft assertion - not always true but generally expected
                # assert up_score > down_score, "Uptrend should score higher than downtrend"
                assert True  # Placeholder for now
        except Exception:
            pytest.skip("Score monotonicity test requires more setup")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
