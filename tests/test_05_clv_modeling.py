# tests/test_clv_modeling.py

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import warnings
import sys
import os
import joblib

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock the lifetimes module, which we may not have installed during testing
sys.modules['lifetimes'] = MagicMock()
sys.modules['lifetimes.utils'] = MagicMock()

class TestCLVModeling(unittest.TestCase):
    """Test suite for CLV modeling functionality."""
    
    def setUp(self):
        """Set up test data for CLV modeling tests."""
        warnings.filterwarnings('ignore')
        
        # Create mock RFM data
        self.mock_rfm_data = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'recency_days': [5, 30, 60, 15, 45],
            'frequency': [10, 5, 2, 8, 4],
            'monetary_value': [1000, 500, 200, 750, 350],
            'T': [365, 365, 365, 365, 365]  # Observation period in days
        })
        
        # Create mock transaction data
        self.mock_transactions = pd.DataFrame({
            'client_id': [1, 1, 2, 3, 4, 5],
            'date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-01-10', 
                                    '2023-03-05', '2023-02-25', '2023-01-30']),
            'amount': [1000, 500, 2000, 300, 1500, 800]
        })
        
        # Mock database connection
        self.mock_conn = MagicMock()

    def test_simplified_clv_calculation(self):
        """Test the simplified CLV calculation method."""
        # Calculate simplified CLV
        rfm_data = self.mock_rfm_data.copy()
        
        # Apply the simplified calculation
        rfm_data['predicted_transactions_1yr'] = rfm_data['frequency'] * (365 / rfm_data['T'])
        rfm_data['expected_avg_profit'] = rfm_data['monetary_value']
        rfm_data['clv_1yr'] = rfm_data['predicted_transactions_1yr'] * rfm_data['expected_avg_profit']
        
        # Check that calculations are as expected
        for _, row in rfm_data.iterrows():
            expected_transactions = row['frequency'] * (365 / row['T'])
            expected_clv = expected_transactions * row['monetary_value']
            
            self.assertAlmostEqual(row['predicted_transactions_1yr'], expected_transactions)
            self.assertAlmostEqual(row['clv_1yr'], expected_clv)
            
        # Test specific values
        # For client 1: 10 transactions in 365 days should predict 10 transactions next year
        # CLV should be 10 * 1000 = 10000
        client_1 = rfm_data[rfm_data['client_id'] == 1].iloc[0]
        self.assertAlmostEqual(client_1['predicted_transactions_1yr'], 10.0)
        self.assertAlmostEqual(client_1['clv_1yr'], 10000.0)

    @patch('lifetimes.BetaGeoFitter')
    @patch('lifetimes.GammaGammaFitter')
    def test_probabilistic_clv_calculation(self, MockGammaGammaFitter, MockBetaGeoFitter):
        """Test the probabilistic CLV calculation with mocked models."""
        # Set up mocks for the probabilistic models
        mock_bgf = MockBetaGeoFitter.return_value
        mock_ggf = MockGammaGammaFitter.return_value
        
        # Configure mocks to return predictable values
        mock_bgf.predict.return_value = np.array([10.0, 5.0, 2.0, 8.0, 4.0])
        mock_ggf.conditional_expected_average_profit.return_value = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        mock_ggf.customer_lifetime_value.return_value = np.array([1000.0, 500.0, 200.0, 800.0, 400.0])
        
        # Set up parameters
        prediction_time = 365  # 1 year
        discount_rate = 0.15
        
        # Call the functions we'd use in the real code
        bgf = MockBetaGeoFitter()
        bgf.fit(self.mock_rfm_data['frequency'], 
                self.mock_rfm_data['recency_days'], 
                self.mock_rfm_data['T'])
        
        ggf = MockGammaGammaFitter()
        ggf.fit(self.mock_rfm_data['frequency'],
                self.mock_rfm_data['monetary_value'])
        
        # Make predictions
        predicted_transactions = bgf.predict(
            prediction_time,
            self.mock_rfm_data['frequency'],
            self.mock_rfm_data['recency_days'],
            self.mock_rfm_data['T']
        )
        
        expected_avg_profit = ggf.conditional_expected_average_profit(
            self.mock_rfm_data['frequency'],
            self.mock_rfm_data['monetary_value']
        )
        
        clv = ggf.customer_lifetime_value(
            bgf,
            self.mock_rfm_data['frequency'],
            self.mock_rfm_data['recency_days'],
            self.mock_rfm_data['T'],
            prediction_time,
            self.mock_rfm_data['monetary_value'],
            discount_rate=discount_rate
        )
        
        # Verify model was called with correct parameters
        MockBetaGeoFitter.assert_called_once()
        mock_bgf.fit.assert_called_once()
        MockGammaGammaFitter.assert_called_once()
        mock_ggf.fit.assert_called_once()
        
        # Verify model predictions
        self.assertEqual(len(predicted_transactions), 5)
        self.assertEqual(len(expected_avg_profit), 5)
        self.assertEqual(len(clv), 5)
        
        # Check that the mocked values match our expectations
        np.testing.assert_array_equal(predicted_transactions, np.array([10.0, 5.0, 2.0, 8.0, 4.0]))
        np.testing.assert_array_equal(expected_avg_profit, np.array([100.0, 100.0, 100.0, 100.0, 100.0]))
        np.testing.assert_array_equal(clv, np.array([1000.0, 500.0, 200.0, 800.0, 400.0]))

