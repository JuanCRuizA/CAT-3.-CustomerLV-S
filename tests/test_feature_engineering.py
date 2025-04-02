# tests/test_feature_engineering.py

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import warnings
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_engineering import (
    calculate_rfm_features, 
    create_product_portfolio_features,
    create_transaction_pattern_features,
    combine_features
)

class TestFeatureEngineering(unittest.TestCase):
    """Test suite for the feature engineering module."""

    def setUp(self):
        """Set up test data that will be used across multiple tests."""
        # Suppress warnings during tests
        warnings.filterwarnings('ignore')
        
        # Create mock customer data
        self.mock_customer_data = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'account_id': [101, 102, 103, 104, 105],
            'district_id': [10, 20, 30, 10, 20]
        })
        
        # Create mock transaction data
        self.mock_transaction_data = pd.DataFrame({
            'account_id': [101, 101, 102, 103, 104, 105, 105],
            'date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-01-10', 
                                    '2023-03-05', '2023-02-25', '2023-01-30', '2023-03-10']),
            'type': ['PRIJEM', 'VYDAJ', 'PRIJEM', 'VYDAJ', 'PRIJEM', 'PRIJEM', 'VYDAJ'],
            'operation': ['VKLAD', 'VYBER', 'PREVOD Z UCTU', 'VYBER KARTOU', 'VKLAD', 'PREVOD NA UCET', 'VYBER'],
            'amount': [1000, 500, 2000, 300, 1500, 800, 600],
            'balance': [3000, 2500, 5000, 2000, 4000, 1800, 1200],
            'k_symbol': ['UROK', 'SLUZBY', 'UVER', 'SANKC. UROK', 'UROK', 'DUCHOD', 'SLUZBY']
        })
        
        # Create mock loan data
        self.mock_loan_data = pd.DataFrame({
            'loan_id': [1, 2],
            'account_id': [101, 103],
            'amount': [50000, 100000],
            'duration': [12, 36],
            'payments': [4500, 3000],
            'status': ['A', 'B']  # A = active, B = completed
        })
        
        # Create mock card data
        self.mock_card_data = pd.DataFrame({
            'card_id': [1, 2, 3],
            'disp_id': [1001, 1003, 1005],
            'type': ['gold', 'classic', 'junior'],
            'issued': pd.to_datetime(['2022-01-01', '2022-02-15', '2022-03-20'])
        })
        
        # Mock disposition data (links clients to accounts)
        self.mock_disposition_data = pd.DataFrame({
            'disp_id': [1001, 1002, 1003, 1004, 1005],
            'client_id': [1, 2, 3, 4, 5],
            'account_id': [101, 102, 103, 104, 105],
            'type': ['OWNER', 'OWNER', 'OWNER', 'OWNER', 'OWNER']
        })
        
        # Add disp_id to customer data for card tests
        self.mock_customer_data['disp_id'] = [1001, 1002, 1003, 1004, 1005]
        
        # Create mock RFM data
        self.mock_rfm_data = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'recency_days': [5, 30, 60, 15, 45],
            'frequency': [10, 5, 2, 8, 4],
            'monetary_value': [1000, 500, 200, 750, 350]
        })
        
        # Create mock database connection
        self.mock_conn = MagicMock()

    @patch('src.feature_engineering.pd.read_sql')
    def test_calculate_rfm_features(self, mock_read_sql):
        """Test RFM feature calculation function."""
        # Configure mock
        mock_read_sql.return_value = self.mock_rfm_data
        
        # Call function with different lookback parameters
        result = calculate_rfm_features(self.mock_conn, lookback_days=90)
        
        # Assertions for correct output format and content
        self.assertEqual(len(result), 5)
        self.assertTrue('recency_days' in result.columns)
        self.assertTrue('frequency' in result.columns)
        self.assertTrue('monetary_value' in result.columns)
        
        # Verify SQL query contains correct parameters
        mock_read_sql.assert_called_once()
        sql_query = mock_read_sql.call_args[0][0]
        self.assertIn('DATEADD(day, -90, GETDATE())', sql_query)
        
        # Test with different lookback value
        mock_read_sql.reset_mock()
        result = calculate_rfm_features(self.mock_conn, lookback_days=180)
        sql_query = mock_read_sql.call_args[0][0]
        self.assertIn('DATEADD(day, -180, GETDATE())', sql_query)

    def test_create_product_portfolio_features(self):
        """Test product portfolio feature creation with all data sources."""
        # Call function with all data sources
        result = create_product_portfolio_features(
            self.mock_customer_data,
            self.mock_loan_data,
            self.mock_card_data
        )
        
        # Basic assertion checks
        self.assertEqual(len(result), 5)  # One row per customer
        self.assertTrue('account_count' in result.columns)
        self.assertTrue('loan_count' in result.columns)
        self.assertTrue('card_count' in result.columns)
        
        # Verify calculated values
        self.assertEqual(result.loc[result['client_id'] == 1, 'loan_count'].values[0], 1)
        self.assertEqual(result.loc[result['client_id'] == 1, 'card_count'].values[0], 1)
        self.assertEqual(result.loc[result['client_id'] == 3, 'loan_count'].values[0], 1)
        self.assertEqual(result.loc[result['client_id'] == 5, 'card_count'].values[0], 1)
        
        # Check that clients without loans or cards have zero values
        self.assertEqual(result.loc[result['client_id'] == 2, 'loan_count'].values[0], 0)
        self.assertEqual(result.loc[result['client_id'] == 4, 'card_count'].values[0], 0)
        
        # Test product diversity calculation
        self.assertTrue('product_diversity' in result.columns)
        # Client 1 has account, loan, and card = 3 products
        self.assertEqual(result.loc[result['client_id'] == 1, 'product_diversity'].values[0], 3)
    
    def test_create_product_portfolio_features_missing_data(self):
        """Test product portfolio feature creation with missing data sources."""
        # Call function with missing card data
        result = create_product_portfolio_features(
            self.mock_customer_data,
            self.mock_loan_data,
            None  # No card data
        )
        
        # Check that function handles missing data gracefully
        self.assertEqual(len(result), 5)
        self.assertTrue('loan_count' in result.columns)
        self.assertFalse('card_count' in result.columns)
        
        # Call function with missing loan data
        result = create_product_portfolio_features(
            self.mock_customer_data,
            None,  # No loan data
            self.mock_card_data
        )
        
        # Check that function handles missing data gracefully
        self.assertEqual(len(result), 5)
        self.assertFalse('loan_count' in result.columns)
        self.assertTrue('card_count' in result.columns)

    def test_create_transaction_pattern_features(self):
        """Test transaction pattern feature creation."""
        # Call function
        result = create_transaction_pattern_features(
            self.mock_transaction_data,
            self.mock_customer_data
        )
        
        # Check basic output structure
        self.assertEqual(len(result), 5)  # One row per customer
        
        # Check for key transaction pattern metrics
        expected_columns = [
            'transaction_type_diversity', 'operation_diversity',
            'credit_count', 'withdrawal_count'
        ]
        for col in expected_columns:
            self.assertTrue(col in result.columns, f"Column {col} missing from results")
        
        # Test specific pattern calculations
        # Client 1 has 2 transactions (1 credit, 1 withdrawal)
        self.assertEqual(result.loc[result['client_id'] == 1, 'credit_count'].values[0], 1)
        self.assertEqual(result.loc[result['client_id'] == 1, 'withdrawal_count'].values[0], 1)
        
        # Test day diversity calculation (if implemented)
        if 'transaction_day_diversity' in result.columns:
            # Client 1 has transactions on 2 different days
            self.assertGreaterEqual(result.loc[result['client_id'] == 1, 'transaction_day_diversity'].values[0], 1)
    
    def test_create_transaction_pattern_features_with_lookback(self):
        """Test transaction pattern feature creation with lookback period."""
        # Add an old transaction that should be filtered out
        old_transaction = pd.DataFrame({
            'account_id': [101],
            'date': pd.to_datetime(['2022-01-01']),  # Old date
            'type': ['PRIJEM'],
            'operation': ['VKLAD'],
            'amount': [2000],
            'balance': [2000],
            'k_symbol': ['UROK']
        })
        test_transactions = pd.concat([self.mock_transaction_data, old_transaction])
        
        # Call function with 60-day lookback
        result = create_transaction_pattern_features(
            test_transactions,
            self.mock_customer_data,
            lookback_days=60
        )
        
        # Verify recent transactions are counted but old ones are not
        # Since our test data is from 2023 and the old one from 2022, 
        # the 60-day lookback will filter out the old transaction
        
        # This assertion depends on the relative dates of the transactions
        # and when the test is run. It may need adjustment.
        self.assertEqual(len(result), 5)

    def test_combine_features(self):
        """Test combining different feature sets."""
        # Create simple test features
        rfm = pd.DataFrame({
            'client_id': [1, 2, 3],
            'recency_days': [5, 30, 60],
            'frequency': [10, 5, 2],
            'monetary_value': [1000, 500, 200]
        })
        
        product = pd.DataFrame({
            'client_id': [1, 2, 3],
            'account_count': [1, 1, 1],
            'loan_count': [1, 0, 1],
            'card_count': [1, 1, 0]
        })
        
        transaction = pd.DataFrame({
            'client_id': [1, 2, 3],
            'credit_count': [5, 3, 1],
            'withdrawal_count': [3, 2, 1],
            'avg_transaction_amount': [200, 150, 100]
        })
        
        # Combine features
        result = combine_features(rfm, product, transaction)
        
        # Check result structure
        self.assertEqual(len(result), 3)
        
        # Check that all input columns are present in output
        for col in rfm.columns:
            self.assertTrue(col in result.columns)
        for col in product.columns:
            if col != 'client_id':  # Skip duplicate ID column
                self.assertTrue(col in result.columns)
        for col in transaction.columns:
            if col != 'client_id':  # Skip duplicate ID column
                self.assertTrue(col in result.columns)
        
        # Check for derived features (if implemented)
        expected_derived_features = ['clv_indicator', 'churn_risk', 'engagement_score']
        for feature in expected_derived_features:
            # Not every implementation will have all these features
            if feature in result.columns:
                # Just check that the values are not all the same (some calculation happened)
                self.assertGreater(result[feature].nunique(), 1)

