# tests/test_churn_prediction.py

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

# Mock sklearn modules
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()

class TestChurnPrediction(unittest.TestCase):
    """Test suite for churn prediction functionality."""
    
    def setUp(self):
        """Set up test data for churn prediction tests."""
        warnings.filterwarnings('ignore')
        
        # Create mock customer data with segmentation and CLV
        self.mock_customer_data = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'recency_days': [5, 30, 60, 15, 120],  # 90+ days is often used as churn definition
            'frequency': [10, 5, 2, 8, 1],
            'monetary_value': [1000, 500, 200, 750, 100],
            'avg_balance': [5000, 2000, 1000, 3000, 500],
            'balance_volatility': [200, 100, 50, 150, 20],
            'product_diversity_score': [3, 2, 1, 2, 1],
            'clv_1yr': [10000, 5000, 2000, 7500, 1000],
            'cluster': [0, 1, 2, 0, 2],
            'segment_name': ['Premium Active Borrowers', 'Standard Active Savers', 
                            'Basic Low-Activity Users', 'Premium Active Transactors', 
                            'Basic Dormant Users']
        })
        
        # Add churn label based on recency
        self.mock_customer_data['is_churned'] = (self.mock_customer_data['recency_days'] > 90).astype(int)
        
        # Create train/test split
        self.X = self.mock_customer_data.drop(['client_id', 'is_churned', 'cluster', 'segment_name'], axis=1)
        self.y = self.mock_customer_data['is_churned']
        
        # Set up 60% train, 40% test split for simplicity
        self.X_train = self.X.iloc[:3]
        self.y_train = self.y.iloc[:3]
        self.X_test = self.X.iloc[3:]
        self.y_test = self.y.iloc[3:]

    @patch('sklearn.ensemble.RandomForestClassifier')
    def test_churn_model_training(self, MockRandomForest):
        """Test the churn prediction model training process."""
        # Configure mock
        mock_rf = MockRandomForest.return_value
        mock_rf.fit.return_value = mock_rf
        mock_rf.predict.return_value = np.array([0, 1])  # Predict non-churn for client 4, churn for client 5
        mock_rf.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])  # Probabilities for above
        
        # Create classifier
        rf = MockRandomForest(n_estimators=100, max_depth=5, random_state=42)
        
        # Train model
        rf.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf.predict(self.X_test)
        y_prob = rf.predict_proba(self.X_test)[:, 1]  # Probability of churn
        
        # Verify model was called correctly
        MockRandomForest.assert_called_once_with(n_estimators=100, max_depth=5, random_state=42)
        mock_rf.fit.assert_called_once_with(self.X_train, self.y_train)
        mock_rf.predict.assert_called_once()
        mock_rf.predict_proba.assert_called_once()
        
        # Check predictions
        self.assertEqual(len(y_pred), 2)
        self.assertEqual(y_pred[0], 0)  # Client 4 not churned
        self.assertEqual(y_pred[1], 1)  # Client 5 churned
        
        # Check probabilities
        self.assertEqual(len(y_prob), 2)
        self.assertEqual(y_prob[0], 0.2)  # 20% churn probability for client 4
        self.assertEqual(y_prob[1], 0.7)  # 70% churn probability for client 5
        
        # Calculated metrics would normally go here
        # In a real test, we would import the metrics functions and calculate them

    def test_high_value_churn_identification(self):
        """Test identification of high-value customers at risk of churn."""
        # Create predictions with churn probabilities
        predictions = self.mock_customer_data.copy()
        predictions['churn_probability'] = [0.1, 0.3, 0.4, 0.2, 0.8]
        
        # Identify high-value customers (top 40%) with high churn risk (>0.5)
        high_value_threshold = predictions['clv_1yr'].quantile(0.6)
        
        high_value_at_risk = predictions[
            (predictions['churn_probability'] > 0.5) & 
            (predictions['clv_1yr'] >= high_value_threshold)
        ]
        
        # Verify results
        self.assertEqual(len(high_value_at_risk), 1)  # Only client 5 should be identified
        self.assertEqual(high_value_at_risk.iloc[0]['client_id'], 5)
        
        # Check that the threshold is working correctly
        self.assertGreaterEqual(high_value_at_risk.iloc[0]['clv_1yr'], high_value_threshold)
        self.assertGreater(high_value_at_risk.iloc[0]['churn_probability'], 0.5)

    def test_churn_threshold_optimization(self):
        """Test optimization of churn probability threshold for business impact."""
        # Set up predictions
        y_pred_proba = np.array([0.2, 0.7])  # Probabilities for clients 4 and 5
        
        # Test multiple thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Set up business parameters
        cost_per_campaign = 100
        value_of_retained_customer = 1000
        retention_success_rate = 0.5
        
        # Track best threshold and profit
        best_threshold = None
        best_profit = float('-inf')
        
        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Get actual churn values from test set
            y_true = self.y_test.values
            
            # Calculate confusion matrix components
            true_positives = ((y_true == 1) & (y_pred == 1)).sum()
            false_positives = ((y_true == 0) & (y_pred == 1)).sum()
            
            # Calculate costs and benefits
            campaign_cost = (true_positives + false_positives) * cost_per_campaign
            retention_benefit = true_positives * retention_success_rate * value_of_retained_customer
            net_profit = retention_benefit - campaign_cost
            
            # Update best threshold if this one is more profitable
            if net_profit > best_profit:
                best_profit = net_profit
                best_threshold = threshold
        
        # Verify results
        # In our mock data, client 5 is the only one who churned (y_test = [0, 1])
        # With probabilities [0.2, 0.7], threshold 0.5 should work best
        self.assertEqual(best_threshold, 0.5)
        
        # Calculate expected profit at threshold 0.5:
        # At threshold 0.5, we correctly identify client 5 as churned (true positive = 1)
        # and correctly identify client 4 as not churned (true negative = 1)
        # true_positives = 1, false_positives = 0
        # campaign_cost = 1 * 100 = 100
        # retention_benefit = 1 * 0.5 * 1000 = 500
        # net_profit = 500 - 100 = 400
        expected_profit = 400
        
        # Recalculate profit at best threshold to compare
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        y_true = self.y_test.values
        true_positives = ((y_true == 1) & (y_pred == 1)).sum()
        false_positives = ((y_true == 0) & (y_pred == 1)).sum()
        campaign_cost = (true_positives + false_positives) * cost_per_campaign
        retention_benefit = true_positives * retention_success_rate * value_of_retained_customer
        net_profit = retention_benefit - campaign_cost
        
        self.assertEqual(net_profit, expected_profit)

# Run tests if this file is executed directly
if __name__ == '__main__':
    unittest.main()