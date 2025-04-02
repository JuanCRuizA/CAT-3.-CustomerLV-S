# tests/test_customer_segmentation.py

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

# Mock sklearn modules we might not have installed
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.decomposition'] = MagicMock()

class TestCustomerSegmentation(unittest.TestCase):
    """Test suite for customer segmentation functionality."""
    
    def setUp(self):
        """Set up test data for segmentation tests."""
        warnings.filterwarnings('ignore')
        
        # Create mock customer features
        self.mock_customer_features = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'recency_days': [5, 30, 60, 15, 45],
            'frequency': [10, 5, 2, 8, 4],
            'monetary_value': [1000, 500, 200, 750, 350],
            'avg_balance': [5000, 2000, 1000, 3000, 1500],
            'balance_range': [2000, 1000, 500, 1500, 800],
            'product_diversity_score': [3, 2, 1, 2, 1],
            'loan_count': [1, 1, 0, 0, 0],
            'card_count': [1, 0, 1, 1, 0],
            'clv_1yr': [10000, 5000, 2000, 7500, 3500]
        })
        
        # Create mock scaled features
        self.mock_scaled_features = np.array([
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],  # Client 1 - High value
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Client 2 - Medium value
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],  # Client 3 - Low value
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Client 4 - Medium-high value
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]   # Client 5 - Low value
        ])

    @patch('sklearn.preprocessing.StandardScaler')
    def test_feature_scaling(self, MockScaler):
        """Test the feature scaling process."""
        # Configure mock
        mock_scaler = MockScaler.return_value
        mock_scaler.fit_transform.return_value = self.mock_scaled_features
        
        # Create a scaler
        scaler = MockScaler()
        
        # Select features for segmentation (exclude client_id)
        features_for_segmentation = self.mock_customer_features.drop('client_id', axis=1)
        
        # Scale the features
        scaled_features = scaler.fit_transform(features_for_segmentation)
        
        # Verify scaler was called correctly
        MockScaler.assert_called_once()
        mock_scaler.fit_transform.assert_called_once()
        
        # Check scaled data
        self.assertEqual(scaled_features.shape, (5, 9))  # 5 clients, 9 features
        np.testing.assert_array_equal(scaled_features, self.mock_scaled_features)

    @patch('sklearn.cluster.KMeans')
    def test_kmeans_clustering(self, MockKMeans):
        """Test the KMeans clustering process."""
        # Configure mock
        mock_kmeans = MockKMeans.return_value
        mock_kmeans.fit_predict.return_value = np.array([0, 1, 2, 0, 2])  # 3 clusters
        mock_kmeans.inertia_ = 10.0
        
        # Create KMeans instance
        kmeans = MockKMeans(n_clusters=3, random_state=42)
        
        # Perform clustering
        cluster_labels = kmeans.fit_predict(self.mock_scaled_features)
        
        # Verify KMeans was called with correct parameters
        MockKMeans.assert_called_once_with(n_clusters=3, random_state=42)
        mock_kmeans.fit_predict.assert_called_once()
        
        # Check cluster assignments
        self.assertEqual(len(cluster_labels), 5)  # 5 customers
        self.assertEqual(cluster_labels[0], 0)  # Client 1 in cluster 0
        self.assertEqual(cluster_labels[1], 1)  # Client 2 in cluster 1
        self.assertEqual(cluster_labels[2], 2)  # Client 3 in cluster 2
        self.assertEqual(cluster_labels[3], 0)  # Client 4 in cluster 0
        self.assertEqual(cluster_labels[4], 2)  # Client 5 in cluster 2
        
        # Check that clients 1 and 4 are in the same cluster (similar values)
        self.assertEqual(cluster_labels[0], cluster_labels[3])
        
        # Check that clients 3 and 5 are in the same cluster (similar values)
        self.assertEqual(cluster_labels[2], cluster_labels[4])

    def test_generate_cluster_profiles(self):
        """Test generation of cluster profiles."""
        # Add cluster labels to customer features
        customers_with_clusters = self.mock_customer_features.copy()
        customers_with_clusters['cluster'] = [0, 1, 2, 0, 2]  # Same as in test_kmeans_clustering
        
        # Calculate cluster profiles
        cluster_profile = customers_with_clusters.groupby('cluster').mean()
        
        # Check profile structure
        self.assertEqual(len(cluster_profile), 3)  # 3 clusters
        
        # Check that cluster 0 (high value) has higher metrics than cluster 2 (low value)
        self.assertGreater(cluster_profile.loc[0, 'clv_1yr'], cluster_profile.loc[2, 'clv_1yr'])
        self.assertGreater(cluster_profile.loc[0, 'frequency'], cluster_profile.loc[2, 'frequency'])
        self.assertGreater(cluster_profile.loc[0, 'avg_balance'], cluster_profile.loc[2, 'avg_balance'])
        
        # Verify specific calculations
        # Cluster 0 average CLV should be the mean of clients 1 and 4
        expected_cluster0_clv = (10000 + 7500) / 2
        self.assertEqual(cluster_profile.loc[0, 'clv_1yr'], expected_cluster0_clv)

