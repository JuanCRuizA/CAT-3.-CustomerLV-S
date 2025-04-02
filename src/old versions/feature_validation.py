# Feature Engineering Validation Script

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import load_transactions_data, load_customer_data
from feature_engineering import (
    calculate_rfm_features, 
    create_product_portfolio_features, 
    create_transaction_pattern_features,
    get_complete_customer_features
)

def validate_feature_extraction():
    """
    Validate feature extraction functions by loading data 
    and applying different feature generation methods
    """
    # Load data
    transactions_df = load_transactions_data()
    customer_df = load_customer_data()

    # 1. RFM Features Validation
    print("\n--- RFM Features Validation ---")
    rfm_features = calculate_rfm_features(transactions_df, customer_df)
    print(f"RFM Features - Total Customers: {len(rfm_features)}")
    print("\nRFM Features Summary:")
    print(rfm_features.describe())

    # Visualize RFM distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(rfm_features['recency_days'], kde=True)
    plt.title('Recency Days Distribution')
    
    plt.subplot(132)
    sns.histplot(rfm_features['frequency'], kde=True)
    plt.title('Transaction Frequency Distribution')
    
    plt.subplot(133)
    sns.histplot(rfm_features['monetary_value'], kde=True)
    plt.title('Monetary Value Distribution')
    
    plt.tight_layout()
    plt.savefig('../reports/rfm_distributions.png')
    plt.close()

    # 2. Product Portfolio Features Validation
    print("\n--- Product Portfolio Features Validation ---")
    product_features = create_product_portfolio_features(
        customer_df, 
        loan_data=load_transactions_data(),  # Replace with actual loan data if available
        card_data=load_transactions_data(),  # Replace with actual card data if available
        order_data=load_transactions_data()  # Replace with actual order data if available
    )
    print(f"Product Portfolio Features - Total Customers: {len(product_features)}")
    print("\nProduct Portfolio Features Summary:")
    print(product_features.describe())

    # 3. Transaction Pattern Features Validation
    print("\n--- Transaction Pattern Features Validation ---")
    transaction_features = create_transaction_pattern_features(transactions_df, customer_df)
    print(f"Transaction Pattern Features - Total Customers: {len(transaction_features)}")
    print("\nTransaction Pattern Features Summary:")
    print(transaction_features.describe())

    # 4. Complete Customer Features
    print("\n--- Complete Customer Features Validation ---")
    complete_features = get_complete_customer_features(
        transactions_df, 
        customer_df
    )
    print(f"Complete Features - Total Customers: {len(complete_features)}")
    print("\nComplete Features Summary:")
    print(complete_features.describe())

    # Advanced Visualization of Complete Features
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    sns.scatterplot(data=complete_features, x='monetary_value', y='engagement_score', hue='product_diversity')
    plt.title('Monetary Value vs Engagement Score')
    
    plt.subplot(222)
    sns.scatterplot(data=complete_features, x='recency_days', y='churn_risk', hue='product_diversity')
    plt.title('Recency Days vs Churn Risk')
    
    plt.subplot(223)
    sns.histplot(complete_features['clv_indicator'], kde=True)
    plt.title('Customer Lifetime Value Indicator')
    
    plt.subplot(224)
    sns.histplot(complete_features['cross_sell_potential'], kde=True)
    plt.title('Cross-Sell Potential')
    
    plt.tight_layout()
    plt.savefig('../reports/complete_features_analysis.png')
    plt.close()

    return complete_features

if __name__ == "__main__":
    validate_feature_extraction()