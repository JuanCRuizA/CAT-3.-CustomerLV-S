"""
Feature engineering functions for CLV and segmentation analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm_features(transaction_data, customer_data, reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for customers
    
    Parameters:
    -----------
    transaction_data : pandas.DataFrame
        Transaction data with Account_id, Trans_date, and Amount
    customer_data : pandas.DataFrame
        Customer data linking client_id to account_id
    reference_date : datetime, optional
        Date to calculate recency from (default: today)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with client_id and RFM features
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Merge transaction data with customer data to get client_id
    merged_data = pd.merge(
        transaction_data,
        customer_data[['client_id', 'account_id']],
        left_on='Account_id',
        right_on='account_id'
    )
    
    # Calculate RFM metrics
    rfm = merged_data.groupby('client_id').agg({
        'Trans_date': lambda x: (reference_date - x.max()).days,  # Recency
        'Trans_id': 'count',  # Frequency
        'Amount': 'sum'   # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['client_id', 'recency_days', 'frequency', 'monetary_value']
    
    return rfm

def create_customer_product_features(customer_data, loan_data, card_data, order_data):
    """
    Create features based on the products each customer has
    
    Parameters:
    -----------
    customer_data : pandas.DataFrame
        Customer data with client_id and account information
    loan_data : pandas.DataFrame
        Loan data 
    card_data : pandas.DataFrame
        Credit card data
    order_data : pandas.DataFrame
        Permanent order data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with product features by client_id
    """
    # Count products by client
    product_features = customer_data.groupby('client_id').agg({
        'account_id': 'nunique'
    }).rename(columns={'account_id': 'account_count'}).reset_index()
    
    # Add loan features
    loan_by_client = loan_data.merge(
        customer_data[['client_id', 'account_id']], 
        left_on='Account_id',
        right_on='account_id'
    ).groupby('client_id').agg({
        'Loan_id': 'count',
        'Amount': 'mean'
    }).rename(columns={
        'Loan_id': 'loan_count',
        'Amount': 'avg_loan_amount'
    }).reset_index()
    
    # Merge with product features
    product_features = pd.merge(
        product_features, 
        loan_by_client, 
        on='client_id', 
        how='left'
    )
    
    # Fill NaN values with 0 for clients without loans
    product_features = product_features.fillna(0)
    
    # Add similar features for cards and orders
    # (Implement similar logic for cards and permanent orders)
    
    return product_features

def combine_features(rfm_features, product_features, demographic_data=None):
    """
    Combine all features into a single customer feature dataset
    
    Parameters:
    -----------
    rfm_features : pandas.DataFrame
        RFM features by client_id
    product_features : pandas.DataFrame
        Product features by client_id
    demographic_data : pandas.DataFrame, optional
        Demographic data by district_id
        
    Returns:
    --------
    pandas.DataFrame
        Combined feature set for all customers
    """
    # Merge RFM and product features
    combined = pd.merge(rfm_features, product_features, on='client_id', how='inner')
    
    # Add demographic features if available
    if demographic_data is not None:
        # Implement demographic data integration
        pass
    
    return combined