"""
Feature engineering functions for Customer Lifetime Value and Segmentation Analysis.

This module contains functions to generate various feature sets for banking customers,
organized into core RFM metrics, product portfolio features, transaction patterns,
and combined feature generation.
"""
import pandas as pd
import numpy as np
from datetime import datetime

#####################################
# SECTION 1: CORE RFM FEATURES
#####################################

def calculate_rfm_features(transaction_data, customer_data, reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for customers
    
    Parameters:
    -----------
    transaction_data : pandas.DataFrame
        Transaction data with account_id, date, and amount
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
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(transaction_data['date']):
        transaction_data['date'] = pd.to_datetime(transaction_data['date'])
    
    # Merge transaction data with customer data to get client_id
    merged_data = pd.merge(
        transaction_data,
        customer_data[['client_id', 'account_id']],
        on='account_id'
    )
    
    # Calculate RFM metrics
    rfm = merged_data.groupby('client_id').agg({
        'date': lambda x: (reference_date - x.max()).days,  # Recency
        'date': 'count',  # Frequency
        'amount': lambda x: x[x > 0].sum()   # Monetary (positive transactions only)
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['client_id', 'recency_days', 'frequency', 'monetary_value']
    
    # Calculate additional RFM-related metrics
    transaction_stats = merged_data.groupby('client_id').agg({
        'date': [lambda x: (x.max() - x.min()).days,  # Customer tenure in days
                 lambda x: x.min()],  # First transaction date
        'amount': ['mean', 'std'],  # Transaction amount statistics
        'balance': ['mean', 'min', 'max']  # Balance statistics
    })
    
    # Flatten multi-level column names
    transaction_stats.columns = ['_'.join(col).strip() for col in transaction_stats.columns.values]
    transaction_stats = transaction_stats.reset_index()
    
    # Rename the columns to be more descriptive
    transaction_stats = transaction_stats.rename(columns={
        'date_<lambda_0>': 'customer_tenure_days',
        'date_<lambda_1>': 'first_transaction_date',
        'amount_mean': 'avg_transaction_amount',
        'amount_std': 'std_transaction_amount',
        'balance_mean': 'avg_balance',
        'balance_min': 'min_balance',
        'balance_max': 'max_balance'
    })
    
    # Merge RFM with additional metrics
    rfm = pd.merge(rfm, transaction_stats, on='client_id', how='left')
    
    # Calculate derived RFM metrics
    rfm['monetary_per_transaction'] = rfm['monetary_value'] / rfm['frequency']
    rfm['transaction_frequency'] = rfm.apply(
        lambda x: x['frequency'] / x['customer_tenure_days'] if x['customer_tenure_days'] > 0 else 0, 
        axis=1
    )
    rfm['balance_range'] = rfm['max_balance'] - rfm['min_balance']
    
    return rfm

#####################################
# SECTION 2: PRODUCT PORTFOLIO FEATURES
#####################################

def create_product_portfolio_features(customer_data, loan_data, card_data, order_data=None):
    """
    Create features based on the banking products each customer has
    
    Parameters:
    -----------
    customer_data : pandas.DataFrame
        Customer data with client_id and account information
    loan_data : pandas.DataFrame
        Loan data with loan details
    card_data : pandas.DataFrame
        Credit card data
    order_data : pandas.DataFrame, optional
        Permanent order data for standing payments
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with product portfolio features by client_id
    """
    # Count accounts by client
    product_features = customer_data.groupby('client_id').agg({
        'account_id': 'nunique'
    }).rename(columns={'account_id': 'account_count'}).reset_index()
    
    # Add loan features
    if loan_data is not None and not loan_data.empty:
        # Merge to get client_id for each loan
        loan_by_client = pd.merge(
            loan_data,
            customer_data[['client_id', 'account_id']], 
            on='account_id'
        )
        
        # Aggregate loan features by client
        loan_features = loan_by_client.groupby('client_id').agg({
            'loan_id': 'count',  # Number of loans
            'amount': ['mean', 'sum'],  # Loan amount statistics
            'duration': 'mean',  # Average loan duration
            'status': lambda x: (x == 'A').sum(),  # Count of active loans (status A)
            'status': lambda x: (x == 'B').sum(),  # Count of completed loans (status B)
            'status': lambda x: (x == 'C').sum(),  # Count of in progress loans (status C)
            'status': lambda x: (x == 'D').sum()   # Count of defaulted loans (status D)
        })
        
        # Flatten multi-level column names
        loan_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in loan_features.columns]
        loan_features = loan_features.reset_index()
        
        # Rename columns
        loan_features = loan_features.rename(columns={
            'loan_id_count': 'loan_count',
            'amount_mean': 'avg_loan_amount',
            'amount_sum': 'total_loan_amount',
            'duration_mean': 'avg_loan_duration',
            'status_<lambda_0>': 'active_loans',
            'status_<lambda_1>': 'completed_loans',
            'status_<lambda_2>': 'in_progress_loans', 
            'status_<lambda_3>': 'defaulted_loans'
        })
        
        # Calculate loan-specific metrics
        loan_features['loan_default_rate'] = loan_features.apply(
            lambda x: x['defaulted_loans'] / x['loan_count'] if x['loan_count'] > 0 else 0, 
            axis=1
        )
        
        # Merge with product features
        product_features = pd.merge(product_features, loan_features, on='client_id', how='left')
    
    # Add credit card features
    if card_data is not None and not card_data.empty:
        # Process card data similarly to loan data
        # First get disposition data to link cards to accounts and clients
        if 'disp_id' in card_data.columns:
            # Get disposition data from customer_data if it contains the necessary columns
            disposition_data = customer_data[['client_id', 'account_id', 'disp_id']] if 'disp_id' in customer_data.columns else None
            
            if disposition_data is not None:
                # Link cards to clients
                card_by_client = pd.merge(card_data, disposition_data, on='disp_id')
                
                # Aggregate card features
                card_features = card_by_client.groupby('client_id').agg({
                    'card_id': 'count',
                    'type': lambda x: (x == 'gold').sum(),  # Count of gold cards
                    'type': lambda x: (x == 'classic').sum(),  # Count of classic cards
                    'type': lambda x: (x == 'junior').sum()  # Count of junior cards
                })
                
                # Fix column names
                card_features.columns = ['card_count', 'gold_cards', 'classic_cards', 'junior_cards']
                card_features = card_features.reset_index()
                
                # Merge with product features
                product_features = pd.merge(product_features, card_features, on='client_id', how='left')
    
    # Process permanent orders if available
    if order_data is not None and not order_data.empty:
        # Link orders to clients
        order_by_client = pd.merge(order_data, customer_data[['client_id', 'account_id']], on='account_id')
        
        # Aggregate order features
        order_features = order_by_client.groupby('client_id').agg({
            'order_id': 'count',  # Number of permanent orders
            'amount': ['mean', 'sum']  # Order amount statistics
        })
        
        # Flatten and rename columns
        order_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in order_features.columns]
        order_features = order_features.rename(columns={
            'order_id_count': 'permanent_order_count',
            'amount_mean': 'avg_order_amount',
            'amount_sum': 'total_order_amount'
        }).reset_index()
        
        # Merge with product features
        product_features = pd.merge(product_features, order_features, on='client_id', how='left')
    
    # Fill NaN values with 0 for clients without certain products
    product_features = product_features.fillna(0)
    
    # Calculate product diversity score (number of different products)
    product_features['product_diversity'] = product_features.apply(
        lambda x: sum([
            1 if x['account_count'] > 0 else 0,
            1 if x.get('loan_count', 0) > 0 else 0,
            1 if x.get('card_count', 0) > 0 else 0,
            1 if x.get('permanent_order_count', 0) > 0 else 0
        ]),
        axis=1
    )
    
    return product_features

#####################################
# SECTION 3: TRANSACTION PATTERN FEATURES
#####################################

def create_transaction_pattern_features(transaction_data, customer_data, lookback_days=None):
    """
    Create features based on transaction patterns
    
    Parameters:
    -----------
    transaction_data : pandas.DataFrame
        Transaction data with account_id, date, type, operation, amount, balance, etc.
    customer_data : pandas.DataFrame
        Customer data linking client_id to account_id
    lookback_days : int, optional
        Number of days to look back for recent transaction patterns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with transaction pattern features by client_id
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(transaction_data['date']):
        transaction_data['date'] = pd.to_datetime(transaction_data['date'])
    
    # Filter for recent transactions if lookback_days is specified
    if lookback_days is not None:
        cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
        transaction_data = transaction_data[transaction_data['date'] >= cutoff_date]
    
    # Merge transaction data with customer data to get client_id
    merged_data = pd.merge(
        transaction_data,
        customer_data[['client_id', 'account_id']],
        on='account_id'
    )
    
    # Calculate transaction pattern metrics
    pattern_features = merged_data.groupby('client_id').agg({
        'type': lambda x: x.nunique(),  # Number of unique transaction types
        'operation': lambda x: x.nunique(),  # Number of unique operations
        'k_symbol': lambda x: x.nunique()  # Number of unique transaction categories
    }).rename(columns={
        'type': 'transaction_type_diversity',
        'operation': 'operation_diversity',
        'k_symbol': 'transaction_category_diversity'
    }).reset_index()
    
    # Credit vs withdrawal patterns
    credit_withdrawal = merged_data.groupby('client_id').apply(
        lambda x: pd.Series({
            'credit_count': sum(x['type'] == 'CREDIT'),
            'withdrawal_count': sum(x['type'] == 'WITHDRAWAL'),
            'credit_amount': x[x['type'] == 'CREDIT']['amount'].sum(),
            'withdrawal_amount': x[x['type'] == 'WITHDRAWAL']['amount'].sum(),
            'avg_credit_amount': x[x['type'] == 'CREDIT']['amount'].mean() if any(x['type'] == 'CREDIT') else 0,
            'avg_withdrawal_amount': x[x['type'] == 'WITHDRAWAL']['amount'].mean() if any(x['type'] == 'WITHDRAWAL') else 0,
            'std_credit_amount': x[x['type'] == 'CREDIT']['amount'].std() if sum(x['type'] == 'CREDIT') > 1 else 0,
            'std_withdrawal_amount': x[x['type'] == 'WITHDRAWAL']['amount'].std() if sum(x['type'] == 'WITHDRAWAL') > 1 else 0
        })
    ).reset_index()
    
    # Calculate derived transaction pattern metrics
    credit_withdrawal['credit_withdrawal_ratio'] = credit_withdrawal.apply(
        lambda x: x['credit_amount'] / x['withdrawal_amount'] if x['withdrawal_amount'] > 0 else float('inf'),
        axis=1
    )
    credit_withdrawal['transaction_volatility'] = credit_withdrawal.apply(
        lambda x: (x['std_credit_amount'] + x['std_withdrawal_amount']) / 2 if (x['credit_count'] > 0 and x['withdrawal_count'] > 0) else 0,
        axis=1
    )
    
    # Temporal patterns (if date information available)
    if 'date' in merged_data.columns:
        # Weekday vs weekend transaction patterns
        merged_data['day_of_week'] = merged_data['date'].dt.dayofweek
        merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
        
        temporal_patterns = merged_data.groupby('client_id').apply(
            lambda x: pd.Series({
                'weekend_transaction_ratio': sum(x['is_weekend']) / len(x) if len(x) > 0 else 0,
                'weekday_transaction_count': sum(~x['is_weekend'].astype(bool)),
                'weekend_transaction_count': sum(x['is_weekend']),
                'transaction_day_diversity': x['day_of_week'].nunique()
            })
        ).reset_index()
        
        # Merge temporal patterns with other features
        pattern_features = pd.merge(pattern_features, temporal_patterns, on='client_id', how='left')
    
    # Merge credit/withdrawal patterns
    pattern_features = pd.merge(pattern_features, credit_withdrawal, on='client_id', how='left')
    
    # Check for balance volatility
    if 'balance' in merged_data.columns:
        balance_patterns = merged_data.groupby('client_id').agg({
            'balance': ['std', lambda x: x.diff().abs().mean()]
        })
        balance_patterns.columns = ['balance_std', 'balance_daily_change']
        balance_patterns = balance_patterns.reset_index()
        
        # Merge balance patterns
        pattern_features = pd.merge(pattern_features, balance_patterns, on='client_id', how='left')
    
    # Fill NaN values with 0
    pattern_features = pattern_features.fillna(0)
    
    return pattern_features

#####################################
# SECTION 4: COMBINED FEATURE GENERATION
#####################################

def combine_features(rfm_features, product_features, transaction_features, demographic_data=None):
    """
    Combine all features into a single customer feature dataset
    
    Parameters:
    -----------
    rfm_features : pandas.DataFrame
        RFM features by client_id
    product_features : pandas.DataFrame
        Product features by client_id
    transaction_features : pandas.DataFrame
        Transaction pattern features by client_id
    demographic_data : pandas.DataFrame, optional
        Demographic data by district_id
        
    Returns:
    --------
    pandas.DataFrame
        Combined feature set for all customers
    """
    # Merge RFM and product features
    combined = pd.merge(rfm_features, product_features, on='client_id', how='inner')
    
    # Merge transaction pattern features
    combined = pd.merge(combined, transaction_features, on='client_id', how='left')
    
    # Add demographic features if available
    if demographic_data is not None and 'district_id' in combined.columns:
        # Join demographic data by district_id
        combined = pd.merge(combined, demographic_data, on='district_id', how='left')
    
    # Fill remaining NaN values
    combined = combined.fillna(0)
    
    # Calculate advanced banking metrics
    
    # 1. Customer value indicators
    combined['clv_indicator'] = combined.apply(
        lambda x: (x['monetary_value'] * 0.5) + (x['product_diversity'] * 0.3) - (x['recency_days'] * 0.2),
        axis=1
    )
    
    # 2. Churn risk indicator (higher value = higher risk)
    combined['churn_risk'] = combined.apply(
        lambda x: (x['recency_days'] * 0.6) + (100 - x['transaction_frequency'] * 100) * 0.4,
        axis=1
    )
    
    # 3. Cross-sell potential
    if 'product_diversity' in combined.columns:
        max_products = 4  # Account, loan, card, permanent order
        combined['cross_sell_potential'] = max_products - combined['product_diversity']
    
    # 4. Customer engagement score
    combined['engagement_score'] = combined.apply(
        lambda x: (
            (100 - min(x['recency_days'], 100)) * 0.4 +  # More recent = better
            min(x['frequency'], 100) * 0.2 +  # More transactions = better
            min(x['transaction_type_diversity'] * 20, 100) * 0.2 +  # More diversity = better
            min(x['product_diversity'] * 25, 100) * 0.2  # More products = better
        ),
        axis=1
    )
    
    return combined


def get_complete_customer_features(transaction_data, customer_data, loan_data=None, card_data=None, 
                                   order_data=None, demographic_data=None, reference_date=None, lookback_days=None):
    """
    Generate a complete set of customer features for CLV and segmentation
    
    Parameters:
    -----------
    transaction_data : pandas.DataFrame
        Transaction data with account_id, date, type, operation, amount, balance, etc.
    customer_data : pandas.DataFrame
        Customer data linking client_id to account_id
    loan_data : pandas.DataFrame, optional
        Loan data
    card_data : pandas.DataFrame, optional
        Credit card data
    order_data : pandas.DataFrame, optional
        Permanent order data
    demographic_data : pandas.DataFrame, optional
        Demographic data by district_id
    reference_date : datetime, optional
        Date to calculate recency from (default: today)
    lookback_days : int, optional
        Number of days to look back for recent transaction patterns
        
    Returns:
    --------
    pandas.DataFrame
        Complete feature set for all customers
    """
    # Calculate RFM features
    rfm_features = calculate_rfm_features(transaction_data, customer_data, reference_date)
    
    # Calculate product portfolio features
    product_features = create_product_portfolio_features(customer_data, loan_data, card_data, order_data)
    
    # Calculate transaction pattern features
    transaction_features = create_transaction_pattern_features(transaction_data, customer_data, lookback_days)
    
    # Combine all features
    complete_features = combine_features(rfm_features, product_features, transaction_features, demographic_data)
    
    return complete_features