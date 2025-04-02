"""
Feature engineering module for BFSI Customer Lifetime Value and Segmentation Analysis.

This module contains functions to generate various feature sets for banking customers,
organized into core RFM metrics, product portfolio features, transaction patterns,
and combined feature generation.
"""
import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime, timedelta
import traceback

#####################################
# DATABASE CONNECTION HANDLING
#####################################

def get_database_connection(connection_string):
    """
    Establish a connection to the database.
    
    Args:
        connection_string (str): Database connection string for SQL Server
        
    Returns:
        pyodbc.Connection: An active database connection
        
    Raises:
        Exception: If connection fails
    """
    try:
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        raise Exception(f"Error connecting to database: {e}")

#####################################
# SECTION 1: CORE RFM FEATURES
#####################################

def calculate_rfm_features(conn, lookback_days=365, reference_date=None):
    """
    Calculate Recency, Frequency, and Monetary (RFM) features for customers from the database.
    Adapted for Czech language transaction types.
    
    Args:
        conn: Database connection object
        lookback_days (int): Number of days to look back for analysis. Defaults to 365 (1 year).
        reference_date: Reference date for recency calculation (default is current date in database)
        
    Returns:
        pandas.DataFrame: DataFrame with RFM features for each customer
    """
    # Set reference date clause
    date_clause = f"'{reference_date.strftime('%Y-%m-%d')}'" if reference_date else "GETDATE()"
    
    print(f"Using reference date clause: {date_clause}")
    
    query = f"""
    WITH CustomerTransactions AS (
        SELECT 
            d.client_id,
            MAX(t.Trans_date) AS last_transaction_date,
            COUNT(t.trans_id) AS transaction_frequency,
            SUM(CASE WHEN t.Trans_type = 'PRIJEM' THEN t.amount ELSE 0 END) AS total_monetary_value,
            AVG(t.amount) AS avg_transaction_amount,
            STDEV(t.amount) AS transaction_amount_volatility,
            MIN(t.Trans_date) AS first_transaction_date,
            AVG(t.balance) AS avg_balance,
            MIN(t.balance) AS min_balance,
            MAX(t.balance) AS max_balance
        FROM 
            Trans t
        JOIN 
            Account a ON t.account_id = a.account_id
        JOIN 
            Disposition d ON a.account_id = d.account_id
        WHERE 
            d.type = 'OWNER'
            AND t.Trans_date <= '{reference_date.strftime('%Y-%m-%d')}'
        GROUP BY 
            d.client_id
    )
    SELECT 
        client_id,
        DATEDIFF(day, last_transaction_date, {date_clause}) AS recency_days,
        transaction_frequency AS frequency,
        total_monetary_value AS monetary_value,
        avg_transaction_amount,
        transaction_amount_volatility,
        DATEDIFF(day, first_transaction_date, {date_clause}) AS customer_tenure_days,
        avg_balance,
        min_balance,
        max_balance,
        max_balance - min_balance AS balance_range,
        CASE 
            WHEN transaction_frequency > 0 THEN total_monetary_value / transaction_frequency 
            ELSE 0 
        END AS monetary_per_transaction
    FROM 
        CustomerTransactions
    """
    
    print("Executing RFM query...")
    # Test the query first
    try:
        result = pd.read_sql(query, conn)
        print(f"RFM query returned {len(result)} rows")
        return result
    except Exception as e:
        print(f"Error executing RFM query: {e}")
        print(f"Query was: {query}")
        traceback.print_exc()
        raise

def calculate_rfm_features_from_dataframes(transaction_data, customer_data, reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features from dataframes
    
    Args:
        transaction_data: DataFrame with account_id, date, and amount
        customer_data: DataFrame linking client_id to account_id
        reference_date: Reference date for recency calculation (default is today)
        
    Returns:
        pandas.DataFrame: DataFrame with client_id and RFM features
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
        'date': [
            lambda x: (reference_date - x.min()).days,  # Customer tenure in days
            lambda x: x.min()  # First transaction date
        ],
        'amount': ['mean', 'std'],  # Transaction amount statistics
        'balance': ['mean', 'min', 'max']  # Balance statistics
    })
    
    # Flatten multi-level column names
    transaction_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in transaction_stats.columns]
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

def extract_product_portfolio_features(conn):
    """
    Extract features related to customer's product portfolio from the database.
    
    Args:
        conn: Database connection object
        
    Returns:
        pandas.DataFrame: DataFrame with product portfolio features
    """
    query = """
    SELECT 
        cl.client_id,
        COUNT(DISTINCT a.account_id) AS total_account_count,
        COUNT(DISTINCT l.loan_id) AS loan_count,
        COUNT(DISTINCT c.card_id) AS card_count,
        COUNT(DISTINCT po.order_id) AS permanent_order_count,
        
        -- Product diversity score
        CASE 
            WHEN COUNT(DISTINCT a.account_id) > 0 THEN 1 ELSE 0 
        END +
        CASE 
            WHEN COUNT(DISTINCT l.loan_id) > 0 THEN 1 ELSE 0 
        END +
        CASE 
            WHEN COUNT(DISTINCT c.card_id) > 0 THEN 1 ELSE 0 
        END +
        CASE 
            WHEN COUNT(DISTINCT po.order_id) > 0 THEN 1 ELSE 0 
        END AS product_diversity_score,
        
        -- Loan-related features
        SUM(CASE WHEN l.loan_id IS NOT NULL THEN l.amount ELSE 0 END) AS total_loan_amount,
        AVG(CASE WHEN l.loan_id IS NOT NULL THEN l.amount END) AS avg_loan_amount,
        SUM(CASE WHEN l.Loan_status = 'A' THEN 1 ELSE 0 END) AS active_loans,
        SUM(CASE WHEN l.Loan_status = 'B' THEN 1 ELSE 0 END) AS completed_loans,
        SUM(CASE WHEN l.Loan_status = 'C' THEN 1 ELSE 0 END) AS in_progress_loans,
        SUM(CASE WHEN l.Loan_status = 'D' THEN 1 ELSE 0 END) AS defaulted_loans,
        
        -- Card-related features
        SUM(CASE WHEN c.type = 'gold' THEN 1 ELSE 0 END) AS gold_cards,
        SUM(CASE WHEN c.type = 'classic' THEN 1 ELSE 0 END) AS classic_cards,
        SUM(CASE WHEN c.type = 'junior' THEN 1 ELSE 0 END) AS junior_cards,
        
        -- Order-related features
        AVG(CASE WHEN po.order_id IS NOT NULL THEN po.amount END) AS avg_order_amount,
        SUM(CASE WHEN po.order_id IS NOT NULL THEN po.amount ELSE 0 END) AS total_order_amount
    FROM 
        Client cl
    LEFT JOIN 
        Disposition d ON cl.client_id = d.client_id
    LEFT JOIN 
        Account a ON d.account_id = a.account_id
    LEFT JOIN 
        Loan l ON a.account_id = l.account_id
    LEFT JOIN 
        Card c ON d.disp_id = c.disp_id
    LEFT JOIN 
        PermanentOrder po ON a.account_id = po.account_id
    WHERE 
        d.type = 'OWNER'
    GROUP BY 
        cl.client_id
    """
    
    print("Executing product portfolio query...")
    try:
        # Execute the query and return results
        portfolio_features = pd.read_sql(query, conn)
        print(f"Product portfolio query returned {len(portfolio_features)} rows")
        
        # Calculate loan default rate
        portfolio_features['loan_default_rate'] = portfolio_features.apply(
            lambda x: x['defaulted_loans'] / x['loan_count'] if x['loan_count'] > 0 else 0, 
            axis=1
        )
        
        # Fill missing values
        portfolio_features = portfolio_features.fillna(0)
        
        return portfolio_features
    except Exception as e:
        print(f"Error executing product portfolio query: {e}")
        print(f"Query was: {query}")
        traceback.print_exc()
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['client_id', 'total_account_count', 'loan_count', 'card_count', 
                                     'permanent_order_count', 'product_diversity_score'])

def create_product_portfolio_features(customer_data, loan_data, card_data, order_data=None):
    """
    Create features based on the banking products each customer has from dataframes
    
    Args:
        customer_data: DataFrame with client_id and account information
        loan_data: DataFrame with loan details
        card_data: DataFrame with credit card data
        order_data: DataFrame with permanent order data (optional)
        
    Returns:
        pandas.DataFrame: DataFrame with product portfolio features by client_id
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

def extract_transaction_behavior_features(conn, lookback_days=365, reference_date=None):
    """
    Analyze customer transaction behavior patterns from the database.
    Adapted for Czech language transaction types.
    
    Args:
        conn: Database connection object
        lookback_days (int): Number of days to look back for analysis
        reference_date: Reference date (default is current date in database)
        
    Returns:
        pandas.DataFrame: DataFrame with transaction behavior features
    """
    # Set reference date clause
    date_clause = f"'{reference_date.strftime('%Y-%m-%d')}'" if reference_date else "GETDATE()"
    
    print(f"Using reference date clause for transaction patterns: {date_clause}")
    
    query = f"""
    WITH TransactionBehavior AS (
        SELECT 
            d.client_id,
            
            -- Transaction type analysis - Adapted for Czech values
            SUM(CASE WHEN t.Trans_type = 'PRIJEM' THEN 1 ELSE 0 END) AS credit_transaction_count,
            SUM(CASE WHEN t.Trans_type = 'VYDAJ' THEN 1 ELSE 0 END) AS withdrawal_transaction_count,
            
            -- Transaction amounts by type
            SUM(CASE WHEN t.Trans_type = 'PRIJEM' THEN t.amount ELSE 0 END) AS credit_amount,
            SUM(CASE WHEN t.Trans_type = 'VYDAJ' THEN t.amount ELSE 0 END) AS withdrawal_amount,
            
            -- Average transaction amounts by type
            AVG(CASE WHEN t.Trans_type = 'PRIJEM' THEN t.amount ELSE NULL END) AS avg_credit_amount,
            AVG(CASE WHEN t.Trans_type = 'VYDAJ' THEN t.amount ELSE NULL END) AS avg_withdrawal_amount,
            
            -- Operations type diversity
            COUNT(DISTINCT t.operation) AS operation_type_diversity,
            COUNT(DISTINCT t.Trans_type) AS transaction_type_diversity,
            COUNT(DISTINCT t.k_symbol) AS transaction_category_diversity,
            
            -- Operation-specific counts
            SUM(CASE WHEN t.operation = 'VKLAD' THEN 1 ELSE 0 END) AS cash_deposit_count,
            SUM(CASE WHEN t.operation = 'VYBER KARTOU' THEN 1 ELSE 0 END) AS card_withdrawal_count,
            SUM(CASE WHEN t.operation = 'VYBER' THEN 1 ELSE 0 END) AS cash_withdrawal_count,
            SUM(CASE WHEN t.operation = 'PREVOD Z UCTU' THEN 1 ELSE 0 END) AS collection_from_bank_count,
            SUM(CASE WHEN t.operation = 'PREVOD NA UCET' THEN 1 ELSE 0 END) AS remittance_to_bank_count,
            
            -- Balance dynamics
            AVG(t.balance) AS avg_account_balance,
            STDEV(t.balance) AS balance_volatility,
            
            -- Temporal patterns (day of week)
            SUM(CASE WHEN DATEPART(dw, t.Trans_date) IN (1, 7) THEN 1 ELSE 0 END) AS weekend_transaction_count,
            SUM(CASE WHEN DATEPART(dw, t.Trans_date) BETWEEN 2 AND 6 THEN 1 ELSE 0 END) AS weekday_transaction_count,
            COUNT(DISTINCT DATEPART(dw, t.Trans_date)) AS transaction_day_diversity
        FROM 
            Trans t
        JOIN 
            Account a ON t.account_id = a.account_id
        JOIN 
            Disposition d ON a.account_id = d.account_id
        WHERE 
            d.type = 'OWNER'
            AND t.Trans_date <= '{reference_date.strftime('%Y-%m-%d')}'
        GROUP BY 
            d.client_id
    )
    SELECT 
        client_id,
        credit_transaction_count,
        withdrawal_transaction_count,
        credit_amount,
        withdrawal_amount,
        avg_credit_amount,
        avg_withdrawal_amount,
        operation_type_diversity,
        transaction_type_diversity,
        transaction_category_diversity,
        cash_deposit_count,
        card_withdrawal_count,
        cash_withdrawal_count,
        collection_from_bank_count,
        remittance_to_bank_count,
        avg_account_balance,
        balance_volatility,
        weekend_transaction_count,
        weekday_transaction_count,
        transaction_day_diversity,
        
        -- Derived features
        CAST(weekend_transaction_count AS FLOAT) / 
            NULLIF((weekend_transaction_count + weekday_transaction_count), 0) AS weekend_transaction_ratio,
        
        CAST(credit_transaction_count AS FLOAT) / 
            NULLIF((credit_transaction_count + withdrawal_transaction_count), 0) AS credit_transaction_ratio,
        
        CAST(credit_amount AS FLOAT) / 
            NULLIF(withdrawal_amount, 0) AS credit_withdrawal_ratio
    FROM 
        TransactionBehavior
    """
    
    print("Executing transaction behavior query...")
    try:
        # Execute the query and return results
        behavior_features = pd.read_sql(query, conn)
        print(f"Transaction behavior query returned {len(behavior_features)} rows")
        
        # Fill missing values
        behavior_features = behavior_features.fillna(0)
        
        return behavior_features
    except Exception as e:
        print(f"Error executing transaction behavior query: {e}")
        print(f"Query was: {query}")
        traceback.print_exc()
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['client_id', 'credit_transaction_count', 'withdrawal_transaction_count'])

def create_transaction_pattern_features(transaction_data, customer_data, lookback_days=None):
    """
    Create features based on transaction patterns from dataframes.
    Adapted for Czech language transaction types.
    
    Args:
        transaction_data: DataFrame with account_id, Trans_date, Trans_type, operation, amount, balance, etc.
        customer_data: DataFrame linking client_id to account_id
        lookback_days: Number of days to look back for recent transaction patterns (optional)
        
    Returns:
        pandas.DataFrame: DataFrame with transaction pattern features by client_id
    """
    # Ensure date column is datetime
    date_column = 'Trans_date' if 'Trans_date' in transaction_data.columns else 'date'
    if not pd.api.types.is_datetime64_any_dtype(transaction_data[date_column]):
        transaction_data[date_column] = pd.to_datetime(transaction_data[date_column])
    
    # Filter for recent transactions if lookback_days is specified
    if lookback_days is not None:
        cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
        transaction_data = transaction_data[transaction_data[date_column] >= cutoff_date]
    
    # Merge transaction data with customer data to get client_id
    merged_data = pd.merge(
        transaction_data,
        customer_data[['client_id', 'account_id']],
        on='account_id'
    )
    
    # Determine column names (handle both naming conventions)
    type_column = 'Trans_type' if 'Trans_type' in merged_data.columns else 'type'
    
    # Calculate transaction pattern metrics
    pattern_features = merged_data.groupby('client_id').agg({
        type_column: lambda x: x.nunique(),  # Number of unique transaction types
        'operation': lambda x: x.nunique(),  # Number of unique operations
        'k_symbol': lambda x: x.nunique()  # Number of unique transaction categories
    }).rename(columns={
        type_column: 'transaction_type_diversity',
        'operation': 'operation_diversity',
        'k_symbol': 'transaction_category_diversity'
    }).reset_index()
    
    # Credit vs withdrawal patterns - adapted for Czech transaction types
    credit_withdrawal = merged_data.groupby('client_id').apply(
        lambda x: pd.Series({
            'credit_count': sum(x[type_column] == 'PRIJEM'),
            'withdrawal_count': sum(x[type_column] == 'VYDAJ'),
            'cash_deposit_count': sum(x['operation'] == 'VKLAD'),
            'card_withdrawal_count': sum(x['operation'] == 'VYBER KARTOU'),
            'cash_withdrawal_count': sum(x['operation'] == 'VYBER'),
            'collection_from_bank_count': sum(x['operation'] == 'PREVOD Z UCTU'),
            'remittance_to_bank_count': sum(x['operation'] == 'PREVOD NA UCET'),
            'credit_amount': x[x[type_column] == 'PRIJEM']['amount'].sum(),
            'withdrawal_amount': x[x[type_column] == 'VYDAJ']['amount'].sum(),
            'avg_credit_amount': x[x[type_column] == 'PRIJEM']['amount'].mean() if any(x[type_column] == 'PRIJEM') else 0,
            'avg_withdrawal_amount': x[x[type_column] == 'VYDAJ']['amount'].mean() if any(x[type_column] == 'VYDAJ') else 0
        })
    ).reset_index()
    
    # Calculate derived transaction pattern metrics
    credit_withdrawal['credit_withdrawal_ratio'] = credit_withdrawal.apply(
        lambda x: x['credit_amount'] / x['withdrawal_amount'] if x['withdrawal_amount'] > 0 else float('inf'),
        axis=1
    )
    credit_withdrawal['credit_transaction_ratio'] = credit_withdrawal.apply(
        lambda x: x['credit_count'] / (x['credit_count'] + x['withdrawal_count']) 
        if (x['credit_count'] + x['withdrawal_count']) > 0 else 0,
        axis=1
    )
    
    # Temporal patterns
    if date_column in merged_data.columns:
        # Weekday vs weekend transaction patterns
        merged_data['day_of_week'] = merged_data[date_column].dt.dayofweek
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
            'balance': ['std', 'mean', lambda x: x.diff().abs().mean()]
        })
        balance_patterns.columns = ['balance_volatility', 'avg_account_balance', 'balance_daily_change']
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
    
    Args:
        rfm_features: DataFrame with RFM features by client_id
        product_features: DataFrame with product features by client_id
        transaction_features: DataFrame with transaction pattern features by client_id
        demographic_data: DataFrame with demographic data by district_id (optional)
        
    Returns:
        pandas.DataFrame: Combined feature set for all customers
    """
    print(f"Combining features: RFM ({len(rfm_features)} rows), Product ({len(product_features)} rows), Transaction ({len(transaction_features)} rows)")
    
    # Merge RFM and product features
    print("Merging RFM and product features...")
    combined = pd.merge(rfm_features, product_features, on='client_id', how='inner')
    print(f"After merging RFM and product features: {len(combined)} rows")
    
    # Merge transaction pattern features
    print("Merging transaction pattern features...")
    combined = pd.merge(combined, transaction_features, on='client_id', how='left')
    print(f"After merging transaction features: {len(combined)} rows")
    
    # Add demographic features if available
    if demographic_data is not None and 'district_id' in combined.columns:
        print("Merging demographic data...")
        # Join demographic data by district_id
        combined = pd.merge(combined, demographic_data, on='district_id', how='left')
        print(f"After merging demographic data: {len(combined)} rows")
    
    # Fill remaining NaN values
    print("Filling missing values...")
    combined = combined.fillna(0)
    
    # Calculate advanced banking metrics
    print("Calculating advanced banking metrics...")
    
    # 1. Customer value indicators
    combined['clv_indicator'] = combined.apply(
        lambda x: (x['monetary_value'] * 0.5) + (x['product_diversity_score'] * 0.3) - (x['recency_days'] * 0.2)
        if 'product_diversity_score' in combined.columns
        else (x['monetary_value'] * 0.6) - (x['recency_days'] * 0.4),
        axis=1
    )
    
    # 2. Churn risk indicator (higher value = higher risk)
    combined['churn_risk'] = combined.apply(
        lambda x: (x['recency_days'] * 0.6) + 
                 ((1 - min(x['frequency'], 100) / 100) * 40),
        axis=1
    )
    
    # 3. Cross-sell potential
    product_col = 'product_diversity_score' if 'product_diversity_score' in combined.columns else 'product_diversity'
    if product_col in combined.columns:
        max_products = 4  # Account, loan, card, permanent order
        combined['cross_sell_potential'] = max_products - combined[product_col]
    
    # 4. Customer engagement score
    type_div_col = 'transaction_type_diversity' if 'transaction_type_diversity' in combined.columns else 'transaction_type_diversity'
    combined['engagement_score'] = combined.apply(
        lambda x: (
            (100 - min(x['recency_days'], 100)) * 0.4 +  # More recent = better
            min(x['frequency'], 100) * 0.2 +  # More transactions = better
            min(x.get(type_div_col, 0) * 20, 100) * 0.2 +  # More diversity = better
            min(x.get(product_col, 0) * 25, 100) * 0.2  # More products = better
        ),
        axis=1
    )
    
    print(f"Feature combination complete. Final dataset has {len(combined)} rows and {len(combined.columns)} columns")
    return combined"
