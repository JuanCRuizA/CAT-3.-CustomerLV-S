import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime, timedelta

class BankingFeatureExtractor:
    """
    A comprehensive feature extraction class for banking customer analysis.
    
    This class provides methods to extract various features from banking 
    transaction and customer data, supporting Customer Lifetime Value (CLV) 
    and customer segmentation analyses.
    """
    
    def __init__(self, connection_string):
        """
        Initialize the feature extractor with a database connection.
        
        Args:
            connection_string (str): Database connection string for SQL Server
        """
        self.connection_string = connection_string
    
    def _get_database_connection(self):
        """
        Establish a connection to the database.
        
        Returns:
            pyodbc.Connection: An active database connection
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except Exception as e:
            raise DatabaseConnectionError(f"Error connecting to database: {e}")
    
    def extract_rfm_features(self, lookback_days=365):
        """
        Calculate Recency, Frequency, and Monetary (RFM) features for customers.
        
        Args:
            lookback_days (int): Number of days to look back for analysis. 
                                 Defaults to 365 (1 year).
        
        Returns:
            pandas.DataFrame: DataFrame with RFM features for each customer
        """
        query = f"""
        WITH CustomerTransactions AS (
            SELECT 
                d.client_id,
                MAX(t.date) AS last_transaction_date,
                COUNT(t.trans_id) AS transaction_frequency,
                SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) AS total_monetary_value,
                AVG(t.amount) AS avg_transaction_amount,
                STDEV(t.amount) AS transaction_amount_volatility
            FROM 
                Transactions t
            JOIN 
                Account a ON t.account_id = a.account_id
            JOIN 
                Disposition d ON a.account_id = d.account_id
            WHERE 
                d.type = 'OWNER'
                AND t.date >= DATEADD(day, -{lookback_days}, GETDATE())
            GROUP BY 
                d.client_id
        )
        SELECT 
            client_id,
            DATEDIFF(day, last_transaction_date, '1998-01-02') AS recency_days,
            transaction_frequency AS frequency,
            total_monetary_value AS monetary_value,
            avg_transaction_amount,
            transaction_amount_volatility
        FROM 
            CustomerTransactions
        """
        
        # Execute the query and return results
        with self._get_database_connection() as conn:
            return pd.read_sql(query, conn)
    
    def extract_product_portfolio_features(self):
        """
        Extract features related to customer's product portfolio.
        
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
            
            -- Card-related features
            MAX(CASE WHEN c.type = 'credit' THEN 1 ELSE 0 END) AS has_credit_card,
            MAX(CASE WHEN c.type = 'debit' THEN 1 ELSE 0 END) AS has_debit_card
        FROM 
            Client cl
        LEFT JOIN 
            Disposition d ON cl.client_id = d.client_id
        LEFT JOIN 
            Account a ON d.account_id = a.account_id
        LEFT JOIN 
            Loan l ON a.account_id = l.account_id
        LEFT JOIN 
            Disposition d2 ON a.account_id = d2.account_id
        LEFT JOIN 
            Card c ON d2.disp_id = c.disp_id
        LEFT JOIN 
            PermanentOrder po ON a.account_id = po.account_id
        WHERE 
            d.type = 'OWNER'
        GROUP BY 
            cl.client_id
        """
        
        # Execute the query and return results
        with self._get_database_connection() as conn:
            return pd.read_sql(query, conn)
    
    def extract_transaction_behavior_features(self, lookback_days=365):
        """
        Analyze customer transaction behavior patterns.
        
        Args:
            lookback_days (int): Number of days to look back for analysis.
        
        Returns:
            pandas.DataFrame: DataFrame with transaction behavior features
        """
        query = f"""
        WITH TransactionBehavior AS (
            SELECT 
                d.client_id,
                
                -- Transaction type analysis
                SUM(CASE WHEN t.type = 'CREDIT' THEN 1 ELSE 0 END) AS credit_transaction_count,
                SUM(CASE WHEN t.type = 'WITHDRAWAL' THEN 1 ELSE 0 END) AS withdrawal_transaction_count,
                SUM(CASE WHEN t.type = 'TRANSFER' THEN 1 ELSE 0 END) AS transfer_transaction_count,
                
                -- Operation type diversity
                COUNT(DISTINCT t.operation) AS operation_type_diversity,
                
                -- Balance dynamics
                AVG(t.balance) AS avg_account_balance,
                STDEV(t.balance) AS balance_volatility,
                
                -- Temporal patterns
                SUM(CASE WHEN DATEPART(dw, t.date) IN (1, 7) THEN 1 ELSE 0 END) AS weekend_transaction_count,
                SUM(CASE WHEN DATEPART(dw, t.date) BETWEEN 2 AND 6 THEN 1 ELSE 0 END) AS weekday_transaction_count
            FROM 
                Transactions t
            JOIN 
                Account a ON t.account_id = a.account_id
            JOIN 
                Disposition d ON a.account_id = d.account_id
            WHERE 
                d.type = 'OWNER'
                AND t.date >= DATEADD(day, -{lookback_days}, GETDATE())
            GROUP BY 
                d.client_id
        )
        SELECT 
            client_id,
            credit_transaction_count,
            withdrawal_transaction_count,
            transfer_transaction_count,
            operation_type_diversity,
            avg_account_balance,
            balance_volatility,
            weekend_transaction_count,
            weekday_transaction_count,
            
            -- Derived features
            CAST(weekend_transaction_count AS FLOAT) / 
                NULLIF((weekend_transaction_count + weekday_transaction_count), 0) AS weekend_transaction_ratio,
            
            CAST(credit_transaction_count AS FLOAT) / 
                NULLIF((credit_transaction_count + withdrawal_transaction_count + transfer_transaction_count), 0) AS credit_transaction_ratio
        FROM 
            TransactionBehavior
        """
        
        # Execute the query and return results
        with self._get_database_connection() as conn:
            return pd.read_sql(query, conn)
    
    def generate_comprehensive_features(self):
        """
        Combine all feature extraction methods into a comprehensive feature set.
        
        Returns:
            pandas.DataFrame: Merged DataFrame with all extracted features
        """
        # Extract features from different methods
        rfm_features = self.extract_rfm_features()
        product_features = self.extract_product_portfolio_features()
        transaction_features = self.extract_transaction_behavior_features()
        
        # Merge features using client_id as the key
        comprehensive_features = rfm_features.merge(
            product_features, on='client_id', how='outer'
        ).merge(
            transaction_features, on='client_id', how='outer'
        )
        
        # Fill any missing values with appropriate defaults
        comprehensive_features.fillna({
            'recency_days': comprehensive_features['recency_days'].max(),
            'frequency': 0,
            'monetary_value': 0,
            'total_account_count': 0,
            'product_diversity_score': 0
        }, inplace=True)
        
        return comprehensive_features

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass

def main():
    """
    Example usage of the BankingFeatureExtractor class.
    
    Note: Replace with actual connection string when using.
    """
    # Example connection string (replace with actual credentials)
    connection_string = (
        "Driver={SQL Server};"
        "Server=SERVERNAME;"
        "Database=CzechBankingAnalysis;"
        "Trusted_Connection=yes;"
    )
    
    # Initialize the feature extractor
    feature_extractor = BankingFeatureExtractor(connection_string)
    
    try:
        # Generate comprehensive features
        comprehensive_features = feature_extractor.generate_comprehensive_features()
        
        # Save to CSV for further analysis
        comprehensive_features.to_csv('banking_customer_features.csv', index=False)
        
        print("Features successfully extracted and saved.")
        print(f"Total customers processed: {len(comprehensive_features)}")
        print("\nFeature summary:")
        print(comprehensive_features.describe())
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()