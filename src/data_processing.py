"""
Data loading and processing functions for Czech banking data.
"""
import pandas as pd
import numpy as np
import pyodbc

def connect_to_db():
    """Establish connection to SQL Server database"""
    conn = pyodbc.connect('Driver={SQL Server};'
                         'Server=JUANCARLOSRUIZA;'
                         'Database=CzechBankingAnalysis;'
                         'Trusted_Connection=yes;')
    return conn

def load_transactions_data(start_date=None, end_date=None):
    """
    Load transaction data with optional date filtering
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        Transactions data
    """
    conn = connect_to_db()
    
    query = """
    SELECT t.*, a.district_id 
    FROM dbo.Trans t
    JOIN dbo.Account a ON t.Account_id = a.account_id
    """
    
    if start_date or end_date:
        query += " WHERE "
        if start_date:
            query += f"t.Trans_date >= '{start_date}'"
        if start_date and end_date:
            query += " AND "
        if end_date:
            query += f"t.Trans_date <= '{end_date}'"
            
    return pd.read_sql(query, conn)

def load_customer_data():
    """Load customer data with relevant account information"""
    conn = connect_to_db()
    
    query = """
    SELECT c.client_id, c.birth_number, c.district_id,
           d.type as disposition_type, a.account_id, 
           a.frequency, a.acc_date
    FROM dbo.Client c
    JOIN dbo.Disposition d ON c.client_id = d.client_id
    JOIN dbo.Account a ON d.account_id = a.account_id
    """
    
    return pd.read_sql(query, conn)

def clean_transaction_data(df):
    """
    Clean transaction data by handling missing values and converting types
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned transaction data
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert transaction date to datetime
    # Assuming Trans_date is in format 'YYMMDD'
    # Implement your date conversion logic here
    
    # Convert amounts to numeric, handling any errors
    numeric_cols = ['Amount', 'Balance']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows with missing values in critical columns
    critical_cols = ['Trans_id', 'Account_id', 'Amount']
    df_clean = df_clean.dropna(subset=critical_cols)
    
    return df_clean