# Data Loading Diagnostic Script

import pandas as pd
import pyodbc
import sys
import os

def diagnostic_data_load():
    """
    Comprehensive diagnostic script for data loading
    """
    # Database Connection
    try:
        conn = pyodbc.connect('Driver={SQL Server};'
                             'Server=JUANCARLOSRUIZA;'
                             'Database=CzechBankingAnalysis;'
                             'Trusted_Connection=yes;')
        print("✅ Database Connection Successful")
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return

    # Table Existence and Basic Checks
    tables_to_check = [
        'Trans', 'Account', 'Client', 'Disposition', 
        'Loan', 'CreditCard', 'PermanentOrder', 'Demograph'
    ]

    # Prevent duplicate checks
    checked_tables = set()

    for table in tables_to_check:
        try:
            query = f"SELECT TOP 5 * FROM {table}"
            df = pd.read_sql(query, conn)
            print(f"✅ {table} Table: {len(df)} sample rows")
            print("   Columns:", list(df.columns))
        except Exception as e:
            print(f"❌ Problem with {table} Table: {e}")

    # Comprehensive Data Relationship Check
    try:
        comprehensive_query = """
        SELECT 
            COUNT(DISTINCT c.client_id) as total_clients,
            COUNT(DISTINCT a.account_id) as total_accounts,
            COUNT(DISTINCT t.trans_id) as total_transactions,
            MIN(t.Trans_date) as earliest_transaction,
            MAX(t.Trans_date) as latest_transaction
        FROM 
            Client c
            LEFT JOIN Disposition d ON c.client_id = d.client_id
            LEFT JOIN Account a ON d.account_id = a.account_id
            LEFT JOIN Trans t ON a.account_id = t.Account_id
        """
        summary = pd.read_sql(comprehensive_query, conn)
        print("\n📊 Data Summary:")
        print(summary.to_string(index=False))
    except Exception as e:
        print(f"❌ Comprehensive Data Check Failed: {e}")

    # Date Format Verification
    try:
        date_check_query = """
        SELECT TOP 10 
            Trans_date, 
            CONVERT(VARCHAR, Trans_date, 120) as ISO_Format,
            CONVERT(VARCHAR, Trans_date, 103) as UK_Format
        FROM Trans
        """
        date_formats = pd.read_sql(date_check_query, conn)
        print("\n📅 Date Format Verification:")
        print(date_formats)
    except Exception as e:
        print(f"❌ Date Format Check Failed: {e}")

    conn.close()

if __name__ == "__main__":
    diagnostic_data_load()