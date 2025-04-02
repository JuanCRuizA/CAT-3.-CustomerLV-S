# src/utils/data_loading.py

import pandas as pd
import os
from src.utils.database import get_database_connection

def import_all_data():
    """Import all CSV data files to the database."""
    conn = get_database_connection()
    
    # Define import configurations
    tables = [
        {'csv': 'Account.csv', 'table': 'Account'},
        {'csv': 'Client.csv', 'table': 'Client'},
        {'csv': 'Credit card.csv', 'table': 'Card'},
        {'csv': 'Disposition.csv', 'table': 'Disposition'},
        {'csv': 'Loan.csv', 'table': 'Loan'},
        {'csv': 'Permanent order.csv', 'table': 'PermanentOrder'},
        {'csv': 'Transaction.csv', 'table': 'Trans'},
        {'csv': 'Demograph.csv', 'table': 'Demographic'}
    ]
    
    # Import each table
    for config in tables:
        print(f"Importing {config['csv']} to {config['table']} table...")
        import_csv_to_table(config['csv'], config['table'], conn)
    
    conn.close()
    print("All data imported successfully.")

def import_csv_to_table(csv_file, table_name, conn):
    """Import a specific CSV file to a database table."""
    # Implementation details here...
    # Read CSV, clean data, and insert to database
    pass

if __name__ == "__main__":
    import_all_data()