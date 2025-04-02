# Test the database connection and feature generation
import feature_engineering as fe

# Test with a sample connection string
try:
    conn = fe.get_database_connection(
        "Driver={SQL Server};"
        "Server=JUANCARLOSRUIZA;"
        "Database=CzechBankingAnalysis;"
        "Trusted_Connection=yes;"
    )
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")