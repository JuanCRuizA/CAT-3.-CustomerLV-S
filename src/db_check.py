import pyodbc
import pandas as pd

# Connection string
connection_string = (
    "Driver={SQL Server};"
    "Server=JUANCARLOSRUIZA;"
    "Database=CzechBankingAnalysis;"
    "Trusted_Connection=yes;"
)

# Connect to database
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# Check which tables exist
cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"- {table[0]}")

# Count rows in each table
for table_name in [row[0] for row in tables]:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        count = cursor.fetchone()[0]
        print(f"Table {table_name}: {count} rows")
    except Exception as e:
        print(f"Error counting rows in {table_name}: {e}")

# Test a simple join to make sure relationships work
try:
    query = """
    SELECT COUNT(*) 
    FROM Trans t
    JOIN Account a ON t.account_id = a.account_id
    JOIN Disposition d ON a.account_id = d.account_id
    WHERE d.type = 'OWNER'
    """
    cursor.execute(query)
    join_count = cursor.fetchone()[0]
    print(f"\nJoin test: {join_count} transactions found with valid account and disposition")
except Exception as e:
    print(f"Join test failed: {e}")

conn.close()