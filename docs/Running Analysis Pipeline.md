## Running the Analysis Pipeline

To reproduce our analysis or run updated versions with new data, follow these steps:

### 1. Initial Database Setup (One-time)

Run the following SQL scripts in sequence:

```bash
# From SQL Server Management Studio, open and execute:
SQL/01_create_tables.sql
SQL/02_import_data.sql
SQL/03_validation_queries.sql

# Alternatively, from command line:
sqlcmd -S SERVERNAME -d CzechBankingAnalysis -i SQL/01_create_tables.sql
sqlcmd -S SERVERNAME -d CzechBankingAnalysis -i SQL/02_import_data.sql
sqlcmd -S SERVERNAME -d CzechBankingAnalysis -i SQL/03_validation_queries.sql

# Run the feature engineering notebook
jupyter notebook notebooks/04_feature_engineering.ipynb

# Run the CLV modeling notebook
jupyter notebook notebooks/05_clv_modeling.ipynb

# Run the segmentation notebook
jupyter notebook notebooks/06_customer_segmentation.ipynb

# Run the churn prediction notebook
jupyter notebook notebooks/07_churn_prediction.ipynb

