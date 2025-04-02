## Project Structure

Our Czech Banking CLV and Segmentation project follows a structured organization:

CAT-3.-CustomerLV-S/
├── data/
│   ├── raw/                  # Original CSV data files
│   │   ├── Account.csv
│   │   ├── Client.csv
│   │   ├── Credit card.csv
│   │   ├── Demograph.csv
│   │   ├── Disposition.csv
│   │   ├── Loan.csv
│   │   ├── Permanent order.csv
│   │   └── Transaction.csv
│   └── processed/            # Generated datasets
│       ├── banking_customer_features.csv
│       ├── customer_clv_predictions.csv
│       ├── customer_features_with_clv.csv
│       ├── customer_segments.csv
│       ├── cluster_profiles.csv
│       ├── diamond_customers.csv
│       ├── high_value_churn_risk.csv
│       └── segment_marketing_recommendations.csv
├── docs/                     # Documentation
│   ├── methodology.md
│   ├── data_dictionary.md
│   ├── interpretations.md
│   ├── technical_setup.md
│   └── figures/
├── models/                   # Saved models and scalers
│   ├── bg_nbd_model.pkl
│   ├── gamma_gamma_model.pkl
│   ├── clv_model_params.json
│   ├── segmentation_scaler.pkl
│   ├── kmeans_model.pkl
│   ├── churn_model.pkl
│   └── churn_prediction_scaler.pkl
├── notebooks/                # Analysis notebooks
│   ├── 01_data_overview.ipynb
│   ├── 02_customer_profiles.ipynb
│   ├── 03_transaction_patterns.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_clv_modeling.ipynb
│   ├── 06_customer_segmentation.ipynb
│   └── 07_churn_prediction.ipynb
├── SQL/                      # SQL scripts
│   ├── 01_create_tables.sql
│   ├── 02_import_data.sql
│   └── 03_validation_queries.sql
├── src/                      # Reusable Python modules
│   ├── init.py
│   ├── feature_engineering.py
│   ├── clv_modeling.py
│   ├── segmentation.py
│   └── utils/
│       ├── init.py
│       ├── data_loading.py
│       └── visualization.py
├── tests/                    # Unit and integration tests
│   ├── test_feature_engineering.py
│   ├── test_clv_modeling.py
│   ├── test_customer_segmentation.py
│   └── test_churn_prediction.py
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
└── README.md                 # Project overview


### Key Directory Functions

- **data/**: Contains both original data sources and generated datasets
  - The `raw/` subfolder houses the original Czech banking CSVs
  - The `processed/` subfolder stores intermediate and final analytical outputs

- **notebooks/**: Contains numbered Jupyter notebooks representing the analytical workflow
  - Notebooks are intended to be run in sequential order
  - Each notebook addresses a specific analytical step

- **src/**: Houses reusable Python modules that encapsulate core functionality
  - These modules are imported by notebooks and scripts
  - They implement key algorithms and data processing logic

- **models/**: Stores trained models and associated metadata
  - Both pickle files for model objects and JSON files for parameters
  - Includes scalers needed for consistent data transformation

- **tests/**: Contains unit and integration tests for validating functionality
  - Organized to mirror the structure of the `src` directory
  - Includes mock data for reproducible testing