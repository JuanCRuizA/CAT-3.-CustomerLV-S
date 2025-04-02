# Makefile for Czech Banking Customer Lifetime Value & Segmentation Project
# Usage: run 'make <target>' where <target> is one of the commands below

.PHONY: setup test lint clean import analyze report all

# Define directories
DATA_DIR = data
RAW_DIR = $(DATA_DIR)/raw
PROCESSED_DIR = $(DATA_DIR)/processed
NOTEBOOKS_DIR = notebooks
MODELS_DIR = models
DOCS_DIR = docs
REPORTS_DIR = reports

# Default target executed when no arguments are given to make
all: clean setup import analyze report

# Set up environment
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt
	mkdir -p $(PROCESSED_DIR) $(MODELS_DIR) $(REPORTS_DIR)/figures

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	mkdir -p $(PROCESSED_DIR) $(MODELS_DIR) $(REPORTS_DIR)/figures

# Import data from CSV to database
import:
	@echo "Importing data to database..."
	python -c "from src.utils.data_loading import import_all_data; import_all_data()"

# Run data validation checks
validate:
	@echo "Validating imported data..."
	python -c "from src.utils.data_validation import validate_data; validate_data()"

# Run feature engineering
features:
	@echo "Running feature engineering..."
	jupyter nbconvert --execute $(NOTEBOOKS_DIR)/04_feature_engineering.ipynb --to html --output-dir=$(REPORTS_DIR)
	@echo "Features generated in $(PROCESSED_DIR)/banking_customer_features.csv"

# Run CLV modeling
clv:
	@echo "Running CLV modeling..."
	jupyter nbconvert --execute $(NOTEBOOKS_DIR)/05_clv_modeling.ipynb --to html --output-dir=$(REPORTS_DIR)
	@echo "CLV predictions saved to $(PROCESSED_DIR)/customer_clv_predictions.csv"

# Run customer segmentation
segment:
	@echo "Running customer segmentation..."
	jupyter nbconvert --execute $(NOTEBOOKS_DIR)/06_customer_segmentation.ipynb --to html --output-dir=$(REPORTS_DIR)
	@echo "Segmentation results saved to $(PROCESSED_DIR)/customer_segments.csv"

# Run churn prediction
churn:
	@echo "Running churn prediction..."
	jupyter nbconvert --execute $(NOTEBOOKS_DIR)/07_churn_prediction.ipynb --to html --output-dir=$(REPORTS_DIR)
	@echo "Churn predictions saved to $(PROCESSED_DIR)/high_value_churn_risk.csv"

# Run the complete analysis pipeline
analyze: features clv segment churn

# Generate documentation
docs:
	@echo "Generating documentation..."
	# Extract visualization figures from notebooks for documentation
	python -c "from src.utils.documentation import extract_figures; extract_figures()"
	@echo "Documentation available in $(DOCS_DIR)"

# Generate final reports and visualizations
report:
	@echo "Generating reports..."
	python src/reporting/generate_executive_summary.py
	python src/reporting/generate_segment_report.py
	python src/reporting/generate_churn_report.py
	@echo "Reports generated in $(REPORTS_DIR)"

# Run complete pipeline on new data
refresh: clean import validate analyze report

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies and create directories"
	@echo "  make test     - Run unit tests"
	@echo "  make lint     - Check code quality"
	@echo "  make clean    - Remove generated files"
	@echo "  make import   - Import CSV data to database"
	@echo "  make validate - Validate imported data"
	@echo "  make features - Generate customer features"
	@echo "  make clv      - Run CLV modeling"
	@echo "  make segment  - Run customer segmentation"
	@echo "  make churn    - Run churn prediction"
	@echo "  make analyze  - Run complete analysis pipeline"
	@echo "  make docs     - Generate documentation"
	@echo "  make report   - Generate final reports"
	@echo "  make refresh  - Run complete pipeline on new data"
	@echo "  make all      - Run everything (clean, setup, import, analyze, report)"
	@echo "  make help     - Show this help message"