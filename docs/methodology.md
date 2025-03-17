# Methodology: Customer Lifetime Value and Segmentation

## Overview
This document outlines the methodological approach used for calculating Customer Lifetime Value (CLV) and developing customer segmentation for the Czech Banking dataset.

## Data Sources
The analysis uses the following tables from the Czech Banking dataset:
- Transactions (1,056,320 records)
- Account (4,500 records)
- Client (5,369 records)
- Disposition (5,369 records)
- Loan (682 records)
- CreditCard (892 records)
- PermanentOrder (6,471 records)
- Demograph (77 records)

## Customer Lifetime Value Calculation

### Approach
We use the BG/NBD (Beta-Geometric/Negative Binomial Distribution) model combined with the Gamma-Gamma model to calculate customer lifetime value.

1. **BG/NBD Model**: Predicts the number of future transactions a customer will make
2. **Gamma-Gamma Model**: Estimates how much value each of those transactions will generate

### Key Parameters
- **Time horizon**: 1 year forward prediction period
- **Discount rate**: 15% annual rate
- **RFM features**: Recency (days since last transaction), Frequency (number of transactions), Monetary Value (average transaction value)

### Implementation
The CLV calculation follows these steps:
1. Calculate RFM metrics for each customer
2. Fit the BG/NBD model to predict future transaction frequency
3. Fit the Gamma-Gamma model to predict future transaction values
4. Combine these predictions with a discount rate to calculate CLV

## Customer Segmentation

### Approach
We use K-means clustering with 5 clusters to segment customers based on their behavioral and value characteristics.

### Segmentation Features
- **Transaction patterns**: Frequency, recency, monetary value
- **Product portfolio**: Account types, loan usage, credit card ownership
- **Customer value**: CLV and profitability metrics
- **Demographic factors**: Regional information when available

### Validation
Segmentation quality is assessed using:
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index
- Business interpretability of segments

## Churn Prediction

### Definition
A customer is considered "churned" if they have not had any transaction activity in the past 90 days.

### Approach
We use a Random Forest classifier to predict customer churn probability based on:
- Transaction frequency declines
- Balance changes
- Product usage patterns
- Customer demographics

### Evaluation
The churn model is evaluated using:
- Precision and recall (with emphasis on high-value customer recall)
- ROC AUC score
- Confusion matrix analysis
- Feature importance analysis

## Implementation Details
[To be completed as the project progresses]