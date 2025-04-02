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

### Segmentation Methodology
Our clustering approach (implemented in `06_customer_segmentation.ipynb`) employs K-means clustering with careful feature selection and evaluation:

### Feature Selection Process

We selected features across five key banking dimensions:
1. **RFM Features**: Recency, frequency, and monetary value metrics
2. **Transaction Features**: Transaction amounts, volatility, patterns
3. **Balance Features**: Average balance, balance range, volatility
4. **Product Features**: Product diversity, loan usage, card adoption
5. **CLV Features**: Forward-looking value predictions

### Optimal Cluster Determination

To identify the optimal number of clusters, we computed and evaluated four clustering quality metrics:
- Within-Cluster Sum of Squares (Elbow method)
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

These metrics were normalized and combined into an aggregate score for each potential cluster count (2-10). While our analysis identified {X} clusters as mathematically optimal, we selected 5 clusters based on banking industry standard practices and interpretability.

### Segment Naming Methodology

We implemented an automatic naming system that creates banking-specific segment names combining:
- **Value tier**: Premium, Standard, or Basic
- **Activity level**: Active, Intermittent, Dormant, or Low-Activity
- **Product focus**: Borrowers, Transactors, Automated Users, Savers, or Basic Users

This naming convention creates intuitive, business-friendly segment labels like "Premium Active Borrowers" that immediately convey key characteristics.


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