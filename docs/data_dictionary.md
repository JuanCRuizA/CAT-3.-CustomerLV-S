# Data Dictionary: Czech Banking Analysis

## Overview
This document provides definitions for all tables and fields used in the Customer Lifetime Value and Segmentation analysis.

## Tables

### Account
Contains basic information about bank accounts.

| Field | Type | Description |
|-------|------|-------------|
| account_id | INT | Primary key, unique identifier for the account |
| district_id | INT | Foreign key to district (Demograph.A1) where account was opened |
| frequency | VARCHAR(50) | Frequency of account statement issuance |
| acc_date | DATE | Date when the account was established |

### Client
Contains information about bank clients.

| Field | Type | Description |
|-------|------|-------------|
| client_id | INT | Primary key, unique identifier for the client |
| birth_number | VARCHAR(20) | Birth number (includes information about birth date and sex) |
| district_id | INT | Foreign key to district (Demograph.A1) where client lives |

### Disposition
Links clients to accounts and defines their relationship.

| Field | Type | Description |
|-------|------|-------------|
| disp_id | INT | Primary key, unique identifier for the disposition |
| client_id | INT | Foreign key to Client table |
| account_id | INT | Foreign key to Account table |
| type | VARCHAR(50) | Type of disposition (OWNER or USER) |

### Trans
Contains transaction data for accounts.

| Field | Type | Description |
|-------|------|-------------|
| Trans_id | INT | Primary key, unique identifier for the transaction |
| Account_id | INT | Foreign key to Account table |
| Trans_date | VARCHAR(10) | Transaction date |
| Trans_type | VARCHAR(10) | Type of transaction |
| Operation | VARCHAR(100) | Description of operation |
| Amount | FLOAT | Transaction amount |
| Balance | FLOAT | Balance after transaction |
| K_symbol | VARCHAR(20) | Transaction characterization |
| Bank | VARCHAR(20) | Bank of the partner |
| Account | FLOAT | Account of the partner |

[Additional tables to be documented...]

## Derived Features

### RFM Features
Features calculated for customer segmentation and CLV modeling.

| Feature | Description | Calculation |
|---------|-------------|-------------|
| recency_days | Days since customer's last transaction | Current date - max(Trans_date) |
| frequency | Number of transactions | count(Trans_id) |
| monetary_value | Total monetary value of transactions | sum(Amount) where Amount > 0 |

[Additional derived features to be documented as they are created...]

## Segment Definitions
[To be completed after segmentation model is built]