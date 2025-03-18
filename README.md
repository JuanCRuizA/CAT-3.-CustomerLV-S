# Customer Lifetime Value (CLV) Analysis and Customer Segmentation Project 

## Current status
This project is in development

## Overview 
Machine learning project to analyse and estimate the Customer Lifetime Value (CLV) and propose Customer Segmentation using historical banking data from a Czech financial dataset. 

This database is a Real anonymized Czech bank set of tables about transactions, account info, and loan records released for PKDD'99 Discovery Challenge. It was taken from this URL: https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions. There, Liz Petrocelli acknowledged that this database was prepared by Petr Berka and Marta Sochorova and the original source is http://lisp.vse.cz/pkdd99/berka.htm. 
 
The database includes 8 primary tables with established foreign key relationships as shown in the ER diagram. There are 4 central tables (it means they have a relationship to 2 to 5 tables) and 4 normal tables (they only have a relationship to 1 table).
 
These are the central tables: The first one is Account (4,500 records and related to 5 tables), their features are: ID, location of the branch (district_id), frequency of issuance of statements, and account’s date of creation (between Jan/1993 and Dec/1997). Another important table is Disposition (5.369 records and related to 3 tables), their features are: ID, identification of a customer (client_id) and an account (account_id) and the type or disposition (owner/user). Next key table is Client (5.369 records), their features are: ID, client’s birthday and sex and district of the client. Last central table is Demograph (77 records), their most important features are ID (A1), district name (A2), number of inhabitants (A4), average salary (A11) and no. of entrepreneurs per 1000 inhabitants (A14).

These are the normal tables: The first one is Transactions (1,056,320 records), their most important features are ID, the account the transaction deals with, date and type of transaction, operation, and amount of money. Another table is CreditCard (892 records), their features are: ID, disposition ID, type and issued date. Next table is Permanent Order (6,471 records), their features are ID, account ID, bank and account which the operation goes, type and amount. The last, and one very salient table, is Loan (682 records), their features are ID, account, date, amount, duration, payments and status.

The Entity-Relationship diagram is in CAT3.CustomerLVS\SQL\diagrams\CzechDatabaseERdiagram.png

## Project Structure 
`src/`: Source code for data preprocessing and model development 
`data/`: Dataset documentation and information 
`docs/`: Project documentation 
`tests/`: Unit tests 

## Features
1.	Customer segmentation - The process of dividing customers into distinct groups based on shared characteristics
2.	Churn prediction - Using machine learning to identify customers likely to leave the bank. 
3.	Customer Lifetime Value (CLV) - A prediction of the total value a customer will bring to the bank over their entire relationship. 
4.	Cross-selling - Using data analytics to identify which additional products to offer existing customers. 
5.	Propensity modeling - Building predictive models to calculate how likely a customer is to: Buy specific products, respond to marketing campaigns, use certain banking channels or become a high-value customer
6.	Customer journey - Analyzing the entire sequence of interactions between a customer and the bank.

## Technical Stack
- Python 3.x
- Libraries: pandas, numpy, scikit-learn
- Jupyter Notebooks/Google Colab

## Results
(pending)


## Installation & Usage
(pending)


## Future Improvements
(pending)
