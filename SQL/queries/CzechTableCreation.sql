CREATE TABLE Account(
    account_id INT PRIMARY KEY,
    district_id INT NOT NULL,
    frequency VARCHAR(50) NOT NULL,
    acc_date DATE NOT NULL
);

CREATE TABLE Client(
    client_id INT PRIMARY KEY,
    birth_number VARCHAR(20) NOT NULL,
    district_id INT NOT NULL,
);

CREATE TABLE dbo.Demograph (
    A1 INT PRIMARY KEY,
    A2 NVARCHAR(100),
    A3 NVARCHAR(70),    -- Ensure this is NVARCHAR, not INT or FLOAT
    A4 FLOAT,
    A5 FLOAT,
    A6 INT,
    A7 INT,
    A8 INT,
    A9 INT,
    A10 FLOAT,
    A11 INT,
    A12 FLOAT,
    A13 FLOAT,
    A14 INT,
    A15 INT,
    A16 INT
);

CREATE TABLE dbo.PermanentOrder (
    Order_id INT PRIMARY KEY,
    Account_id INT,
    Bank_to NVARCHAR(10),    
    Account_to FLOAT,
    Amount FLOAT,
    K_symbol NVARCHAR(20),
);

CREATE TABLE dbo.Loan (
    Loan_id INT PRIMARY KEY,
    Account_id INT,
    Loan_date NVARCHAR(10),    
    Amount FLOAT,
    Duration INT,
    Payments FLOAT,
    Loan_status NVARCHAR(1),
);

CREATE TABLE dbo.Trans (
    Trans_id INT PRIMARY KEY,
    Account_id INT,
    Trans_date NVARCHAR(10),    
    Trans_type NVARCHAR(10),
    Operation NVARCHAR(100),
    Amount FLOAT,
    Balance FLOAT,
    K_symbol NVARCHAR(20),
    Bank NVARCHAR(20),
    Account FLOAT,
);