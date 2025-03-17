BULK INSERT dbo.Client
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Client.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);

BULK INSERT dbo.PermanentOrder
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\PermanentOrder.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);


BULK INSERT dbo.Loan
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Loan.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);

BULK INSERT dbo.Trans
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Trans.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);


INSERT INTO dbo.Transactions
SELECT * FROM OPENROWSET(
    BULK 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Transactions.csv',
    FORMATFILE = 'FORMAT = ''CSV'', FIRSTROW = 2, FIELDTERMINATOR = '';'', ROWTERMINATOR = ''\n'''
) AS source;

SELECT TOP 10 * FROM dbo.Client;
SELECT TOP 10 * FROM dbo.Account;
SELECT TOP 10 * FROM dbo.Disposition;
SELECT TOP 20 * FROM dbo.Demograph;
SELECT TOP 20 * FROM dbo.PermanentOrder;
SELECT TOP 15 * FROM dbo.Loan;
SELECT TOP 100 * FROM dbo.Trans;
SELECT COUNT(*) FROM dbo.Trans;

SELECT COUNT(*)
FROM Trans;

DROP TABLE dbo.Demograph;
DROP TABLE dbo.PermanentOrder;
DROP TABLE dbo.Transactions;


ALTER TABLE Disposition
ALTER COLUMN client_id INT;


ALTER TABLE dbo.Disposition
ADD CONSTRAINT FK_Disposition_Client 
FOREIGN KEY (client_id) 
REFERENCES dbo.Client(client_id);

ALTER TABLE Disposition
ALTER COLUMN account_id INT;

ALTER TABLE dbo.Disposition
ADD CONSTRAINT FK_Disposition_Account 
FOREIGN KEY (account_id) 
REFERENCES dbo.Account(account_id);

ALTER TABLE dbo.PermanentOrder
ADD CONSTRAINT FK_PermanentOrder_Account
FOREIGN KEY (Account_id)
REFERENCES dbo.Account(account_id);

ALTER TABLE dbo.Trans
ADD CONSTRAINT FK_Trans_Account
FOREIGN KEY (Account_id)
REFERENCES dbo.Account (account_id);

ALTER TABLE dbo.CreditCard
ADD CONSTRAINT FK_CreditCard_Disposition
FOREIGN KEY (disp_id)
REFERENCES dbo.Disposition(disp_id)

ALTER TABLE dbo.Account
ADD CONSTRAINT FK_Account_Demograph
FOREIGN KEY (district_id)
REFERENCES dbo.Demograph(A1)
