-- Adjust the file path to match your CSV location
BULK INSERT dbo.Account
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Account.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);

BULK INSERT dbo.Demograph
FROM 'C:\Users\carlo\Documents\4.DS\CAT3.CustomerLVS\data\Demograph.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,           -- Skip header row if present
    FIELDTERMINATOR = ';',  -- Adjust based on your CSV format
    ROWTERMINATOR = '\n',
    TABLOCK
);