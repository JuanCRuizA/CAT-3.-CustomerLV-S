-- Query to count the frequency of each transaction date
SELECT 
    Trans_date AS transaction_date,
    COUNT(*) AS transaction_count,
    FORMAT(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), '0.00') + '%' AS percentage
FROM 
    Trans
GROUP BY 
    Trans_date
ORDER BY 
    transaction_count DESC;