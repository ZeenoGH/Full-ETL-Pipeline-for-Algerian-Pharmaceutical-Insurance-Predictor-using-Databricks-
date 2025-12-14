-- ============================================================
-- Algerian Pharmaceutical Market Analysis - SQL Queries
-- Dashboard Analytics Queries
-- ============================================================

-- Query 1: Market Overview - Key Performance Indicators
-- Purpose: Executive summary with total medications, manufacturers, and price statistics
SELECT 
    COUNT(*) as total_medications,
    COUNT(DISTINCT lab_name) as manufacturers,
    COUNT(DISTINCT therapeutic_class) as disease_categories,
    ROUND(AVG(price), 0) as avg_price_DA,
    ROUND(PERCENTILE(price, 0.5), 0) as median_price_DA,
    MAX(price) as most_expensive_DA
FROM medications;


-- Query 2: Price Distribution by Disease Area
-- Purpose: Identify which disease categories have the most expensive treatments
SELECT 
    therapeutic_class as disease_area,
    COUNT(*) as drug_count,
    ROUND(AVG(price), 0) as avg_price_DA,
    ROUND(MIN(price), 0) as min_price_DA,
    ROUND(MAX(price), 0) as max_price_DA
FROM medications
WHERE therapeutic_class IS NOT NULL
GROUP BY therapeutic_class
HAVING COUNT(*) >= 30
ORDER BY avg_price_DA DESC
LIMIT 12;


-- Query 3: Top Manufacturers - Market Dominance
-- Purpose: Show pharmaceutical companies with highest market presence
SELECT 
    lab_name as manufacturer,
    COUNT(*) as products,
    ROUND(AVG(price), 0) as avg_price_DA,
    COUNT(DISTINCT therapeutic_class) as disease_areas
FROM medications
WHERE lab_name IS NOT NULL
GROUP BY lab_name
ORDER BY products DESC
LIMIT 10;


-- Query 4: Market Segmentation by Price Category
-- Purpose: Understand affordable vs premium medication distribution
SELECT 
    price_category,
    COUNT(*) as medications_count,
    ROUND(AVG(price), 0) as avg_price_DA,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM medications), 1) as market_pct
FROM medications
GROUP BY price_category
ORDER BY 
    CASE price_category 
        WHEN 'Low' THEN 1 
        WHEN 'Medium' THEN 2 
        WHEN 'High' THEN 3 
    END;


-- Query 5: Medication Forms Distribution
-- Purpose: Most common dosage forms in the market
SELECT 
    form,
    COUNT(*) as count,
    ROUND(AVG(price), 0) as avg_price_DA,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM medications WHERE form IS NOT NULL), 1) as market_pct
FROM medications
WHERE form IS NOT NULL
GROUP BY form
ORDER BY count DESC
LIMIT 8;


-- Query 6: Top 15 Most Expensive Medications
-- Purpose: Identify premium medications and their characteristics
SELECT 
    name,
    therapeutic_class as disease_area,
    lab_name as manufacturer,
    price as price_DA,
    form
FROM medications
WHERE price IS NOT NULL
ORDER BY price DESC
LIMIT 15;


-- Query 7: Affordability Metrics
-- Purpose: Calculate percentage of affordable medications
SELECT 
    ROUND(COUNT(CASE WHEN price <= 100 THEN 1 END) * 100.0 / COUNT(*), 1) as affordable_drugs_pct,
    ROUND(COUNT(CASE WHEN price > 500 THEN 1 END) * 100.0 / COUNT(*), 1) as premium_drugs_pct,
    ROUND(AVG(CASE WHEN price <= 100 THEN price END), 0) as avg_affordable_price_DA
FROM medications
WHERE price IS NOT NULL;


-- Query 8: Disease Coverage Statistics
-- Purpose: Number of disease areas covered and drugs per area
SELECT 
    COUNT(DISTINCT therapeutic_class) as disease_areas_covered,
    ROUND(AVG(drugs_per_area), 0) as avg_drugs_per_disease_area
FROM (
    SELECT therapeutic_class, COUNT(*) as drugs_per_area
    FROM medications
    WHERE therapeutic_class IS NOT NULL
    GROUP BY therapeutic_class
);


-- Query 9: Manufacturer Size Analysis
-- Purpose: Compare pricing strategies by manufacturer size
SELECT 
    CASE 
        WHEN product_count >= 100 THEN 'Large (100+ products)'
        WHEN product_count >= 50 THEN 'Medium (50-99 products)'
        ELSE 'Small (<50 products)'
    END as manufacturer_size,
    COUNT(DISTINCT lab_name) as manufacturers,
    ROUND(AVG(avg_price), 0) as avg_price_DA,
    SUM(product_count) as total_products
FROM (
    SELECT 
        lab_name,
        COUNT(*) as product_count,
        AVG(price) as avg_price
    FROM medications
    WHERE lab_name IS NOT NULL AND price IS NOT NULL
    GROUP BY lab_name
)
GROUP BY 
    CASE 
        WHEN product_count >= 100 THEN 'Large (100+ products)'
        WHEN product_count >= 50 THEN 'Medium (50-99 products)'
        ELSE 'Small (<50 products)'
    END
ORDER BY avg_price_DA DESC;


-- Query 10: Top Active Ingredients
-- Purpose: Most commonly used generic drug substances
SELECT 
    generic as active_ingredient,
    COUNT(*) as formulations,
    ROUND(AVG(price), 0) as avg_price_DA,
    COUNT(DISTINCT lab_name) as manufacturers
FROM medications
WHERE generic IS NOT NULL AND generic != ''
GROUP BY generic
ORDER BY formulations DESC
LIMIT 10;


-- Query 11: Price Range Distribution
-- Purpose: Histogram-style distribution of medication prices
SELECT 
    CASE 
        WHEN price <= 50 THEN '0-50 DA'
        WHEN price <= 100 THEN '51-100 DA'
        WHEN price <= 200 THEN '101-200 DA'
        WHEN price <= 500 THEN '201-500 DA'
        WHEN price <= 1000 THEN '501-1000 DA'
        ELSE '1000+ DA'
    END as price_range,
    COUNT(*) as medications
FROM medications
WHERE price IS NOT NULL
GROUP BY 
    CASE 
        WHEN price <= 50 THEN '0-50 DA'
        WHEN price <= 100 THEN '51-100 DA'
        WHEN price <= 200 THEN '101-200 DA'
        WHEN price <= 500 THEN '201-500 DA'
        WHEN price <= 1000 THEN '501-1000 DA'
        ELSE '1000+ DA'
    END
ORDER BY MIN(price);


-- Query 12: Insurance Reimbursement Statistics
-- Purpose: Analyze reimbursable vs non-reimbursable medications
SELECT 
    CASE 
        WHEN refundable = true THEN 'Covered by Insurance'
        WHEN refundable = false THEN 'Not Covered'
        ELSE 'Unknown'
    END as coverage_status,
    COUNT(*) as medications_count,
    ROUND(AVG(price), 0) as avg_price_DA,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM medications), 1) as percentage
FROM medications
GROUP BY 
    CASE 
        WHEN refundable = true THEN 'Covered by Insurance'
        WHEN refundable = false THEN 'Not Covered'
        ELSE 'Unknown'
    END
ORDER BY medications_count DESC;


-- ============================================================
-- ML Feature Engineering Queries
-- ============================================================

-- Query 13: ML Dataset Preparation
-- Purpose: Clean dataset for machine learning model training
SELECT 
    price,
    form,
    therapeutic_class,
    pharmacological_class,
    lab_name,
    CASE 
        WHEN refundable IS NULL THEN false
        ELSE refundable
    END as refundable_label
FROM medications
WHERE 
    price IS NOT NULL 
    AND form IS NOT NULL 
    AND therapeutic_class IS NOT NULL 
    AND pharmacological_class IS NOT NULL 
    AND lab_name IS NOT NULL;


-- Query 14: Class Imbalance Check
-- Purpose: Verify target variable distribution for ML
SELECT 
    CASE 
        WHEN refundable = true THEN 'Covered'
        ELSE 'Not Covered'
    END as label,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM medications), 1) as percentage
FROM medications
GROUP BY 
    CASE 
        WHEN refundable = true THEN 'Covered'
        ELSE 'Not Covered'
    END;


-- ============================================================
-- End of SQL Queries
-- Total Queries: 14
-- ============================================================
