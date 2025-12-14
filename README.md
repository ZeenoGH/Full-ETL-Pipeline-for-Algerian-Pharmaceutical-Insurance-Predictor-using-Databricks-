# Algerian Pharmaceutical Market Analysis & Insurance Prediction System

End-to-end data pipeline built on Databricks for analyzing the Algerian pharmaceutical market and predicting insurance reimbursement.

## What's included
- API scraping of 2,908 medications from Algerian government health portal
- Full ETL pipeline using PySpark for data cleaning and transformation
- Delta Lake tables for reliable storage and versioning
- SQL analytics dashboard with pricing trends, manufacturer market share, and therapeutic categories
- ML model (Random Forest) predicting insurance coverage with 85-90% accuracy
- Streamlit deployment for real-time predictions

## Tech stack
Databricks (PySpark, Delta Lake), Python requests for API ingestion, SQL for analytics, scikit-learn for ML with class weighting to handle 6:1 imbalance, Streamlit for frontend.

## How it works
Data gets pulled via REST API, cleaned in Spark (handling nulls, duplicates, encoding), saved to Delta tables, analyzed with SQL queries, then fed into a trained Random Forest model that accounts for price, form, therapeutic class, and manufacturer to predict coverage.

