# data-science/training_data

Core data extraction and preprocessing pipeline for the primary wallet_modeling primary model. Retrieves raw datasets from BigQuery and ensures data completeness through multithreaded row imputation.

### Critical Data Consistency Feature

The profits row imputation system ensures every coin-wallet pair has records at period boundaries (training start/end, modeling start/end). This is essential because the base profits_df only includes records on dates with transactions, and the imputation allows us to see the impact of price changes as a result of holding behavior related to opening and closing balances, even if the wallet did not actually transact on period start or end dates.

## Key Modules

### `data_retrieval.py`
- `retrieve_market_data()` - Extracts coin market data from BigQuery with memory optimization
- `retrieve_profits_data()` - Retrieves wallet transaction history with cohort filtering
- `retrieve_macro_trends_data()` - Pulls Bitcoin indicators and Google Trends data
- `clean_market_data()` - Applies data quality filters and removes problematic coins
- `clean_profits_df()` - Filters outlier wallets and validates data integrity

### `profits_row_imputation.py`
- `impute_profits_for_multiple_dates()` - Main interface for batch row imputation
- `multithreaded_impute_profits_rows()` - Coordinates parallel processing across coin partitions
- `impute_profits_df_rows()` - Vectorized imputation logic without Python loops
- `calculate_new_profits_values()` - Computes financial metrics using price movements

