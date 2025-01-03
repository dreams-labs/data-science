# Modeling Strategy

## Overview

This project aims to predict cryptocurrency price movements using wallet transaction data. To ensure the model generalizes well to unseen data, we’ve structured the data into distinct time periods and evaluation sets.

## Dataset Structure

Data will be segmented into four sets:
* **Training set:** used to train the model
* **Validation set:** used to tune model parameters
* **Test set:** used to assess whether model generalizes within the same time period as the Training and Validation sets
* **Future set:** used to assess whether the model generalizes within a different time period from the Training, Validation, and Test sets.

## Time Periods

It will be very important to strictly segment price data into different periods, because prices are used for feature engineering, model construction, model validation, and model generalization. If price movements from one of these processes were also included in another, data leakaage would contaminate model performance and likely render it unable to generalize its performance.

As such, price movements will be split into three periods:

1. **Training Period** (e.g. price movements up to 3/1/24)
   - Price movements during this period will be used for feature engineering. This ensures that the model only relies on information that was available at the time of the prediction.

2. **Modeling Period** (e.g. 3/1/24–5/1/24):
   - Price movements during this period will generate the target variables in the Training, Validation, and Test sets.
   - Price movements during this period will also be used to build features for the Future set since it will have access to all transactions up until the start of the Future period.

3. **Future Period** (e.g. 5/1/24-7/1/24):
   - Price movements during this period will generate the target variables in the Future set to assess whether the model can generalize beyond the original time frame, particularly in future market conditions that may be significantly different from the past.

### Evaluation Strategy

- **Validation and Test Sets**: These sets assess whether the model can generalize within the training period. Strong performance here would indicate that the model is able to learn meaningful patterns without overfitting.
- **Future Set**: Performance on this set will determine if the model can generalize to other time periods, particularly when market conditions change.


# Testing

Tests are built in the tests/ directory and configured with pytest.ini.

## Running Tests

Tests can be initiated from the main data-science directory with the standard command >`pytest`.

There is a `integration` flag applied to slow tests that involve large dataframes or queries. To ignore these but run other tests, use the command >`pytest -m "not integration"`.


## Key Dataframes

### profits_df

#### Summary
Shows the transaction history of each wallet-coin pair over a given period of time.
Unique on: coin_id-wallet_address-date
Size: up to hundreds of millions of rows

#### Columns:
coin_id                         category
date                      datetime64[ns]
wallet_address                     int32
profits_cumulative               float32
usd_balance                      float32  # reflects ending balance as of that date
usd_net_transfers                float32  # token transfers * current price USD
usd_inflows                      float32
usd_inflows_cumulative           float32
is_imputed                       boolean  # whether the record was imputed on the start or end date. if True, usd_net_transfers is always 0

#### Imputation logic and data consistency
Imputation logic exists to ensure period start and end dates reflect the current prices as of those dates. When transitioning from imputed to actual records, balances must exceed net transfers. Period boundaries must align exactly, with wallet-coin pair balances matching within 0.0001% tolerance at transition points. All USD balances must remain positive throughout the period.

The relavant dates for the given period are:
* period_start_date: the date on which the period starts
* period_end_date: the period on which the period ends
* starting_balance_date: the day before the period_start_date

Imputations for starting_balance_date
All coin-wallet pairs with an existing token balance going into the period will have a row imputed on the starting_balance_date. These rows will have is_imputed==True, usd_balance>0, usd_net_transfers==0, usd_inflows==0. The purpose of these rows is to show each coin-wallet pair's starting balances as of the end of the day prior to period start. The transfers are set to 0 because the transfer activity happens prior to the period_start_date, but the balance is needed because it represents the opening balance on the period_start_date. The imputed records usd_balance reflects the token balance * the current token price as of EOD starting_balance_date. If a coin-wallet pair has no balance as of the starting_balance_date, no row will be imputed for the pair.

Imputations for period_end_date
All coin-wallet pairs with an existing token balance as of the period_end_date must have a row in profits_df to show their unrealized gains based on the prices as of the period_end_date. If a coin-wallet pair has an observed transfer on the date, then that record will have is_imputed==False and a usd_balance that accurately reflects the remaining balance at current prices. If a coin-wallet pair does not have a transfer on the date, then a row will be imputed where is_imputed==True, usd_balance > 0, usd_net_transfers == 0. The purpose of these rows is to show unrealized gains for the wallet-coin pair at the prices as of the period_end_date.

Consistency between periods
The imputation logic is designed in such a way that the last balance in the training_profits_df matches the first balance in the modeling_profits_df exactly, and that the sum of all usd_net_transfers in both periods adds up to the full total of transfers between the training start date and modeling end date. Each profits_df fits exactly next to each other temporally to ensure that all records are included in the analysis.


### market_data_df