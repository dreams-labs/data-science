# 01/08/25 DDA-434 hybrid wallet address key

## checkpoint 4: full hybrid key functionality in walletmodel

### prod winsorized balanced
Model Performance Summary
===================================
Test Samples:             51,661

Core Metrics
-----------------------------------
R² Score:                 0.349
RMSE:                     0.421
MAE:                      0.292

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.746
RMSE:                     0.228
MAE:                      0.131



## checkpoint 3: after adding hybrid key functionality through transfers_df

### Prod winsorized yes 0x00000 coin wallet profits
matches checkpoint 1 rerun
Core Metrics
-----------------------------------
R² Score:                 0.347
RMSE:                     0.426
MAE:                      0.295

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.743
RMSE:                     0.230
MAE:                      0.131



Prod winsorized no 0x00000 coin wallet profits
Core Metrics
-----------------------------------
R² Score:                 0.346
RMSE:                     0.423
MAE:                      0.294

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.229
MAE:                      0.132


## checkpoint 2: after removing 0x0000 addresses from coin_wallet_transfers


### Dev Speedy: crypto_net_gain/max_investment/winsorized

Core Metrics
-----------------------------------
R² Score:                 0.267
RMSE:                     0.291
MAE:                      0.210

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.670
RMSE:                     0.149
MAE:                      0.081


### Prod Balanced: crypto_net_gain/max_investment/winsorized

Core Metrics
-----------------------------------
R² Score:                 0.354
RMSE:                     0.425
MAE:                      0.295

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.229
MAE:                      0.132


with updated coin_wallet_profits data
### Prod Balanced: crypto_net_gain/max_investment/winsorized

Core Metrics
-----------------------------------
R² Score:                 0.354
RMSE:                     0.425
MAE:                      0.295

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.229
MAE:                      0.132





### Prod Balanced: crypto_net_gain/max_investment/ntile_rank

Core Metrics
-----------------------------------
R² Score:                 0.347
RMSE:                     0.254
MAE:                      0.203

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.626
RMSE:                     0.169
MAE:                      0.118



## checkpoint 1

### Prod Balanced winsorized rerun with checkpoint 3 data

Core Metrics
-----------------------------------
R² Score:                 0.347
RMSE:                     0.426
MAE:                      0.295

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.743
RMSE:                     0.230
MAE:                      0.131



### Dev Speedy

Core Metrics
-----------------------------------
R² Score:                 0.286
RMSE:                     0.244
MAE:                      0.196

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.652
RMSE:                     0.131
MAE:                      0.075



### Prod Balanced

Core Metrics
-----------------------------------
R² Score:                 0.346
RMSE:                     0.252
MAE:                      0.201

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.628
RMSE:                     0.168
MAE:                      0.117






# 01/08/25 DDA-561 target variables

Prod Balanced

Core Metrics
-----------------------------------
R² Score:                 0.346
RMSE:                     0.252
MAE:                      0.201

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.628
RMSE:                     0.168
MAE:                      0.117
