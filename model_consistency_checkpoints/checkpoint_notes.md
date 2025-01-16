# DDA 572 coin model checkpoints

## checkpoint 1 coin model
Model Performance Summary
===================================
Test Samples:             238

Core Metrics
-----------------------------------
R² Score:                 0.146
RMSE:                     0.436
MAE:                      0.299

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.046
Standard Dev of Residuals:0.433
95% Prediction Interval:  ±0.849
## checkpoint 1 wallet model
Model Performance Summary
===================================
Test Samples:             15,485

Core Metrics
-----------------------------------
R² Score:                 0.350
RMSE:                     0.367
MAE:                      0.259

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.752
RMSE:                     0.182
MAE:                      0.107



# DDA 557 feature efficiency

## checkpoint 4 trading features refactored further

### faster learning rate
Model Performance Summary
===================================
Test Samples:             15,485

Core Metrics
-----------------------------------
R² Score:                 0.350
RMSE:                     0.367
MAE:                      0.259

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.752
RMSE:                     0.182
MAE:                      0.107

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.004
Standard Dev of Residuals:0.367
95% Prediction Interval:  ±0.720



### same model as previous
Model Performance Summary
===================================
Test Samples:             15,485

Core Metrics
-----------------------------------
R² Score:                 0.356
RMSE:                     0.366
MAE:                      0.257

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.766
RMSE:                     0.177
MAE:                      0.101


## checkpoint 3 trading features refactored
new config
Model Performance Summary
===================================
Test Samples:             15,485

Core Metrics
-----------------------------------
R² Score:                 0.356
RMSE:                     0.366
MAE:                      0.257

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.766
RMSE:                     0.177
MAE:                      0.101



old config
Model Performance Summary
===================================
Test Samples:             7,640

Core Metrics
-----------------------------------
R² Score:                 0.242
RMSE:                     0.444
MAE:                      0.316

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.775
RMSE:                     0.209
MAE:                      0.114


## checkpoint 2 partial refactor of trading
Model Performance Summary
===================================
Test Samples:             7,640

Core Metrics
-----------------------------------
R² Score:                 0.243
RMSE:                     0.443
MAE:                      0.315

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.778
RMSE:                     0.208
MAE:                      0.113



## checkpoint 1

### faster version
Model Performance Summary
===================================
Test Samples:             7,640

Core Metrics
-----------------------------------
R² Score:                 0.234
RMSE:                     0.446
MAE:                      0.317

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.791
RMSE:                     0.202
MAE:                      0.101

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.006
Standard Dev of Residuals:0.446
95% Prediction Interval:  ±0.874


### old version
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




# DDA 518 orchestrator functions

## checkpoint 1
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
