# DDA 806 wallet model performance
checkpoint 1: roc.666 dda_806_wallet_model_performance
Model Performance Summary
Target: coin_return 0.3 to
ID: f07f68c4-a8e8-48ae-8ec9-d19db748b5f4
===================================
Test Samples:             159
Number of Features:       54

Test Set Classification Metrics
True Positives:             29/159 (18.2)
-----------------------------------
ROC AUC:                    0.666
Log Loss:                   0.524
Accuracy:                   0.755
Precision:                  0.321
Recall:                     0.310
F1 Score:                   0.316


# dda 607 coin model more features
checkpoint 5: c.175 include some non wallet features
Model Performance Summary
===================================
Test Samples:             458
Number of Features:       90
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.175
RMSE:                     0.387
MAE:                      0.321


checkpoint 4: c.150 winsorized 0.1
Model Performance Summary
===================================
Test Samples:             458
Number of Features:       105
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.150
RMSE:                     0.392
MAE:                      0.328



checkpoint 3: c.166 min 25 wallets
Model Performance Summary
===================================
Test Samples:             458
Number of Features:       105
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.166
RMSE:                     0.261
MAE:                      0.220


checkpoint 2: c.146 filter score segments
Model Performance Summary
===================================
Test Samples:             512
Number of Features:       105
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.146
RMSE:                     0.265
MAE:                      0.223


Model Performance Summary
===================================
Test Samples:             512
Number of Features:       126
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.135
RMSE:                     0.266
MAE:                      0.224



clusters k4
Model Performance Summary
===================================
Test Samples:             512
Number of Features:       855
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.126
RMSE:                     0.268
MAE:                      0.226


checkpoint 1: c.137 add clusters k2
Model Performance Summary
===================================
Test Samples:             512
Number of Features:       665
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.137
RMSE:                     0.266
MAE:                      0.223


# dda 623 ablation testing
it begins
Model Performance Summary
===================================
Test Samples:             583
Number of Features:       2,561
Features per Window:      0

Core Metrics
-----------------------------------
R² Score:                 0.093
RMSE:                     0.264
MAE:                      0.222
