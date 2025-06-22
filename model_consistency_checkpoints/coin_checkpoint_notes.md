# DDA 858 wallet segments
checkpoint 1: lifetime 120% best model
score_threshold	lifetime_return_all	lifetime_return_above_threshold	total_coins_above_threshold
0	0.05	-0.645920932293	-0.811502933502	4172
1	0.1	-0.645920932293	-0.703640341759	2449
2	0.15	-0.645920932293	-0.730792820454	1382
3	0.2	-0.645920932293	-0.723976969719	722
4	0.25	-0.645920932293	-0.949538111687	406
5	0.3	-0.645920932293	-0.972379267216	214
6	0.35	-0.645920932293	-0.928154945374	126
7	0.4	-0.645920932293	-0.809118688107	69
8	0.45	-0.645920932293	-0.738984286785	41
9	0.5	-0.645920932293	0.260417699814	19
10	0.55	-0.645920932293	1.2066757679	8
11	0.6	-0.645920932293	1.1453461647	6
12	0.65	-0.645920932293	0.426909446716	4
13	0.7	-0.645920932293	0.37182700634	2
14	0.75	-0.645920932293	0.0612534284592	1
15	0.8	-0.645920932293	0	0
16	0.85	-0.645920932293	0	0
17	0.9	-0.645920932293	0	0
18	0.925	-0.645920932293	0	0
19	0.95	-0.645920932293	0	0
20	0.975	-0.645920932293	0	0


# DDA 822 modeling
checkpoint 1: works for negatives
Target: cw_coin_return -inf to -0.4
ID: e4937a58-c61c-4222-a1ac-8d22ce7a4f1b
===================================
Test Samples:             34,109
Val Positive Samples:     456,436 (27.9%)
Number of Features:       64
Features per Window:      16

Classification Metrics:      Val   |  Test
-------------------------------------------
Val ROC AUC:                0.611  |  0.930
Val Accuracy:               0.720  |  0.926
Val Precision:              0.459  |  0.964
Val Recall:                 0.001  |  0.183
Val F1 Score:               0.001  |  0.308


# DDA 818 wallet model across macros
checkpoint 3: roc.956 60d validation checks out?
Model Performance Summary
Target: cw_crypto_net_gain/crypto_inflows/base 0.5 to inf
ID: d32d22e1-9565-4a99-ab59-bb3317287359
===================================
Test Samples:             103,648
Test Positive Samples:    56,130 (54.15%)
Number of Features:       372
Features per Window:      62

Test Set Classification Metrics
True Positives:             5366/103648 (5.2)
-----------------------------------
ROC AUC:                    0.932
Log Loss:                   0.117
Accuracy:                   0.950
Precision:                  0.875
Recall:                     0.043
F1 Score:                   0.082



checkpoint 2: roc.956 15d validation checks out?
Model Performance Summary
Target: cw_crypto_net_gain/crypto_inflows/base 0.4 to inf
ID: 16aab8cf-3f12-4fab-9bd7-42a9453a7956
===================================
Test Samples:             95,074
Test Positive Samples:    141,897 (149.25%)
Number of Features:       496
Features per Window:      62

Test Set Classification Metrics
True Positives:             2706/95074 (2.8)
-----------------------------------
ROC AUC:                    0.956
Log Loss:                   0.063
Accuracy:                   0.974
Precision:                  0.909
Recall:                     0.096
F1 Score:                   0.174


checkpoint 1: roc.920 validation checks out?
Model Performance Summary
Target: cw_crypto_net_gain/crypto_inflows/base 0.4 to inf
ID: 890bb610-0778-4afc-a8dd-498a7b4277c9
===================================
Test Samples:             22,385
Test Positive Samples:    122,465 (547.09%)
Number of Features:       496
Features per Window:      62

Test Set Classification Metrics
True Positives:             1881/22385 (8.4)
-----------------------------------
ROC AUC:                    0.920
Log Loss:                   0.174
Accuracy:                   0.918
Precision:                  0.805
Recall:                     0.035
F1 Score:                   0.067


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
