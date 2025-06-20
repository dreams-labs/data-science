# DDA 827 investing orchestration
checkpoint 1: wt5.711 high performing with transfers features
ID: 7f913fd2-999f-4f4e-b3a7-dc0cb41e5c33
===================================
Test Samples:             94,864
Val Positive Samples:     39,140 (41.3%)
Number of Features:       420
Features per Window:      70

Classification Metrics:      Val   |  Test
-------------------------------------------
Val ROC AUC:                0.824  |  0.919
Val Accuracy:               0.933  |  0.954
Val Precision:              0.780  |  0.922
Val Recall:                 0.001  |  0.013
Val F1 Score:               0.002  |  0.027

Validation Returns    | Cutoff |  Mean   |  W-Mean
--------------------------------------------------
Overall Average       |    n/a |   0.068 |   0.048
Param Threshold       |   0.85 |   0.992 |   0.992
5 Highest Scores      |   0.92 |   0.711 |   0.711
Top 1% Scores         |   0.23 |   0.213 |   0.210
Top 5% Scores         |   0.10 |   0.156 |   0.131
F0.10 Score           |   0.35 |   0.333 |   0.329
F0.25 Score           |   0.29 |   0.279 |   0.277
F0.50 Score           |   0.04 |   0.170 |   0.121
F1 Score              |   0.03 |   0.182 |   0.125
F2 Score              |   0.01 |   0.169 |   0.116


# DDA 794 target var df terminology
checkpoint 1b: w1%.311 rename complete (main)
Model Performance Summary
Target: cw_crypto_net_flows/crypto_inflows/base 0.4 to inf
ID: 080f4a6e-1115-4c67-8a50-94d8ba426046
===================================
Test Samples:             188,145
Number of Features:       372
Features per Window:      62

Classification Metrics
-----------------------------------
ROC AUC:                    0.939
Log Loss:                   0.110
Accuracy:                   0.955
Precision:                  0.871
Recall:                     0.090
F1 Score:                   0.164

Validation Metrics
-----------------------------------
Val ROC AUC:                0.765
Val Accuracy:               0.862
Val Precision:              0.547
Val Recall:                 0.003
Val F1 Score:               0.005

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       2614/3837386 (0.07%)
Positive Mean Outcome:      0.241
Positive W-Mean Outcome:    0.269
Top 1% W-Mean Outcome:      0.311
Top 5% W-Mean Outcome:      0.277
Overall W-Mean Outcome:     0.115


checkpoint 1: w1%.340 rename complete
Model Performance Summary
Target: cw_crypto_net_flows/crypto_inflows/base 0.4 to inf
ID: 6c717c6f-5210-44c0-abfd-a516e1bb0fa2
===================================
Test Samples:             188,145
Number of Features:       372
Features per Window:      62

Classification Metrics
-----------------------------------
ROC AUC:                    0.940
Log Loss:                   0.110
Accuracy:                   0.955
Precision:                  0.872
Recall:                     0.090
F1 Score:                   0.163

Validation Metrics
-----------------------------------
Val ROC AUC:                0.770
Val Accuracy:               0.862
Val Precision:              0.686
Val Recall:                 0.006
Val F1 Score:               0.013

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       5009/3837386 (0.13%)
Positive Mean Outcome:      0.406
Positive W-Mean Outcome:    0.412
Top 1% W-Mean Outcome:      0.340
Top 5% W-Mean Outcome:      0.286
Overall W-Mean Outcome:     0.115



# DDA 785 wallet scores for may
checkpoint 2: flows with training boolean: w1%.305
Model Performance Summary
Target: cw_crypto_net_flows/crypto_inflows/base 0.4 to inf
ID: f94683b8-5143-488d-b169-142e4a8fa621
===================================
Test Samples:             153,572
Number of Features:       372
Features per Window:      62

Classification Metrics
-----------------------------------
ROC AUC:                    0.928
Log Loss:                   0.186
Accuracy:                   0.904
Precision:                  0.801
Recall:                     0.065
F1 Score:                   0.120

Validation Metrics
-----------------------------------
Val ROC AUC:                0.856
Val Accuracy:               0.927
Val Precision:              0.501
Val Recall:                 0.013
Val F1 Score:               0.025

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       1570/838092 (0.19%)
Positive Mean Outcome:      0.356
Positive W-Mean Outcome:    0.364
Top 1% W-Mean Outcome:      0.305
Top 5% W-Mean Outcome:      0.271
Overall W-Mean Outcome:     0.095


checkpoint 1: flows baseline: w1%.322
Model Performance Summary
Target: cw_crypto_net_flows/crypto_inflows/base 0.4 to inf
ID: 4c7b18c3-9232-41c3-949e-890db7e63fda
===================================
Test Samples:             153,572
Number of Features:       372
Features per Window:      62

Classification Metrics
-----------------------------------
ROC AUC:                    0.926
Log Loss:                   0.188
Accuracy:                   0.904
Precision:                  0.789
Recall:                     0.066
F1 Score:                   0.122

Validation Metrics
-----------------------------------
Val ROC AUC:                0.865
Val Accuracy:               0.926
Val Precision:              0.481
Val Recall:                 0.008
Val F1 Score:               0.015

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       966/838092 (0.12%)
Positive Mean Outcome:      0.315
Positive W-Mean Outcome:    0.323
Top 1% W-Mean Outcome:      0.322
Top 5% W-Mean Outcome:      0.285
Overall W-Mean Outcome:     0.095


# DDA 741 investing period performance
checkpoint 5: c1%.40 works in bull market?
Model Performance Summary
Target: coin_return 0.8 to
ID: 81ba23fe-488b-4dfd-a819-4fcec05e7dfd
===================================
Test Samples:             1,632
Number of Features:       30


Classification Metrics
-----------------------------------
ROC AUC:                    0.831
Log Loss:                   0.404
Accuracy:                   0.838
Precision:                  0.251
Recall:                     0.731
F1 Score:                   0.374

Validation Metrics
-----------------------------------
Val ROC AUC:                0.607
Val Accuracy:               0.925
Val Precision:              0.000
Val Recall:                 0.000
Val F1 Score:               0.000

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       0/2558 (0.00%)
Positive Mean Outcome:      nan
Positive W-Mean Outcome:    nan
Top 1% W-Mean Outcome:      0.400
Top 5% W-Mean Outcome:      0.331
Overall W-Mean Outcome:     0.113



checkpoint 4: regression?
Model Performance Summary
Target: coin_return_pctile_full
ID: 754a5d9d-66c9-4f73-9c7c-9272a50e542c
===================================
Test Samples:             1,713
Number of Features:       24


Core Metrics
-----------------------------------
R² Score:                 0.071
RMSE:                     0.279
MAE:                      0.234

Validation Set Metrics
-----------------------------------
R² Score:                 -0.019
RMSE:                     0.306
MAE:                      0.265

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.015
Std of Residuals:         0.278
95% Prediction Interval:  ±0.546


checkpoint 3: c1%.041 back to square 1 I guess
Model Performance Summary
Target: coin_return 0.1 to
ID: 0aa3e5e4-1215-4d87-a575-dcca8a3870ea
===================================
Test Samples:             1,713
Number of Features:       24


Classification Metrics
-----------------------------------
ROC AUC:                    0.833
Log Loss:                   0.385
Accuracy:                   0.849
Precision:                  0.718
Recall:                     0.541
F1 Score:                   0.617

Validation Metrics
-----------------------------------
Val ROC AUC:                0.479
Val Accuracy:               0.761
Val Precision:              0.000
Val Recall:                 0.000
Val F1 Score:               0.000

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.50
Positive Predictions:       0/12087 (0.00%)
Positive Mean Outcome:      nan
Positive W-Mean Outcome:    nan
Top 1% W-Mean Outcome:      0.041
Top 5% W-Mean Outcome:      -0.003
Overall W-Mean Outcome:     -0.048


# DDA 741 investing period performance
checkpoint 2: c1%.804 model rebuild
Model Performance Summary
Target: coin_return 0.5 to
ID: 4972b23c-79de-4ee3-921b-41056608525b
===================================
Test Samples:             4,671
Number of Features:       48


Classification Metrics
-----------------------------------
ROC AUC:                    0.816
Log Loss:                   0.249
Accuracy:                   0.909
Precision:                  0.662
Recall:                     0.100
F1 Score:                   0.174

Validation Metrics
-----------------------------------
Val ROC AUC:                0.685
Val Accuracy:               0.916
Val Precision:              0.727
Val Recall:                 0.048
Val F1 Score:               0.090

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.50
Positive Predictions:       132/23083 (0.57%)
Positive Mean Outcome:      1.106
Positive W-Mean Outcome:    0.912
Top 1% W-Mean Outcome:      0.804
Top 5% W-Mean Outcome:      0.497
Overall W-Mean Outcome:     -0.047


checkpoint 1: c1%.776 model updates
Model Performance Summary
Target: coin_return 0.5 to
ID: f8857fcf-6f44-48ce-89b5-8a60337a5af3
===================================
Test Samples:             6,141
Number of Features:       51


Classification Metrics
-----------------------------------
ROC AUC:                    0.807
Log Loss:                   0.241
Accuracy:                   0.914
Precision:                  0.522
Recall:                     0.045
F1 Score:                   0.083

Validation Metrics
-----------------------------------
Val ROC AUC:                0.644
Val Accuracy:               0.913
Val Precision:              0.812
Val Recall:                 0.021
Val F1 Score:               0.041

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.50
Positive Predictions:       69/30387 (0.23%)
Positive Mean Outcome:      2.162
Positive W-Mean Outcome:    1.072
Top 1% W-Mean Outcome:      0.776
Top 5% W-Mean Outcome:      0.485
Overall W-Mean Outcome:     -0.024


# DDA 778 parquet logic modeling
checkpoint 1: pos1.00
Model Performance Summary
Target: coin_return 0.5 to
ID: 4e19e982-3520-4993-bcbb-9453cbeff8f0
===================================
Test Samples:             4,294
Number of Features:       96


Classification Metrics
-----------------------------------
ROC AUC:                    0.813
Log Loss:                   0.212
Accuracy:                   0.926
Precision:                  0.500
Recall:                     0.060
F1 Score:                   0.107

Validation Metrics
-----------------------------------
Val ROC AUC:                0.767
Val Accuracy:               0.916
Val Precision:              1.000
Val Recall:                 0.002
Val F1 Score:               0.003

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.60
Positive Predictions:       3/21193 (0.01%)
Positive Mean Outcome:      1.006
Positive W-Mean Outcome:    1.006
Top 1% W-Mean Outcome:      0.717
Top 5% W-Mean Outcome:      0.532
Overall W-Mean Outcome:     -0.055




# DDA 777 modeling
checkpoint 1: pos.809 appears to be working but needs audits
Target: coin_return_winsorized 0.3 to
ID: 7fd85eda-2f8b-44f0-bfdb-694dc3aa9e64
===================================
Test Samples:             4,275
Number of Features:       24


Classification Metrics
-----------------------------------
ROC AUC:                    0.745
Log Loss:                   0.410
Accuracy:                   0.824
Precision:                  0.500
Recall:                     0.005
F1 Score:                   0.011

Validation Metrics
-----------------------------------
Val ROC AUC:                0.764
Val Accuracy:               0.823
Val Precision:              0.808
Val Recall:                 0.006
Val F1 Score:               0.011

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.50
Positive Predictions:       26/21238 (0.12%)
Positive Mean Outcome:      0.809
Positive W-Mean Outcome:    0.809
Top 1% W-Mean Outcome:      0.611
Top 5% W-Mean Outcome:      0.465
Overall W-Mean Outcome:     0.027


# DDA 625 coin epochs v1
checkpoint 1: vauc.611 coin epochs working
Model Performance Summary
Target: coin_return_winsorized 0.3 to
ID: 0bf585c2-b1e0-4f8e-9b25-546724704a08
===================================
Test Samples:             2,554
Number of Features:       72


Classification Metrics
-----------------------------------
ROC AUC:                    0.671
Log Loss:                   0.478
Accuracy:                   0.795
Precision:                  0.667
Recall:                     0.015
F1 Score:                   0.030

Validation Metrics
-----------------------------------
Val ROC AUC:                0.611
Val Accuracy:               0.927
Val Precision:              0.083
Val Recall:                 0.003
Val F1 Score:               0.006

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.50
Positive Predictions:       36/12700 (0.28%)
Positive Mean Outcome:      -0.071
Positive W-Mean Outcome:    -0.071
Top 1% W-Mean Outcome:      0.018
Top 5% W-Mean Outcome:      -0.061
Overall W-Mean Outcome:     -0.151


# DDA 769 coin model score dist toggle
checkpoint 1: w1%_.211 rolled back some drop patterns
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/base 0.4 to inf
ID: b13e0dc1-3e58-4ccd-b43b-23d88c5ded89
===================================
Test Samples:             178,992
Number of Features:       372
Features per Window:      62

Classification Metrics
-----------------------------------
ROC AUC:                    0.935
Log Loss:                   0.110
Accuracy:                   0.954
Precision:                  0.868
Recall:                     0.093
F1 Score:                   0.167

Validation Metrics
-----------------------------------
Val ROC AUC:                0.845
Val Accuracy:               0.948
Val Precision:              0.273
Val Recall:                 0.007
Val F1 Score:               0.014

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       4395/3305076 (0.13%)
Positive Mean Outcome:      0.620
Positive W-Mean Outcome:    0.278
Top 1% W-Mean Outcome:      0.211
Top 5% W-Mean Outcome:      0.131
Overall W-Mean Outcome:     -0.006


# DDA 760 ablation
checkpoint w_.6_1.934 ablation analysis complete
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/base 0.6 to inf
ID: 47290d65-c40e-4d02-b89c-3d549c8ad20f
===================================
Test Samples:             191,189
Number of Features:       285
Features per Window:      56

Classification Metrics
-----------------------------------
ROC AUC:                    0.924
Log Loss:                   0.101
Accuracy:                   0.962
Precision:                  0.859
Recall:                     0.065
F1 Score:                   0.121

Validation Metrics
-----------------------------------
Val ROC AUC:                0.952
Val Accuracy:               0.942
Val Precision:              0.909
Val Recall:                 0.121
Val F1 Score:               0.214

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.80
Positive Predictions:       20883/2385725 (0.88%)
Positive Mean Outcome:      1.934
Positive W-Mean Outcome:    1.357
Top 1% W-Mean Outcome:      1.337
Top 5% W-Mean Outcome:      0.946
Overall W-Mean Outcome:     0.069



# DDA 718 coin model grid search
checkpoint 1: seems alive
Model Performance Summary
Target: coin_return_winsorized 0.3
ID: 93eac06a-bf0f-409b-8853-6523b66342da
===================================
Test Samples:             1,116
Number of Features:       18
Features per Window:      0

Classification Metrics
-----------------------------------
ROC AUC:                    0.574
Log Loss:                   0.346
Accuracy:                   0.890
Precision:                  0.000
Recall:                     0.000
F1 Score:                   0.000

Validation Metrics
-----------------------------------
Val ROC AUC:                0.523
Val Accuracy:               0.689
Val Precision:              0.652
Val Recall:                 0.009
Val F1 Score:               0.017

Validation Return Metrics
-----------------------------------
Positive Threshold:         0.20
Positive Predictions:       23/5577 (0.41%)
Positive Mean Outcome:      0.788
Positive W-Mean Outcome:    0.788
Top 1% W-Mean Outcome:      0.551
Top 5% W-Mean Outcome:      0.410
Overall W-Mean Outcome:     0.260


# DDA 528 investing period validation
checkpoint 1: is that life?
Model Performance Summary
Target: coin_return
ID: 421ab86f-2ad6-4631-a69c-e1104b9eb82e
===================================
Test Samples:             1,262
Number of Features:       180
Features per Window:      0

Classification Metrics
-----------------------------------
ROC AUC:                  0.594
Log Loss:                 0.218
Accuracy:                 0.942
Precision:                0.000
Recall:                   0.000
F1 Score:                 0.000

Validation Return Metrics
----------------------------------
Val ROC AUC:              0.588
Top 1% Mean Return:       0.173
Top 5% Mean Return:       -0.115
Overall Mean Return:      106691.453


# dda 706 refactor coin model
checkpoint 1: cauc.597 features refactored
Model Performance Summary
Target: coin_return
ID: b81a7dc3-faf5-428d-936a-fd22e54b7878
===================================
Test Samples:             423
Number of Features:       104
Features per Window:      0

Classification Metrics
-----------------------------------
ROC AUC:                  0.671
Log Loss:                 0.445
Accuracy:                 0.823
Precision:                0.571
Recall:                   0.104
F1 Score:                 0.176

Validation Return Metrics
-----------------------------------
Val ROC AUC:              0.597
Top 1% Mean Return:       0.053
Top 5% Mean Return:       0.068
Overall Mean Return:      0.023


# dda 721 predict outside of db
checkpoint 1: w.978 predict with db up to 2/26/25
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: bfa55a33-712e-4d82-bb5c-11fc942bcb62
===================================
Test Samples:             231,803
Number of Features:       335
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.978
Log Loss:                 0.060
Accuracy:                 0.977
Precision:                0.811
Recall:                   0.402
F1 Score:                 0.538



# dda 719 reversed AUC investigation
checkpoint 2: wval.870 dda719 data dda699 configs
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: 3908d7c0-b87a-497f-a71d-8b4ed3c52c41
===================================
Test Samples:             181,792
Number of Features:       335
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.985
Log Loss:                 0.028
Accuracy:                 0.990
Precision:                0.815
Recall:                   0.403
F1 Score:                 0.540

Validation Return Metrics
-----------------------------------
Val ROC AUC:              0.870
Top 1% Avg Return:        0.382
Top 5% Avg Return:        0.390
Overall Avg Return:       -0.024


checkpoint 1: wval.871 dda699 data works fine
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: cac7d580-1480-48fc-9b5d-c2eb9d2d0907
===================================
Test Samples:             181,792
Number of Features:       335
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.984
Log Loss:                 0.028
Accuracy:                 0.991
Precision:                0.815
Recall:                   0.410
F1 Score:                 0.546

Validation Return Metrics
-----------------------------------
Val ROC AUC:              0.871
Top 1% Avg Return:        0.417
Top 5% Avg Return:        0.387
Overall Avg Return:       -0.024


# dda 703 financial metrics in summary
checkpoint 2: auc.867 financial metrics working
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: 7d356e75-c371-40eb-80ee-344dcf730cd9
===================================
Test Samples:             181,792
Number of Features:       335
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.984
Log Loss:                 0.028
Accuracy:                 0.990
Precision:                0.806
Recall:                   0.407
F1 Score:                 0.541

Validation Return Metrics
-----------------------------------
Val ROC AUC:              0.867
Top 1% Avg Return:        0.409
Top 5% Avg Return:        0.378
Overall Avg Return:       -0.024


checkpoint 1: aucprod.868 concurrent
Classification Metrics
-----------------------------------
ROC AUC:                  0.985
Log Loss:                 0.027
Accuracy:                 0.991
Precision:                0.822
Recall:                   0.415
F1 Score:                 0.552

Validation Return Metrics
-----------------------------------
Val ROC AUC:              0.868
Top 1% Avg Return:        0.411
Top 5% Avg Return:        0.381
Overall Avg Return:       -0.024


# dda 685 training and val set concurrently
checkpoint 2: aucprod.640 concurrent
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: 6e8c9d89-4920-41df-8392-e71916dbe582
===================================
Test Samples:             6,763
Number of Features:       140
Features per Window:      35

Classification Metrics
-----------------------------------
ROC AUC:                  0.802
Log Loss:                 0.239
Accuracy:                 0.915
Precision:                0.463
Recall:                   0.065
F1 Score:                 0.114

Validation Set Metrics
-----------------------------------
Val ROC AUC:              0.640
Val Accuracy:             0.916
Val Precision:            0.000
Val Recall:               0.000
Val F1 Score:             0.000


checkpoint 2: aucprod.666 baseline
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: 868ac354-d820-4eaf-8fec-7da2106e5b75
===================================
Test Samples:             6,763
Number of Features:       140
Features per Window:      35

Classification Metrics
-----------------------------------
ROC AUC:                  0.799
Log Loss:                 0.241
Accuracy:                 0.915
Precision:                0.457
Recall:                   0.056
F1 Score:                 0.100

Validation Set Metrics
-----------------------------------
Val ROC AUC:              0.666
Val Accuracy:             0.916
Val Precision:            0.000
Val Recall:               0.000
Val F1 Score:             0.000


checkpoint 1: aucdev.590 baseline
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: 529710d5-aa7f-44d6-b503-b1c674b3ac00
===================================
Test Samples:             1,368
Number of Features:       268
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.965
Log Loss:                 0.153
Accuracy:                 0.941
Precision:                0.788
Recall:                   0.708
F1 Score:                 0.746

Validation Set Metrics
-----------------------------------
Val ROC AUC:              0.590
Val Accuracy:             0.961
Val Precision:            0.000
Val Recall:               0.000
Val F1 Score:             0.000


checkpoint 1: auc.561 concurrent is working
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: afc9dfbc-bd19-42a6-8b4d-601f096dcd50
===================================
Test Samples:             1,368
Number of Features:       268
Features per Window:      67

Classification Metrics
-----------------------------------
ROC AUC:                  0.964
Log Loss:                 0.159
Accuracy:                 0.935
Precision:                0.769
Recall:                   0.673
F1 Score:                 0.717

Validation Set Metrics
-----------------------------------
Val ROC AUC:              0.561
Val Accuracy:             0.938
Val Precision:            0.000
Val Recall:               0.000
Val F1 Score:             0.000


# dda 696 classification evaluator
checkpoint 1: auc.70 summary report working
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: 857fd0c4-2bfd-4817-a1f1-543757ec693f
===================================
Test Samples:             181,792
Number of Features:       280
Features per Window:      70

Classification Metrics
-----------------------------------
Accuracy:                 0.986
Precision:                0.845
Recall:                   0.432
F1 Score:                 0.572
ROC AUC:                  0.979
Log Loss:                 0.041

Validation Set Metrics
-----------------------------------
Val Accuracy:             0.934
Val Precision:            0.000
Val Recall:               0.000
Val F1 Score:             0.000
Val ROC AUC:              0.698



# dda 472 classification model
checkpoint 2: auc.69 viable predictions?
Model Performance Summary
Target: cw_crypto_net_gain/max_investment/winsorized
ID: ed358ac1-78e2-4dfd-a338-132f844623aa
===================================
Test Samples:             181,792
Number of Features:       280
Features per Window:      70

Core Metrics
-----------------------------------
R² Score:                 0.334
RMSE:                     0.118
MAE:                      0.014

Validation Set Metrics
-----------------------------------
R² Score:                 -0.070
RMSE:                     0.256
MAE:                      0.066



checkpoint 1: w.653 basic classification model working
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: 787ab9b0-7609-458e-b817-2ed09a288d78
===================================
Test Samples:             57,544
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.653
RMSE:                     0.167
MAE:                      0.068

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.292
RMSE:                     0.227
MAE:                      0.155



# dda 694 epoch assignments
checkpoint 1: w.659 assignment logic completed
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: dd6e0209-1f31-41d2-9f0a-b3547e190784
===================================
Test Samples:             57,544
Number of Features:       84
Features per Window:      21

Core Metrics
-----------------------------------
R² Score:                 0.659
RMSE:                     0.226
MAE:                      0.119

Validation Set Metrics
-----------------------------------
R² Score:                 -0.171
RMSE:                     0.451
MAE:                      0.230

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.759
RMSE:                     0.132
MAE:                      0.064


# dda 692 parallelizing
checkpoint 1: w.676 modeling cohort logic into model
Model Performance Summary
Target: crypto_net_gain/max_investment/winsorized
ID: c3c4b854-7a8b-49bd-8f95-d7c195c8a33d
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.676
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.125
MAE:                      0.066


# dda 683 modeling
checkpoint 1: whv.327
Model Performance Summary
ID: c82f2ea4-fed2-481a-a509-75a42392fee5
===================================
Test Samples:             182,779
Number of Features:       275
Features per Window:      55

Core Metrics
-----------------------------------
R² Score:                 0.363
RMSE:                     180376.594
MAE:                      5260.404

Validation Set Metrics
-----------------------------------
R² Score:                 0.327
RMSE:                     74372.711
MAE:                      3159.549

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.491
RMSE:                     83309.445
MAE:                      2752.584


# dda 675 parallelizing
checkpoint 2: w.675 parallelize modeling and training features done
Model Performance Summary
ID: 18ebc4da-302a-463f-9ea6-f2afe397adfa
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.675
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.125
MAE:                      0.066



checkpoint 1: w.674 reorder epoch generator (2m 12s)
Model Performance Summary
ID: 53b58dc7-7620-4bbf-bd8a-75ebf7e40afe
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.674
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.125
MAE:                      0.066


# dda 683 hybrid iterations
checkpoint 1: w.675 small fixes
Model Performance Summary
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.675
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.782
RMSE:                     0.126
MAE:                      0.066


checkpoint 1: wh.899 small fixes
Model Performance Summary
===================================
Test Samples:             117,117
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.899
RMSE:                     0.131
MAE:                      0.062

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.861
RMSE:                     0.116
MAE:                      0.061

checkpoint 1: wh.899 df logic implemented


# dda 681 get hybrid ids via dfs
checkpoint 1: w.676 df logic implemented
Model Performance Summary
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.676
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.125
MAE:                      0.066


checkpoint 1: HyDev.807 df logic implemented
Model Performance Summary
===================================
Test Samples:             2,324
Number of Features:       600
Features per Window:      150

Core Metrics
-----------------------------------
R² Score:                 0.807
RMSE:                     0.227
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.845
RMSE:                     0.135
MAE:                      0.072


# dda 673 get hybrid id mappings from bigquery
checkpoint 1: w.676 df retrieved with other datasets
Model Performance Summary
===================================
Test Samples:             57,544
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.676
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.125
MAE:                      0.066


# dda 679 toggle transfers features
checkpoint 3: w.677 toggle cohort upload to bigquery (2m 13s)
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.677
RMSE:                     0.221
MAE:                      0.119

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.779
RMSE:                     0.128
MAE:                      0.068


checkpoint 2: w.675 var name for toggle_scenario_features
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.675
RMSE:                     0.221
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.778
RMSE:                     0.128
MAE:                      0.069


checkpoint 1: w.677 toggle set to off (2m24s)
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       136
Features per Window:      34

Core Metrics
-----------------------------------
R² Score:                 0.677
RMSE:                     0.221
MAE:                      0.119

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.779
RMSE:                     0.128
MAE:                      0.069


checkpoint 1: w.683 toggle set to on (2m55s)
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068


# dda 677 hybrid and base features
checkpoint 1: w.683 features integrated
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068



# dda 676 fix verbosity
checkpoint 1: HyDev.819 split out _process_single_epoch()
Model Performance Summary
===================================
Test Samples:             2,111
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.231
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.822
RMSE:                     0.140
MAE:                      0.077


# dda 676 fix verbosity
checkpoint 1: w.683 fixed verbosity
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068

checkpoint 1: HyDev.819 fixed verbosity
Model Performance Summary
===================================
Test Samples:             2,111
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.819
RMSE:                     0.217
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.834
RMSE:                     0.135
MAE:                      0.079


# dda 583 retry hybridization
checkpoint 4: w.682 basic hybridization working
Model Performance Summary
===================================
Test Samples:             114,621
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.193
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.770
RMSE:                     0.119
MAE:                      0.070


checkpoint 4: HDEV.820 move transfers mapping out of bigquery
Model Performance Summary
===================================
Test Samples:             3,972
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.820
RMSE:                     0.179
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.803
RMSE:                     0.119
MAE:                      0.066


checkpoint 3: HDEV.821 remove pickling logic from functions
Model Performance Summary
===================================
Test Samples:             3,972
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.821
RMSE:                     0.185
MAE:                      0.097

Validation Set Metrics
-----------------------------------
R² Score:                 -0.587
RMSE:                     0.404
MAE:                      0.227

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.803
RMSE:                     0.119
MAE:                      0.067


checkpoint 2: HDEV.819  hybridize_wallet_address() working in multiwindow generator
Model Performance Summary
===================================
Test Samples:             70,629
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.819
RMSE:                     0.148
MAE:                      0.076

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.804
RMSE:                     0.096
MAE:                      0.057



checkpoint 1: HDEV.839 update hybridize_wallet_address()
Model Performance Summary
===================================
Test Samples:             2,111
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.839
RMSE:                     0.224
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.831
RMSE:                     0.136
MAE:                      0.081

checkpoint 1: w.682 update hybridize_wallet_address()
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068


# dda 669.2 predictive segments
checkpoint 1: w.682 tweaked analysis
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


# dda 669 predictive segments
checkpoint 1: w.681 analysis working
Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.219
MAE:                      0.118

Validation Set Metrics
-----------------------------------
R² Score:                 -0.134
RMSE:                     0.405
MAE:                      0.220

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.781
RMSE:                     0.127
MAE:                      0.069


# dda 665 validation scoring
checkpoint 3: w.682 validation reporting working
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Validation Set Metrics
-----------------------------------
R² Score:                 -0.148
RMSE:                     0.408
MAE:                      0.222


checkpoint 2: w.682 validation grid search working
Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


checkpoint 1: w.682 update macro drop features
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       152
Features per Window:      38

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


# dda 668 short windows
checkpoint 2: w.680 remove time boundaries on market indicators
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.680
RMSE:                     0.220
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.781
RMSE:                     0.127
MAE:                      0.069



checkpoint 1: w.682 confirming baseline
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.069


# dda 667 feature importance reporting
checkpoint 1: w.682 graph updated
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.069


# dda 664 grid search functionality expansion
checkpoint 13: w.683 r2 test working for target vars
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 12: w.683 metapipeline integrated
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 11: w.683 modifying wallet_model variables
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 10: w.683 add raise to grid search
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 9: w.683 extract grid search preparation from function in base model
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 8: w.683 grid search working
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 7: w.683 metapipeline working
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 6: w.683 target var to y pipeline
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 5: w.683 integrate y_pipeline
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 4: w.683 integrate _get_wallet_pipeline()
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067



checkpoint 3: w.683 simplify construct_wallet_model()
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 2: w.683 change base model pipeline function
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


checkpoint 1: w.683 refactor wm._prepare_data
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


# dda 663 finding predictiveness
checkpoint 1: w.683 can predict # of coins traded
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.067


# dda 660 multithread epochs
checkpoint 3: w.681 ready to merge
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


checkpoint 2: w.683 multithreading implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068


checkpoint 1: w.682 split out single epoch function
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.069


# dda 661 fix epoch terminology
checkpoint 1: w.683 epoch terms implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068


# dda 649 n_threads consolodation
checkpoint 1: w.681 n_threads consolidated
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.782
RMSE:                     0.127
MAE:                      0.069


# dda 661 macro features
checkpoint 1: w.683 macro features added
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       184
Features per Window:      46

Core Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.126
MAE:                      0.068


# dda 659 validation date range audit
checkpoint 1: w.681 validation range implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.219
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


# dda 535 logger MILESTONEs
checkpoint 1: CHECKPOINTERw.680 milestone implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.680
RMSE:                     0.220
MAE:                      0.119

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


checkpoint 1: PRODUCTIONw.690 milestone implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.690
RMSE:                     0.216
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.793
RMSE:                     0.124
MAE:                      0.065



# dda 655 complete profits_df bug
checkpoint 1: w.694 macro trends df fixed
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.215
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.798
RMSE:                     0.122
MAE:                      0.066



# dda 656 more macroeconomic features
checkpoint 2: w.694 ready to merge
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.215
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.795
RMSE:                     0.123
MAE:                      0.067


checkpoint 1: w.686 more timing features
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       140
Features per Window:      35

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.218
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.790
RMSE:                     0.125
MAE:                      0.068


# dda 654 macroeconomic features impact
checkpoint 1: w.688 no impact
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.217
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.789
RMSE:                     0.125
MAE:                      0.065



# dda 635 macroeconomic features
checkpoint 9: w.685 features now in model
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       148
Features per Window:      37

Core Metrics
-----------------------------------
R² Score:                 0.685
RMSE:                     0.218
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.792
RMSE:                     0.124
MAE:                      0.066



checkpoint 9: DEVw.817 features now in model
Model Performance Summary
===================================
Test Samples:             2,066
Number of Features:       148
Features per Window:      37

Core Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.233
MAE:                      0.124

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.839
RMSE:                     0.138
MAE:                      0.076


checkpoint 8: w. full features in parallel with windows
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.217
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.786
RMSE:                     0.126
MAE:                      0.066


checkpoint 8: DEVw.809 full features in parallel with windows
Model Performance Summary
===================================
Test Samples:             2,066
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.238
MAE:                      0.126

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.830
RMSE:                     0.142
MAE:                      0.078


checkpoint 7: DEVw.815 update prepare_training_data
Model Performance Summary
===================================
Test Samples:             2,066
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.815
RMSE:                     0.234
MAE:                      0.124

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.837
RMSE:                     0.139
MAE:                      0.076


checkpoint 6: w.689 multiwindow working
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.217
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.787
RMSE:                     0.125
MAE:                      0.066


checkpoint 6: DEVw.815 multiwindow working
Model Performance Summary
===================================
Test Samples:             2,066
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.815
RMSE:                     0.234
MAE:                      0.124

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.837
RMSE:                     0.139
MAE:                      0.076


checkpoint 5: DEVw.839 log dev score
===================================
Test Samples:             2,066
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.839
RMSE:                     0.222
MAE:                      0.120

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.147
MAE:                      0.085

checkpoint 5: w.839 log dev score
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.787
RMSE:                     0.125
MAE:                      0.067



checkpoint 4: w.686 generalizing indicators function
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.787
RMSE:                     0.125
MAE:                      0.067


checkpoint 3: w.686 macro_trends_df online
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.786
RMSE:                     0.126
MAE:                      0.067


checkpoint 2: w.690 add macro df to orchestrators
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.690
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.126
MAE:                      0.068


checkpoint 1: w.688 multiwindow orchestrator fix
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.217
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.787
RMSE:                     0.125
MAE:                      0.066


# dda 650 reconcile windows vs single
checkpoint 3: mw.688 they're matching??
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.217
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.786
RMSE:                     0.126
MAE:                      0.067

checkpoint 3: sw.686 they're matching??
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


checkpoint 2: w.686 coin cohorts both match 4938
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


checkpoint 1: w.686 baseline single
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


# dda 573 pull from complete dfs
checkpoint 4: w.686 closing out ticket to move to new branch
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066



checkpoint 3: w.686 single run; trying to reconcile windows vs single
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


checkpoint 2: w.686 functions work but index trouble
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


checkpoint 1: w.686 df params for retrieve_cleaned_period_datasets()
faster params
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.216
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.788
RMSE:                     0.125
MAE:                      0.066


base params
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.693
RMSE:                     0.214
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.798
RMSE:                     0.122
MAE:                      0.065


# dda 633 predicting val period
checkpoint 8: w.689 faster model params
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.215
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.782
RMSE:                     0.127
MAE:                      0.069


checkpoint 7: w.696 testing validation
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.802
RMSE:                     0.121
MAE:                      0.064


checkpoint 6: w.696 add market timing data availability and val period predictions functional
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.802
RMSE:                     0.121
MAE:                      0.064



checkpoint 5: w.698 add score saving bool
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.698
RMSE:                     0.212
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.803
RMSE:                     0.121
MAE:                      0.064


checkpoint 4: w.698 updated features logic for validation functionality
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.698
RMSE:                     0.212
MAE:                      0.108


checkpoint 3: w.698 back to 2023 laptop
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.698
RMSE:                     0.212
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.803
RMSE:                     0.121
MAE:                      0.064


checkpoint 2: w.698 adding validation period logic to training data orchestrator
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.698
RMSE:                     0.212
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.803
RMSE:                     0.121
MAE:                      0.064


checkpoint 1: xw.721 multi window model working
Model Performance Summary
===================================
Test Samples:             238,722
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.721
RMSE:                     0.207
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.761
RMSE:                     0.135
MAE:                      0.083


# dda 642 multi window model
checkpoint 10: w.696 it works?
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 9: w.696 _init_
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 8: w.696 prepare_training_data()
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 7: w.696 _define_training_wallet_cohort()
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 6: w.696 working on orchestrator
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 5: w.696 updated model reporting
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 4: w.696 wallet_model index match function
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065



checkpoint 3: w.696 wallet_model updated
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 2: w.696 slowly reintegrating
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 1: w.696 multiwindow model works
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 -0.839
RMSE:                     0.368
MAE:                      0.242


checkpoint 1: xw.685 multiwindow model works
Model Performance Summary
===================================
Test Samples:             113,598
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.685
RMSE:                     0.197
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.785
RMSE:                     0.122
MAE:                      0.072


# dda 645 parquet file handling
checkpoint 1: w.696 implemented file handling
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


# dda 636 time windows configs
checkpoint 1: w.696 config and data generation working
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065



# dda 640 class for wallet_training_data
checkpoint 1: w.696 classification complete
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


# dda 639 wallets config allow external date additions
checkpoint 1: w.696 changes implemented
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065

# dda 634 remove module level configs
checkpoint 3: w.696 fully methodize and update notebook
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 2: w.696 class methods are working but why is the score higher
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.696
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.800
RMSE:                     0.121
MAE:                      0.065


checkpoint 1: w.689 baseline
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.216
MAE:                      0.110

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.797
RMSE:                     0.122
MAE:                      0.065



# dda 629 predictiveness of wallet model
checkpoint 2: w.689 aligns with checkpoint 1 rerun
Model Performance Summary
===================================
Test Samples:             57,906
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.216
MAE:                      0.110

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.797
RMSE:                     0.122
MAE:                      0.065


checkpoint 1: w.691 base case
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068


# dda 615 index passthrough
checkpoint 7: w.691 indexify pri
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068


checkpoint 6: w.691 speed up validate_inputs
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068



checkpoint 5: w.691 ensure_index in prepare_dataframes()
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068



checkpoint 4: w.691 indexify split_training_window_profits_dfs
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068


checkpoint 3: w.691 restyle
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.783
RMSE:                     0.127
MAE:                      0.068


checkpoint 2: w.691 add sort
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.691
RMSE:                     0.213
MAE:                      0.109



checkpoint 1: w.693 base case
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.693
RMSE:                     0.212
MAE:                      0.109


# dda 607 coin model features
checkpoint 1: w.693 back to base case
Model Performance Summary
===================================
Test Samples:             57,955
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.693
RMSE:                     0.212
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.776
RMSE:                     0.129
MAE:                      0.069


# dda 627
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.712
RMSE:                     0.198
MAE:                      0.097

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.760
RMSE:                     0.122
MAE:                      0.066



# min threshold $1k min child wght 30 subsample .9 lr .07 gamma 0.01
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.197
MAE:                      0.096


# min threshold $1k min child wght 30 subsample .9
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.197
MAE:                      0.097

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.768
RMSE:                     0.120
MAE:                      0.063


# min threshold $1k min child wght 30 depth 18
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.198
MAE:                      0.098

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.765
RMSE:                     0.120
MAE:                      0.061


# min threshold $1k min child wght 30
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.197
MAE:                      0.096


# min threshold $1k min child wght 20
Model Performance Summary
===================================
Test Samples:             276,949
Number of Features:       248
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.197
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.771
RMSE:                     0.119
MAE:                      0.063


#
add         '2024-03-01',
dur: 3m30s
Model Performance Summary
===================================
Test Samples:             69,462
Number of Features:       217
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.719
RMSE:                     0.187
MAE:                      0.092

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.777
RMSE:                     0.110
MAE:                      0.059


add         '2024-07-01',
new features dur: 3m17s
base features dur: 2m31s
Model Performance Summary
===================================
Test Samples:             69,462
Number of Features:       186
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.717
RMSE:                     0.188
MAE:                      0.092

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.772
RMSE:                     0.111
MAE:                      0.060



# dda 627 softer coin filters
max_gap_days: 20
Removed 666 coins (563 for gaps, 175 for volume) and 536951 total records.
Model Performance Summary
===================================
Test Samples:             69,462
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.708
RMSE:                     0.191
MAE:                      0.096

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.778
RMSE:                     0.109
MAE:                      0.060



max_gap_days: 60
Removed 406 coins (269 for gaps, 175 for volume) and 338000 total records.
Model Performance Summary
===================================
Test Samples:             70,193
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.708
RMSE:                     0.191
MAE:                      0.096

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.762
RMSE:                     0.113
MAE:                      0.063


checkpoint 2: w.713 more parallelization
Model Performance Summary
===================================
Test Samples:             69,961
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.189
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.760
RMSE:                     0.114
MAE:                      0.063



min_daily_volume: 500
Model Performance Summary
===================================
Test Samples:             69,961
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.189
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.760
RMSE:                     0.114
MAE:                      0.063


min_daily_volume: 1000
Removed 551 coins (350 for gaps, 273 for volume) and 438953 total records.
Model Performance Summary
===================================
Test Samples:             69,839
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.704
RMSE:                     0.193
MAE:                      0.096

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.784
RMSE:                     0.108
MAE:                      0.059


min_daily_volume: 100
Removed 393 coins (350 for gaps, 62 for volume) and 326257 total records.
Model Performance Summary
===================================
Test Samples:             70,022
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.705
RMSE:                     0.192
MAE:                      0.096

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.766
RMSE:                     0.112
MAE:                      0.062


base
Removed 473 coins (350 for gaps, 175 for volume) and 387628 total records.



# dda 624 multithread wallet windows
checkpoint 1: w.713 working multithreading
Model Performance Summary
===================================
Test Samples:             69,961
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.713
RMSE:                     0.189
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.760
RMSE:                     0.114
MAE:                      0.063



# dda 626 new coins
checkpoint 2: w.712 add 2022-01-01 window w new coins
Model Performance Summary
===================================
Test Samples:             69,961
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.712
RMSE:                     0.189
MAE:                      0.095

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.762
RMSE:                     0.113
MAE:                      0.063


add 2022-01-01 window
Model Performance Summary
===================================
Test Samples:             68,198
Number of Features:       155
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.716
RMSE:                     0.184
MAE:                      0.093

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.778
RMSE:                     0.107
MAE:                      0.061


checkpoint 1: w.692 base with eval+test set split
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.692
RMSE:                     0.209
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.806
RMSE:                     0.117
MAE:                      0.063


# dda 621 phase training
base high performance
12d/30w/5000n/.02lr/.005g
target_variable: crypto_net_gain
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.793
RMSE:                     6046.907
MAE:                      2343.128

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.888
RMSE:                     2420.432
MAE:                      1053.557


target_variable: crypto_net_flows/max_investment/winsorized
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.764
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.833
RMSE:                     0.091
MAE:                      0.053


target_variable: crypto_net_gain/max_investment/winsorized
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.764
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.833
RMSE:                     0.091
MAE:                      0.053


phases:
    - params:
        max_depth: 9
        min_child_weight: 100
        gamma: 0.02
    - params:
        max_depth: 12
        min_child_weight: 30
        gamma: 0.005
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.763
RMSE:                     0.159
MAE:                      0.080

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.825
RMSE:                     0.093
MAE:                      0.056


12d/30w/5000n/.02lr/.005g alpha 1.0
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.763
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.837
RMSE:                     0.089
MAE:                      0.053



12d/30w/5000n/.02lr/.005g
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.764
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.833
RMSE:                     0.091
MAE:                      0.053



5000/0.05 reg_alpha 20/5/0.1/0

Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.760
RMSE:                     0.160
MAE:                      0.080

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.823
RMSE:                     0.093
MAE:                      0.057


5000/0.05 reg_alpha 20/5/0
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.758
RMSE:                     0.161
MAE:                      0.080

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.819
RMSE:                     0.094
MAE:                      0.058

# dda 619 grid searchin
12d/30w/500n/.2lr lambda 1 alpha 5 subsample 0.8
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.165
MAE:                      0.084

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.780
RMSE:                     0.104
MAE:                      0.069


12d/30w/500n/.2lr lambda .5 alpha 5
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.748
RMSE:                     0.164
MAE:                      0.083

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.807
RMSE:                     0.098
MAE:                      0.061

12d/30w/500n/.2lr lambda 1 alpha 5
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.749
RMSE:                     0.164
MAE:                      0.083

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.810
RMSE:                     0.097
MAE:                      0.060


12d/20w/500n/.2lr lambda 1 alpha 5
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.746
RMSE:                     0.165
MAE:                      0.084

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.097
MAE:                      0.060


12d/30w/500n/.2lr lambda 1 alpha 10
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.165
MAE:                      0.085

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.806
RMSE:                     0.098
MAE:                      0.059


12d/30w/500n/.2lr lambda 1 alpha 5
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.749
RMSE:                     0.164
MAE:                      0.083

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.810
RMSE:                     0.097
MAE:                      0.060


12d/30w/500n/.2lr lambda 1 alpha 2
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.746
RMSE:                     0.165
MAE:                      0.083

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.096
MAE:                      0.060


12d/30w/500n/.2lr lambda 1 alpha 0.5
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.165
MAE:                      0.083

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.805
RMSE:                     0.098
MAE:                      0.060


12d/30w/500n/.2lr lambda 1 alpha 0
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.165
MAE:                      0.084

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.796
RMSE:                     0.100
MAE:                      0.063


12d/30w/500n/.2lr lambda 3 alpha 0
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.744
RMSE:                     0.165
MAE:                      0.084

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.795
RMSE:                     0.100
MAE:                      0.062



12d/20w/5000n/.02lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.764
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.838
RMSE:                     0.089
MAE:                      0.052


12d/30w/5000n/.02lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.764
RMSE:                     0.159
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.833
RMSE:                     0.091
MAE:                      0.053


12d/30w/5000n/.015lr/.005g eval size .05
rmse at 200
[200]	validation_0-rmse:0.18599	validation_1-rmse:0.20190
15d/20w/5000n/.015lr/.005g eval size .05
rmse at .212 val1 rmse
[126]	validation_0-rmse:0.18095	validation_1-rmse:0.20174

rmse at 200
[200]	validation_0-rmse:0.15262	validation_1-rmse:0.18401


15d/20w/5000n/.015lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.762
RMSE:                     0.159
MAE:                      0.077

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.834
RMSE:                     0.091
MAE:                      0.053



15d/20w/2500n/.015lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.762
RMSE:                     0.160
MAE:                      0.077

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.836
RMSE:                     0.090
MAE:                      0.052



15d/20w/1500n/.02lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.759
RMSE:                     0.161
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.837
RMSE:                     0.089
MAE:                      0.052


17d/20w/1000n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.757
RMSE:                     0.161
MAE:                      0.077

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.825
RMSE:                     0.093
MAE:                      0.056


17d/10w/600n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.753
RMSE:                     0.162
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.832
RMSE:                     0.091
MAE:                      0.053



17d/15w/600n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.753
RMSE:                     0.162
MAE:                      0.079

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.834
RMSE:                     0.090
MAE:                      0.052



17d/15w/400n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.748
RMSE:                     0.164
MAE:                      0.081

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.839
RMSE:                     0.089
MAE:                      0.049

19d/15w/400n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.749
RMSE:                     0.164
MAE:                      0.079

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.833
RMSE:                     0.091
MAE:                      0.053


21d/15w/400n/.03lr/.005g eval size .05
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.748
RMSE:                     0.164
MAE:                      0.078

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.830
RMSE:                     0.091
MAE:                      0.053


21d/15w/400n/.03lr/.005g eval size .1
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.165
MAE:                      0.079

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.819
RMSE:                     0.094
MAE:                      0.055


23d/15w/400n/.03lr/.005g
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.739
RMSE:                     0.167
MAE:                      0.080

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.793
RMSE:                     0.101
MAE:                      0.060

21d/15w/400n/.03lr/.01g
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.739
RMSE:                     0.167
MAE:                      0.081

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.804
RMSE:                     0.098
MAE:                      0.057


21d/15w/400n/.03lr/.005g
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       429
Features per Window:      39

Core Metrics
-----------------------------------
R² Score:                 0.740
RMSE:                     0.167
MAE:                      0.080

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.805
RMSE:                     0.098
MAE:                      0.057


# dda 597 eval set logic
checkpoint 2: w.682 base params test performance 60/20/20
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.682
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.762
RMSE:                     0.130
MAE:                      0.070


checkpoint 1: rsme.20830 assess performance vs 20% eval set
end validation outcome
[0]	validation_0-rmse:0.37012	validation_1-rmse:0.37039
[100]	validation_0-rmse:0.11748	validation_1-rmse:0.21537
[200]	validation_0-rmse:0.08937	validation_1-rmse:0.21043
[300]	validation_0-rmse:0.07262	validation_1-rmse:0.20884
[399]	validation_0-rmse:0.06199	validation_1-rmse:0.20830

base end val outcome
[0]	validation_0-rmse:0.36929	validation_1-rmse:0.36896
[100]	validation_0-rmse:0.11871	validation_1-rmse:0.21650
[200]	validation_0-rmse:0.09012	validation_1-rmse:0.21173
[300]	validation_0-rmse:0.07287	validation_1-rmse:0.21016
[399]	validation_0-rmse:0.06166	validation_1-rmse:0.20973


# dda 598
checkpoint 1: w.694 performance matches
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.210
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


# dda 612 portfolio features
checkpoint 6: w.694 rebuild all files, ready to merge?
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.210
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


checkpoint 5: w.694 add largest_coin_frac+total_usd_balance
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       128
Features per Window:      32

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.210
MAE:                      0.108

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


checkpoint 4: w.694 add balance|largest_coin_frac
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       124
Features per Window:      31

Core Metrics
-----------------------------------
R² Score:                 0.694
RMSE:                     0.210
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


checkpoint 3: w.689 tree depth back to 21
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.211
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


checkpoint 2: w.688 begin grid
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.212
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.815
RMSE:                     0.115
MAE:                      0.061


checkpoint 1: w.688 slightly too low?
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.212
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.815
RMSE:                     0.115
MAE:                      0.061


# dda X596 market cap aggregations
checkpoint 4: w.675 base case too low again
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.675
RMSE:                     0.216
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.060


checkpoint 3: w.679 baseline
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.215
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.114
MAE:                      0.060



# dda 596 market cap aggregations
checkpoint 5: w.689 ready to merge
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.211
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


checkpoint 4: w.689 min/max unadj
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.689
RMSE:                     0.211
MAE:                      0.109

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


checkpoint 3: w.687 concentration and stdv unadj
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       120
Features per Window:      30

Core Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.212
MAE:                      0.110

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.820
RMSE:                     0.113
MAE:                      0.060


largest_coin_usd/market_cap_filled
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.114
MAE:                      0.060



checkpoint 3: w.686 concentration_index/market_cap_unadj
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.212
MAE:                      0.110

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.821
RMSE:                     0.113
MAE:                      0.059


checkpoint 2: w. base case fixed drop patterns
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.215
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.114
MAE:                      0.060


checkpoint 1: w.667 too low base case
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       108
Features per Window:      27

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.815
RMSE:                     0.115
MAE:                      0.060


# dda 616 buys scenarios
checkpoint 2: w.679 best sells only helpful feature
best sells only
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.215
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.115
MAE:                      0.062



checkpoint 1: w.678 no scenarios
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


# dda 613 timing feature params
checkpoint 7: reenable new timing params
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061



checkpoint 6: backtrack to base timing params
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.666
RMSE:                     0.219
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.808
RMSE:                     0.117
MAE:                      0.063


checkpoint 5: w.678 I thought this was base...
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


checkpoint 4: w.678 min transacction size 30
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


checkpoint 3: w.678 min transacction size 3
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.817
RMSE:                     0.114
MAE:                      0.061


checkpoint 2: w. offset winsorization 0.05
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.676
RMSE:                     0.216
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.**062**


checkpoint 1: w.681 offset winsorization 0.03
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.818
RMSE:                     0.114
MAE:                      0.061


checkpoint 1: w.679 offset winsorization 0.01
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.818
RMSE:                     0.114
MAE:                      0.061


# dda 592 longer term timing features
checkpoint 7: w.681 final params
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       108
Features per Window:      27

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.115
MAE:                      0.061


checkpoint 6: w.679 price lag50,200 lead100; vol lag80 lead150
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.115
MAE:                      0.062


checkpoint 5: w.679 price lag50,200; vol lag80
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       104
Features per Window:      26

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.816
RMSE:                     0.114
MAE:                      0.061


checkpoint 4: w.679 price lag60,100 lead100; vol lag80 lead100,150
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.215
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


checkpoint 4: w.681 price lag60,100; vol lag80 lead150 plus more i didnt notice
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.811
RMSE:                     0.116
MAE:                      0.062


checkpoint 3: w.681 add vol_lag_80
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       112
Features per Window:      28

Core Metrics
-----------------------------------
R² Score:                 0.681
RMSE:                     0.214
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.115
MAE:                      0.062


checkpoint 2: w.678 price lag50,200; vol lead 100
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       108
Features per Window:      27

Core Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.115
MAE:                      0.062


checkpoint 1: w.677 price lag 50,200
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       100
Features per Window:      25

Core Metrics
-----------------------------------
R² Score:                 0.677
RMSE:                     0.215
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.814
RMSE:                     0.115
MAE:                      0.062


checkpoint 1: w.677 price lag 200
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       96
Features per Window:      24

Core Metrics
-----------------------------------
R² Score:                 0.676
RMSE:                     0.216
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.813
RMSE:                     0.115
MAE:                      0.062


base scenario, random seed changes from new columns in full training data df
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.666
RMSE:                     0.219
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.808
RMSE:                     0.117
MAE:                      0.063



# dda 606 perfect buys
checkpoint 2: w.669 with sells_best
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       96
Features per Window:      24

Core Metrics
-----------------------------------
R² Score:                 0.669
RMSE:                     0.218
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.808
RMSE:                     0.117
MAE:                      0.063


checkpoint 1: w.667 ready to test features
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062



# dda 609 more transfers features
checkpoint 3: w.667 fix greater than or equals
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


checkpoint 2: w.667 initial hold time not predictive
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


checkpoint 1: w.667 split transfers features function
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


# dda 608 training data
checkpoint 5: w.667 time to merge
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.002
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.429


checkpoint 4: w.667 feature selection w nulls
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062



checkpoint 3: w.667 feature selection into model
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062



checkpoint 2: w.667 indexify create_partitions
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


checkpoint 1: w.667 add ensure_index()
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.002
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.429


# dda 602 sell features
CHECKPOINT2 checkpoint 9 ready to merge
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


CHECKPOINT2 checkpoint 8 add sells to removal patterns
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


CHECKPOINT2 checkpoint 7 feature loop
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062



CHECKPOINT2 checkpoint 6 index added
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


CHECKPOINT2 checkpoint 5 ready to add index
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


CHECKPOINT2 checkpoint 4 updated transfers query
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061
****


CHECKPOINT2 checkpoint 3 add back profits_df join
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


CHECKPOINT2 with commented hybrid features
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


CHECKPOINT2 branch
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


checkpoint 1: remove quotes
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


# dda 605 scenario features
checkpoint 3
with all scenario features added


confirm base performance
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


checkpoint 2 with best/worst sells
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.669
RMSE:                     0.218
MAE:                      0.112

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.804
RMSE:                     0.118
MAE:                      0.065



checkpoint 1 new laptop
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


# dda 460 max unrealized gain/loss
checkpoint 5 refactor and add ratios base
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061



checkpoint 4 add sells_best/crypto_net_flows_crypto_inflows_winsorized
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       96
Features per Window:      24

Core Metrics
-----------------------------------
R² Score:                 0.669
RMSE:                     0.218
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.811
RMSE:                     0.116
MAE:                      0.062


with all
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       116
Features per Window:      29

Core Metrics
-----------------------------------
R² Score:                 0.664
RMSE:                     0.220
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.808
RMSE:                     0.117
MAE:                      0.063


checkpoint 3 ready to start ablation testing
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


checkpoint 2 preparing to integrate
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


checkpoint 1 ready to start
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


# dda 595 grid searchin
revisiting # dda 465 market cap features and # dda 593 better ablation
and found all performances had dropped to w.667, doesn't seem worth extended effort to dig deeper right now and may related to database changes
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061


checkpoint 2 need to return to old config
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92
Features per Window:      23

Core Metrics
-----------------------------------
R² Score:                 0.667
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.061

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.002
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.429


checkpoint 1 data to 2020
Model Performance Summary
===================================
Test Samples:             79,166
Number of Features:       330

Core Metrics
-----------------------------------
R² Score:                 0.756
RMSE:                     0.161
MAE:                      0.076

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.824
RMSE:                     0.093
MAE:                      0.056

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.161
95% Prediction Interval:  ±0.316

# dda 593 better ablation
checkpoint 2
alas TWR did not help
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       92

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.428


checkpoint 1 verifying model working normally
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       350

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.428



# dda 437 time weighted returns
checkpoint 3 toggle added
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       350

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062



checkpoint 2 working on grid search logic
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       350

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


checkpoint 1 ready to integrate code
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       275

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.219
95% Prediction Interval:  ±0.428


# dda 590 add feature counts
checkpoint 1 all set
Model Performance Summary
===================================
Test Samples:             56,708
Number of Features:       275

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


# dda 465 market cap features
checkpoint 6 base params
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


checkpoint 6 default fill to $1M
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.670
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


checkpoint 5 decrease imputation threshold to 0.1
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.219
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062



checkpoint 4 tested with features
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.668
RMSE:                     0.219
MAE:                      0.114

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.812
RMSE:                     0.116
MAE:                      0.062


checkpoint 3 loop is working
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.655
RMSE:                     0.224
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.061


checkpoint 2 add colname param
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.655
RMSE:                     0.224
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.061


checkpoint 1 remove twb
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.655
RMSE:                     0.224
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.061

checkpoint 1 with twb
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.661
RMSE:                     0.222
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.796
RMSE:                     0.120
MAE:                      0.065



# dda 589 twb indexify
checkpoint 1
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.661
RMSE:                     0.222
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.796
RMSE:                     0.120
MAE:                      0.065


# dda 586 performance features
checkpoint 6
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.661
RMSE:                     0.222
MAE:                      0.115

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.796
RMSE:                     0.120
MAE:                      0.065

checkpoint 5 with twb perf only
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.660
RMSE:                     0.222
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.794
RMSE:                     0.121
MAE:                      0.065

checkpoint 4 with twb
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.223
MAE:                      0.116

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.792
RMSE:                     0.122
MAE:                      0.066


checkpoint 3 filter combinations of features
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.655
RMSE:                     0.224
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.061


checkpoint 2 add crypto_inflows
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.654
RMSE:                     0.224
MAE:                      0.117

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.808
RMSE:                     0.117
MAE:                      0.062



checkpoint 1 grid search logic
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.651
RMSE:                     0.225
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


# dda 555 cash flows trading features
checkpoint 7 remove pre-2023
Model Performance Summary
===================================
Test Samples:             56,708

Core Metrics
-----------------------------------
R² Score:                 0.651
RMSE:                     0.225
MAE:                      0.118

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.809
RMSE:                     0.117
MAE:                      0.062


checkpoint 6 up to date
Model Performance Summary
===================================
Test Samples:             79,677

Core Metrics
-----------------------------------
R² Score:                 0.714
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.623
RMSE:                     0.136
MAE:                      0.068


checkpoint 5 add sells logic
Model Performance Summary
===================================
Test Samples:             79,677

Core Metrics
-----------------------------------
R² Score:                 0.714
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.623
RMSE:                     0.136
MAE:                      0.068

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.175
95% Prediction Interval:  ±0.343


checkpoint 4
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.130
MAE:                      0.069


checkpoint 3 more restructuring
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.130
MAE:                      0.069

checkpoint 2: restructuring in trading_features
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.130
MAE:                      0.069


checkpoint 1 working after rename cols
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.130
MAE:                      0.069



# dda 585 grid searchin
checkpoint 3 good to move on
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.710
RMSE:                     0.175
MAE:                      0.086

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.657
RMSE:                     0.130
MAE:                      0.069

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.000
Standard Dev of Residuals:0.175
95% Prediction Interval:  ±0.344



checkpoint 2
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.707
RMSE:                     0.176
MAE:                      0.087

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.626
RMSE:                     0.136
MAE:                      0.069


smaller gamma
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.706
RMSE:                     0.177
MAE:                      0.088

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.635
RMSE:                     0.134
MAE:                      0.069


checkpoint 1 deeper trees
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.705
RMSE:                     0.177
MAE:                      0.089

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.639
RMSE:                     0.133
MAE:                      0.067



# dda 579 ablation experiment
checkpoint 8
back to november 2024
increase max gap days to 30 from 14 0.08LR
Model Performance Summary
===================================
Test Samples:             79,724

Core Metrics
-----------------------------------
R² Score:                 0.693
RMSE:                     0.181
MAE:                      0.093

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.656
RMSE:                     0.130
MAE:                      0.065


checkpoint 7
faster learning rate 0.07
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.690
RMSE:                     0.318
MAE:                      0.162

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.683
RMSE:                     0.201
MAE:                      0.135


faster learning rate 0.06
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.690
RMSE:                     0.318
MAE:                      0.163

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.200
MAE:                      0.134


end trading features analysis
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.319
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.202
MAE:                      0.136



add trading features
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.319
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.678
RMSE:                     0.202
MAE:                      0.136

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.319
95% Prediction Interval:  ±0.625


remove correlated features .97
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.320
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.199
MAE:                      0.132


remove correlated features .99
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.320
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.199
MAE:                      0.132


remove 2022 window boundary
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.686
RMSE:                     0.320
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.688
RMSE:                     0.199
MAE:                      0.132


predict march 2024 with data through 2020
Model Performance Summary
===================================
Test Samples:             66,456

Core Metrics
-----------------------------------
R² Score:                 0.687
RMSE:                     0.319
MAE:                      0.164

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.685
RMSE:                     0.200
MAE:                      0.132


predict march 2024 with data through 2021
Model Performance Summary
===================================
Test Samples:             61,078

Core Metrics
-----------------------------------
R² Score:                 0.671
RMSE:                     0.336
MAE:                      0.178

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.679
RMSE:                     0.205
MAE:                      0.139


predict march 2024 with data through 2022
Model Performance Summary
===================================
Test Samples:             48,450

Core Metrics
-----------------------------------
R² Score:                 0.640
RMSE:                     0.369
MAE:                      0.204

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.715
RMSE:                     0.224
MAE:                      0.147


predict march 2024
Model Performance Summary
===================================
Test Samples:             33,727

Core Metrics
-----------------------------------
R² Score:                 0.578
RMSE:                     0.456
MAE:                      0.259

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.758
RMSE:                     0.261
MAE:                      0.155


volume longer sma
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.662
RMSE:                     0.180
MAE:                      0.106

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.769
RMSE:                     0.120
MAE:                      0.079


even higher learning rate, add trees
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.659
RMSE:                     0.181
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.768
RMSE:                     0.120
MAE:                      0.079


Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.659
RMSE:                     0.181
MAE:                      0.107

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.761
RMSE:                     0.121
MAE:                      0.081


raise learning rate
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.649
RMSE:                     0.184
MAE:                      0.110

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.741
RMSE:                     0.126
MAE:                      0.084


dropping most timing aggregations
Core Metrics
-----------------------------------
R² Score:                 0.645
RMSE:                     0.185
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.736
RMSE:                     0.128
MAE:                      0.085


with price and volume
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.641
RMSE:                     0.186
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.737
RMSE:                     0.128
MAE:                      0.086


with volume timing
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.640
RMSE:                     0.186
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.734
RMSE:                     0.128
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.186
95% Prediction Interval:  ±0.365


Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.638
RMSE:                     0.186
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.730
RMSE:                     0.129
MAE:                      0.086



add back timing features just prices
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.638
RMSE:                     0.186
MAE:                      0.113

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.730
RMSE:                     0.129
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.186
95% Prediction Interval:  ±0.366

checkpoint 6 modeling june 24
Model Performance Summary
===================================
Test Samples:             51,409

Core Metrics
-----------------------------------
R² Score:                 0.635
RMSE:                     0.187
MAE:                      0.111

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.720
RMSE:                     0.132
MAE:                      0.086



checkpoint 5
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493

Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.238
95% Prediction Interval:  ±0.466


checkpoint 4
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.206
RMSE:                     0.257
MAE:                      0.211

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.005
Standard Dev of Residuals:0.257
95% Prediction Interval:  ±0.504


Model Performance Summary
===================================
Test Samples:             59,156

Core Metrics
-----------------------------------
R² Score:                 0.647
RMSE:                     0.225
MAE:                      0.129

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.777
RMSE:                     0.131
MAE:                      0.075

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.225
95% Prediction Interval:  ±0.442


checkpoint 3
10 modeling balance
Model Performance Summary
===================================
Test Samples:             55,894

Core Metrics
-----------------------------------
R² Score:                 0.659
RMSE:                     0.224
MAE:                      0.128

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.766
RMSE:                     0.134
MAE:                      0.076



checkpoint 2 ablated
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.656
RMSE:                     0.224
MAE:                      0.128

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.758
RMSE:                     0.136
MAE:                      0.077

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.000
Standard Dev of Residuals:0.224
95% Prediction Interval:  ±0.438


checkpoint 1 cuttin em out
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.642
RMSE:                     0.228
MAE:                      0.133

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.720
RMSE:                     0.146
MAE:                      0.084




# dda 570 ablation analysis
checkpoint 6 ready to merge
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.238
95% Prediction Interval:  ±0.466


checkpoint 5 working with grid search
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


checkpoint 4 drop cols in pipeline
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


checkpoint 3 restyle base model
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


checkpoint 2
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493



checkpoint 1 inflows threhsold $10k
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.238
95% Prediction Interval:  ±0.466

coin
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208




checkpoint 1 inflows threhsold $5k
Model Performance Summary
===================================
Test Samples:             58,573

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.235
MAE:                      0.146

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.705
RMSE:                     0.147
MAE:                      0.085

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.235
95% Prediction Interval:  ±0.461

coin model
Model Performance Summary
===================================
Test Samples:             457

Core Metrics
-----------------------------------
R² Score:                 0.136
RMSE:                     0.274
MAE:                      0.232

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.010
Standard Dev of Residuals:0.274
95% Prediction Interval:  ±0.537


# dda 551 confidence metrics for scores
checkpoint 4 prepare to merge
wallet
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

coin
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


checkpoint 3 try with confidence
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.213
RMSE:                     0.256
MAE:                      0.211

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.002
Standard Dev of Residuals:0.256
95% Prediction Interval:  ±0.502


checkpoint 2 coins 1 date
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


checkpoint 1 wallet base
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.000
Standard Dev of Residuals:0.238
95% Prediction Interval:  ±0.466



# dda 520 coin model multiple dates
checkpoint 1 with 2 dates
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.229
RMSE:                     0.254
MAE:                      0.210

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.002
Standard Dev of Residuals:0.254
95% Prediction Interval:  ±0.497


checkpoint 1 with 1 date
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493



# dda 575 coin model optimizations
checkpoint 1 possibly overtuned
Model Performance Summary
===================================
Test Samples:             442

Core Metrics
-----------------------------------
R² Score:                 0.239
RMSE:                     0.252
MAE:                      0.208

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.252
95% Prediction Interval:  ±0.493


# dda 574 hybrid run
checkpoint 5 wallet min coins 0
Model Performance Summary
===================================
Test Samples:             53,868

Core Metrics
-----------------------------------
R² Score:                 0.612
RMSE:                     0.238
MAE:                      0.147

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.702
RMSE:                     0.151
MAE:                      0.086

checkpoint 5 coin model min coins 0
Model Performance Summary
===================================
Test Samples:             402

Core Metrics
-----------------------------------
R² Score:                 0.075
RMSE:                     0.406
MAE:                      0.271


checkpoint 4 coin with hybridized
Model Performance Summary
===================================
Test Samples:             364

Core Metrics
-----------------------------------
R² Score:                 0.062
RMSE:                     0.429
MAE:                      0.283

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.007
Standard Dev of Residuals:0.429
95% Prediction Interval:  ±0.841


checkpoint 3 wallet I don't know why it changed
Model Performance Summary
===================================
Test Samples:             66,365

Core Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.176
MAE:                      0.102

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.748
RMSE:                     0.114
MAE:                      0.063

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.176
95% Prediction Interval:  ±0.344


checkpoint 2 wallet hybridized after timing min transaction fix
Model Performance Summary
===================================
Test Samples:             66,365

Core Metrics
-----------------------------------
R² Score:                 0.741
RMSE:                     0.177
MAE:                      0.103

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.115
MAE:                      0.064

Residuals Analysis
-----------------------------------
Mean of Residuals:        0.001
Standard Dev of Residuals:0.177
95% Prediction Interval:  ±0.347


checkpoint 1 wallet hybridized
Model Performance Summary
===================================
Test Samples:             66,365

Core Metrics
-----------------------------------
R² Score:                 0.743
RMSE:                     0.177
MAE:                      0.103

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.745
RMSE:                     0.115
MAE:                      0.064

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.000
Standard Dev of Residuals:0.177
95% Prediction Interval:  ±0.347

checkpoint 1 wallet after timing min transaction fix
Model Performance Summary
===================================
Test Samples:             15,470

Core Metrics
-----------------------------------
R² Score:                 0.352
RMSE:                     0.362
MAE:                      0.254

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.768
RMSE:                     0.176
MAE:                      0.100

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.362
95% Prediction Interval:  ±0.709


# dda 572 rerun
checkpoint 1
Coin Model Performance Summary
===================================
Test Samples:             226

Core Metrics
-----------------------------------
R² Score:                 0.110
RMSE:                     0.414
MAE:                      0.284

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.004
Standard Dev of Residuals:0.414
95% Prediction Interval:  ±0.811

Wallet Model Performance Summary
===================================
Test Samples:             15,470

Core Metrics
-----------------------------------
R² Score:                 0.353
RMSE:                     0.361
MAE:                      0.253

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.769
RMSE:                     0.175
MAE:                      0.099

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.361
95% Prediction Interval:  ±0.708


# dda 557 rerun

checkpoint 4
Model Performance Summary
===================================
Test Samples:             15,470

Core Metrics
-----------------------------------
R² Score:                 0.353
RMSE:                     0.361
MAE:                      0.253

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.769
RMSE:                     0.175
MAE:                      0.099

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.361
95% Prediction Interval:  ±0.708



checkpoint 3
Model Performance Summary
===================================
Test Samples:             15,470

Core Metrics
-----------------------------------
R² Score:                 0.353
RMSE:                     0.361
MAE:                      0.253

Inactive Wallets Cohort Metrics
-----------------------------------
R² Score:                 0.769
RMSE:                     0.175
MAE:                      0.099

Residuals Analysis
-----------------------------------
Mean of Residuals:        -0.001
Standard Dev of Residuals:0.361
95% Prediction Interval:  ±0.708


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
