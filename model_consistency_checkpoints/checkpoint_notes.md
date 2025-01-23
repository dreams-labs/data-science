# dda 460 max unrealized gain/loss
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
