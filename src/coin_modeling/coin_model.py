"""
Models coin performance
"""
# pylint:disable=invalid-name  # X_test isn't camelcase
import logging
from typing import Dict, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Set up logger at the module level
logger = logging.getLogger(__name__)




def train_coin_prediction_model(modeling_df: pd.DataFrame):
    """
    Trains coin prediction model with proper validation and feature selection.
    """
    # 1. Define features to use
    feature_cols = [
        # Market cap features
        'log_market_cap', 'is_micro_cap', 'is_small_cap', 'is_mid_cap',

        # Wallet metrics
        'weighted_avg_score', 'mean_score', 'score_std', 'score_confidence',
        'top_wallet_score_ratio',

        # Balance metrics
        'log_total_balance', 'log_avg_wallet_balance',
        'top_wallet_balance_pct', 'wallet_concentration',

        # Activity metrics
        'total_wallets', 'wallet_activity', 'balance_weighted_confidence'
    ]

    # 2. Prepare train/test split
    X = modeling_df[feature_cols].values
    y = modeling_df['coin_return'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=modeling_df['is_mid_cap']
    )

    # 3. Create model with proper parameters
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42
    )

    # 4. Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5. Create evaluator with feature names
    evaluator = wime.RegressionEvaluator(
        y_train=y_train,
        y_true=y_test,
        y_pred=y_pred,
        model=model,
        feature_names=feature_cols
    )

    return model, evaluator
