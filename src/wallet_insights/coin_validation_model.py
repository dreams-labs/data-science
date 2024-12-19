import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_insights.coin_validation_analysis as wicv
import wallet_insights.wallet_model_evaluation as wime
import utils as u


# pylint:disable=invalid-name  # X doesn't conform to snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def prepare_features_and_targets(
    modeling_profits_df: pd.DataFrame,
    modeling_market_data_df: pd.DataFrame,
    wallet_scores_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepares modeling dataset using modeling period data instead of validation period.

    Params:
    - coin_validation_df (DataFrame): Contains wallet metrics and market stats
    - modeling_profits_df (DataFrame): Raw profits data from modeling period
    - modeling_market_data_df (DataFrame): Market data from modeling period
    - wallet_scores_df (DataFrame): Wallet scores from model predictions

    Returns:
    - modeling_df (DataFrame): Prepared modeling data with features and target
    """
    # 1. Calculate modeling period coin performance
    coin_performance_df = wicv.calculate_coin_performance(
        modeling_market_data_df,
        wallets_config['training_data']['modeling_period_start'],
        wallets_config['training_data']['modeling_period_end']
    )

    # 2. Calculate modeling period wallet metrics
    modeling_wallet_metrics = wicv.calculate_coin_metrics_from_wallet_scores(
        modeling_profits_df,
        wallet_scores_df
    )

    # 3. Create feature matrix
    coin_modeling_df = modeling_wallet_metrics.join(
        coin_performance_df[['coin_return', 'market_cap_filled']],
        how='inner'
    )

    # 4. Add engineered features
    coin_modeling_df['log_market_cap'] = np.log1p(coin_modeling_df['market_cap_filled'])
    coin_modeling_df['log_total_balance'] = np.log1p(coin_modeling_df['total_balance'])
    coin_modeling_df['log_avg_wallet_balance'] = np.log1p(coin_modeling_df['avg_wallet_balance'])
    coin_modeling_df['wallet_concentration'] = coin_modeling_df['top_wallet_balance'] / coin_modeling_df['total_balance']
    coin_modeling_df['wallet_activity'] = coin_modeling_df['total_wallets'] * coin_modeling_df['score_confidence']

    # 5. Calculate ratios and interaction terms
    coin_modeling_df['top_wallet_score_ratio'] = (
        coin_modeling_df['weighted_avg_score'] / coin_modeling_df['mean_score']
    )
    coin_modeling_df['balance_weighted_confidence'] = (
        coin_modeling_df['score_confidence'] *
        coin_modeling_df['top_wallet_balance_pct']
    )

    # 6. Add market cap segment indicators
    coin_modeling_df['is_micro_cap'] = coin_modeling_df['market_cap_filled'] < 1e6
    coin_modeling_df['is_small_cap'] = (
        (coin_modeling_df['market_cap_filled'] >= 1e6) &
        (coin_modeling_df['market_cap_filled'] < 35e6)
    )
    coin_modeling_df['is_mid_cap'] = coin_modeling_df['market_cap_filled'] >= 35e6

    # 7. Winsorize target to reduce impact of outliers
    coin_modeling_df['coin_return_raw'] = coin_modeling_df['coin_return']
    coin_modeling_df['coin_return'] = u.winsorize(coin_modeling_df['coin_return'],0.05)

    return coin_modeling_df

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
