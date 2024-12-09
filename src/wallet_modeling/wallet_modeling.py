"""
Calculates metrics aggregated at the wallet level
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase


# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



def generate_target_variables(wallets_df):
    """
    Generates various target variables for modeling wallet performance.

    Parameters:
    - wallets_df: pandas DataFrame with columns ['net_gain', 'invested']
    - winsorization: how much the returns column should be winsorized

    Returns:
    - DataFrame with additional target variables
    """
    metrics_df = wallets_df[['invested','net_gain']].copy().round(6)
    returns_winsorization = wallets_config['modeling']['returns_winsorization']
    epsilon = 1e-10

    # Calculate base return
    metrics_df['return'] = np.where(abs(metrics_df['invested']) == 0,0,
                                    metrics_df['net_gain'] / metrics_df['invested'])

    # Apply winsorization
    if returns_winsorization > 0:
        metrics_df['return'] = u.winsorize(metrics_df['return'],returns_winsorization)

    # Risk-Adjusted Dollar Return
    metrics_df['risk_adj_return'] = metrics_df['net_gain'] * \
        (1 + np.log10(metrics_df['invested'] + epsilon))

    # Normalize returns
    metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
        (metrics_df['return'].max() - metrics_df['return'].min())

    # Normalize logged investments
    log_invested = np.log10(metrics_df['invested'] + epsilon)
    metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
        (log_invested.max() - log_invested.min())

    # Performance score
    metrics_df['performance_score'] = (0.6 * metrics_df['norm_return'] +
                                     0.4 * metrics_df['norm_invested'])

    # Log-weighted return
    metrics_df['log_weighted_return'] = metrics_df['return'] * \
        np.log10(metrics_df['invested'] + epsilon)

    # Hybrid score (combining absolute and relative performance)
    max_gain = metrics_df['net_gain'].abs().max()
    metrics_df['norm_gain'] = metrics_df['net_gain'] / max_gain
    metrics_df['hybrid_score'] = (metrics_df['norm_gain'] +
                                metrics_df['norm_return']) / 2

    # Size-adjusted rank
    # Create mask for zero values
    zero_mask = metrics_df['invested'] == 0

    # Create quartiles series initialized with 'q0' for zero values
    quartiles = pd.Series('q0', index=metrics_df.index)

    # Calculate quartiles for non-zero values
    non_zero_quartiles = pd.qcut(metrics_df['invested'][~zero_mask],
                                q=4,
                                labels=['q1', 'q2', 'q3', 'q4'])

    # Assign the quartiles to non-zero values
    quartiles[~zero_mask] = non_zero_quartiles

    # Calculate size-adjusted rank within each quartile
    metrics_df['size_adjusted_rank'] = metrics_df.groupby(quartiles)['return'].rank(pct=True)


    # Clean up intermediate columns
    cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain']
    metrics_df = metrics_df.drop(columns=[c for c in cols_to_drop
                                        if c in metrics_df.columns])

    return metrics_df.round(6)



def train_xgb_model(modeling_df, return_data=True):
    """
    Trains an XGBoost model using the provided DataFrame and configuration.

    Args:
        modeling_df (pd.DataFrame): Input DataFrame for modeling
        return_data (bool): If True, returns additional data for analysis

    Returns:
        dict: Dictionary containing the pipeline and optionally training data
    """
    # Make copy of input DataFrame
    df = modeling_df.copy()

    # Drop columns if configured
    if wallets_config['modeling']['drop_columns']:
        existing_columns = [col for col in wallets_config['modeling']['drop_columns']
                          if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)

    # Prepare features and target
    target_var = wallets_config['modeling']['target_variable']
    X = df.drop(target_var, axis=1)
    y = df[target_var]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create preprocessing steps
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )

    # Initialize XGBoost model
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1
    )

    # Create and fit pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb)
    ])

    pipeline.fit(X_train, y_train)

    # Prepare return dictionary
    result = {'pipeline': pipeline}

    # Optionally return data for analysis
    if return_data:
        y_pred = pipeline.predict(X_test)
        result.update({
            'X': X,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
        })

    return result
