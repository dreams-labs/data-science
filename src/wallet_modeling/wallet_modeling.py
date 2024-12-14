"""
Calculates metrics aggregated at the wallet level
"""
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint:disable=invalid-name  # X_test isn't camelcase


# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


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
