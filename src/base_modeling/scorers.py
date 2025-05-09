# pylint:disable=wrong-import-position
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=unused-argument  # X and y params are always needed for pipeline structure

# Local module imports
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# -----------------------------------
#          Scorer Functions
# -----------------------------------

def custom_neg_rmse_scorer(estimator, X, y):
    """
    Custom scorer that transforms y before computing RMSE.
    Applies the estimator's y_pipeline to extract the proper target.
    Returns negative RMSE for grid search scoring.
    """
    y_trans = estimator.y_pipeline.transform(y)
    y_pred = estimator.predict(X)
    rmse = root_mean_squared_error(y_trans, y_pred)
    return -rmse


def custom_r2_scorer(estimator, X, y):
    """
    Custom scorer for R² that first applies the pipeline's y transformation.

    Parameters:
      estimator: The fitted MetaPipeline, which includes a y_pipeline.
      X (DataFrame or array): Feature data.
      y (DataFrame or array): The raw target data.

    Returns:
      R² score computed on the transformed target and predictions.
    """
    y_trans = estimator.y_pipeline.transform(y)
    y_pred = estimator.predict(X)
    return r2_score(y_trans, y_pred)


def validation_r2_scorer(wallet_model):
    """
    Factory function that returns a custom scorer using validation data.

    Params:
    - wallet_model: WalletModel instance containing validation data

    Returns:
    - scorer function compatible with scikit-learn
    """
    def scorer(estimator, X=None, y=None):
        """Score using the validation data instead of provided X and y"""
        if wallet_model.X_validation is None or wallet_model.validation_wallet_features_df is None:
            raise ValueError("Validation data not set in wallet_model")

        # Transform y using the pipeline
        y_trans = estimator.y_pipeline.transform(wallet_model.validation_wallet_features_df)

        # Get predictions
        y_pred = estimator.predict(wallet_model.X_validation)

        # Calculate and return R2 score
        return r2_score(y_trans, y_pred)

    return scorer


def validation_auc_scorer(wallet_model):
    """
    Factory function that returns a custom scorer using validation data and ROC AUC.

    Params:
    - wallet_model: WalletModel instance containing validation data

    Returns:
    - scorer function compatible with scikit-learn that computes ROC AUC.
    """
    def scorer(estimator, X=None, y=None):
        if wallet_model.X_validation is None or wallet_model.validation_wallet_features_df is None:
            raise ValueError("Validation data not set in wallet_model")

        # Transform true labels using the y_pipeline
        y_true = estimator.y_pipeline.transform(wallet_model.validation_wallet_features_df)

        # Transform validation features for probability prediction
        X_val_trans = estimator.x_transformer_.transform(wallet_model.X_validation)

        # Predict class probabilities and select the positive class index
        probas = estimator.estimator.predict_proba(X_val_trans)
        pos_idx = list(estimator.estimator.classes_).index(1)

        # Compute and return ROC AUC
        return roc_auc_score(y_true, probas[:, pos_idx])

    return scorer


def validation_top_percentile_returns_scorer(wallet_model, top_pct: float):
    """
    Factory function that returns a custom scorer computing the mean actual return
    of the top n% of wallets by predicted probability on the validation set.

    Params:
    - wallet_model: WalletModel instance containing validation data
    - top_pct (float): Fraction (0 < top_pct <= 1) representing the top n% to evaluate

    Returns:
    - scorer function compatible with scikit-learn that computes mean return.
    """
    def scorer(estimator, X=None, y=None):
        # Ensure validation data is available
        if wallet_model.X_validation is None or wallet_model.validation_wallet_features_df is None:
            raise ValueError("Validation data not set in wallet_model")

        # Get actual returns
        target_var = wallet_model.modeling_config['target_variable']
        returns = wallet_model.validation_wallet_features_df[target_var].reindex(wallet_model.X_validation.index)
        returns = u.winsorize(returns,0.001)

        # Predict class probabilities for positive class
        X_val_trans = estimator.x_transformer_.transform(wallet_model.X_validation)
        probas = estimator.estimator.predict_proba(X_val_trans)
        pos_idx = list(estimator.estimator.classes_).index(1)
        probs = probas[:, pos_idx]

        # Align probabilities and returns on the same index before combining
        proba_series = pd.Series(probs, index=wallet_model.X_validation.index, name='proba')
        df = pd.concat([proba_series, returns.rename('ret')], axis=1).dropna()

        # Compute cutoff for top_pct
        cutoff = np.percentile(df['proba'], 100 * (1 - top_pct))
        top_df = df[df['proba'] >= cutoff]

        # Return mean actual return of top slice; if empty, return nan
        return top_df['ret'].mean() if not top_df.empty else float('nan')

    return scorer


def validation_top_scores_returns_scorer(wallet_model):
    """
    Factory function returning a scorer that computes the mean actual return
    of wallets with predicted probability above the model's y_pred_threshold.

    Params:
    - wallet_model: WalletModel instance containing validation data

    Returns:
    - scorer function for sklearn, computing mean return for predictions >= threshold.
    """
    def scorer(estimator, X=None, y=None):
        # Ensure validation data is available
        if wallet_model.X_validation is None or wallet_model.validation_wallet_features_df is None:
            raise ValueError("Validation data not set in wallet_model")

        # Get actual returns aligned to validation index
        target_var = wallet_model.modeling_config['target_variable']
        returns = wallet_model.validation_wallet_features_df[target_var].reindex(wallet_model.X_validation.index)

        # Mild winsorization to limit impact of major outliers
        returns = u.winsorize(returns,0.005)

        # Predict class probabilities for positive class
        X_val_trans = estimator.x_transformer_.transform(wallet_model.X_validation)
        probas = estimator.estimator.predict_proba(X_val_trans)
        pos_idx = list(estimator.estimator.classes_).index(1)
        probs = probas[:, pos_idx]

        # Combine into DataFrame and drop missing
        df = pd.DataFrame({'proba': probs, 'ret': returns}).dropna()

        # Retrieve threshold from model config
        threshold = wallet_model.modeling_config.get('y_pred_threshold')
        if threshold is None:
            raise ValueError("y_pred_threshold not set in modeling_config")

        # Select all records with probability >= threshold
        selected = df[df['proba'] >= threshold]

        # Return mean actual return; nan if none meet threshold
        return selected['ret'].mean() if not selected.empty else float('nan')

    return scorer
