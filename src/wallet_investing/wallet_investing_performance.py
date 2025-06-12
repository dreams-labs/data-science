"""
Analyzes performance of buy targets across multiple epochs.
"""
import logging
import pandas as pd

# Local module imports
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



def compute_epoch_buy_metrics(all_epochs_trading_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes buy and overall performance metrics for each epoch.

    Params:
    - all_epochs_trading_df (DataFrame): trading data indexed by epoch_start_date

    Returns:
    - buys_metrics_df (DataFrame): epoch-level performance metrics
    """
    epoch_metrics_list = []

    for epoch_start in all_epochs_trading_df.index.get_level_values('epoch_start_date').unique():

        trading_df = (all_epochs_trading_df[
            all_epochs_trading_df.index.get_level_values('epoch_start_date')== epoch_start
        ]).copy()

        # Add winsorized return column
        trading_df['coin_return_wins'] = u.winsorize(trading_df['coin_return'], 0.01)

        # Performance of bought coins
        coins_bought = trading_df[trading_df['is_buy']]['coin_return'].count()
        median_buy_return = trading_df[trading_df['is_buy']]['coin_return'].median()
        mean_buy_return = trading_df[trading_df['is_buy']]['coin_return'].mean()
        wins_buy_return = trading_df[trading_df['is_buy']]['coin_return_wins'].mean()
        best_buy_return = trading_df[trading_df['is_buy']]['coin_return'].max()

        # Overall performance
        median_overall_return = trading_df['coin_return'].median()
        mean_overall_return = trading_df['coin_return'].mean()
        wins_overall_return = trading_df['coin_return_wins'].mean()

        epoch_metrics_list.append({
            'epoch_start_date': epoch_start.strftime('%Y-%m-%d'),
            'coins_bought': coins_bought,
            'median_buy_return': median_buy_return,
            'wins_buy_return': wins_buy_return,
            'mean_buy_return': mean_buy_return,
            'best_buy_return': best_buy_return,
            'median_overall_return': median_overall_return,
            'wins_overall_return': wins_overall_return,
            'mean_overall_return': mean_overall_return,
        })

    buys_metrics_df = pd.DataFrame(epoch_metrics_list)
    return buys_metrics_df
