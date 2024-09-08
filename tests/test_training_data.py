"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=E0401
# pylint: disable=C0413

import sys
import os
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# import training_data python functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data as td

load_dotenv()
logger = dc.setup_logger()

@pytest.mark.slow  # Add this marker to indicate a slow test
def test_retrieve_transfers_data_no_duplicates():
    """
    retrieves transfers_df and performs data quality checks
    """
    logger.info("Testing transfers_df from retrieve_transfers_data()...")

    # Example modeling period start date
    modeling_period_start = '2024-03-01'

    # Retrieve transfers_df
    transfers_df = td.retrieve_transfers_data(modeling_period_start)

    # Test 1: there should be no duplicate records
    # --------------------------------------------
    # Assert that the original and deduped DataFrames have the same length
    deduped_df = transfers_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
    logger.info(f"Original transfers_df length: {len(transfers_df)}, Deduplicated: {len(deduped_df)}")

    assert len(transfers_df) == len(deduped_df), "There are duplicate rows based on coin_id, wallet_address, and date"


    # Test 2: all coin-wallet pairs should have a record at the end of the training period
    # ------------------------------------------------------------------------------------
    # count the number of coin-wallet pairs that exist during the training period
    transfers_df_filtered = transfers_df[transfers_df['date']<modeling_period_start]
    deduped_df = transfers_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
    pairs_in_training_period = len(deduped_df)

    # confirm that all pairs have a record on the last day of training to ensure accurate profitability calculations
    training_period_end = pd.to_datetime(modeling_period_start) - pd.Timedelta(1, 'day')
    period_end_df = transfers_df[transfers_df['date']==training_period_end]
    pairs_at_training_period_end = len(period_end_df)

    logger.info(f"Found {pairs_in_training_period} total pairs in training period with {pairs_at_training_period_end} having data at period end.")

    assert pairs_in_training_period == pairs_at_training_period_end, "Not all training data coin-wallet pairs have a record at the end of the training period"