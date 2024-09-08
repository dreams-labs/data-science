import sys
import os
import pandas as pd
from dotenv import load_dotenv
from dreams_core import core as dc

# import training_data python functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data as td

load_dotenv()
logger = dc.setup_logger()

def test_retrieve_transfers_data_no_duplicates():
    logger.info(f"Checking retrieve_transfers_data() for duplicates...")
    # Example modeling period start date
    modeling_period_start = '2024-03-01'

    # Retrieve the transfers data using your function
    transfers_df = td.retrieve_transfers_data(modeling_period_start)

    # Drop duplicates
    deduped = transfers_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()

    # Assert that the original and deduped DataFrames have the same length
    logger.info(f"Original transfers_df length: {len(transfers_df)}, Deduplicated: {len(deduped)}")
    assert len(transfers_df) == len(deduped), "There are duplicate rows based on coin_id, wallet_address, and date"
