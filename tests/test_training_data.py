"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures

import sys
import os
import yaml
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# import training_data python functions
# pylint: disable=E0401 # can't find import
# pylint: disable=C0413 # import not at top of doc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data as td # type: ignore[reportMissingImports]

load_dotenv()
logger = dc.setup_logger()





# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------------- #
# calculate_wallet_profitability() unit tests
# ---------------------------------------- #

@pytest.fixture
def sample_transfers_df():
    """
    Create a sample transfers DataFrame for testing.
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH', 'BTC', 'ETH', 'MYRO', 'MYRO', 'MYRO', 
                    'BTC', 'ETH', 'BTC', 'ETH', 'MYRO'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet1', 'wallet2', 'wallet2', 'wallet2', 'wallet3', 'wallet3', 'wallet3',
                           'wallet1', 'wallet1', 'wallet2', 'wallet2', 'wallet3'],
        'date': [
            '2023-01-01', '2023-02-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-02-01', 
            '2023-01-01', '2023-02-01', '2023-03-01',
            '2023-04-01', '2023-04-01', '2023-04-01', '2023-04-01', '2023-04-01'
        ],
        'net_transfers': [10.0, 5, 100, 50, 20, 25, 1000, 500, -750,
                          0, 0, 0, -10, 0],
        'balance': [10.0, 15, 100, 50, 20, 75, 1000, 1500, 750,
                    15, 100, 20, 65, 750]
    }
    df = pd.DataFrame(data)

    # Convert coin_id to categorical and date to datetime
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])

    return df

@pytest.fixture
def sample_prices_df():
    """
    Create a sample prices DataFrame for testing.
    """
    data = {
        'date': [
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'
        ],
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH', 'ETH', 'MYRO', 'MYRO', 'MYRO', 'MYRO'],
        'price': [20000.0, 21000, 22000, 23000, 1500, 1600, 1700, 1800, 10, 15, 12, 8]
    }
    df = pd.DataFrame(data)

    # Convert coin_id to categorical and date to datetime
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])

    return df


@pytest.mark.unit
def test_calculate_wallet_profitability_profitability(sample_transfers_df, sample_prices_df):
    """
    Test profitability calculations for multiple wallets and coins.
    """
    profits_df = td.prepare_profits_data(sample_transfers_df, sample_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Check profitability for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(10000)  # (21000 - 20000) * 10
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_cumulative'].values[0] == pytest.approx(10000)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(30000)  # (23000 - 21000) * 15
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(40000)  # 10000 + 15000 + 15000

    # Check profitability for wallet2, ETH
    wallet2_eth = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'ETH')]
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (1600 - 1500) * 50
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_cumulative'].values[0] == pytest.approx(5000)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(15000)  # (1800 - 1600) * 75
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(20000)  # 5000 + 15000

    # Check profitability for wallet3, MYRO
    wallet3_myro = result[(result['wallet_address'] == 'wallet3') & (result['coin_id'] == 'MYRO')]
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (15 - 10) * 1000
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_change'].values[0] == pytest.approx(-4500)  # (12 - 15) * 1500
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_cumulative'].values[0] == pytest.approx(500)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(-3000)  # (8 - 12) * 750
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(-2500)  # 500 - 3000

# pylint: disable=R0914 # too many local variables
@pytest.mark.unit
def test_calculate_wallet_profitability_usd_calculations(sample_transfers_df, sample_prices_df):
    """
    Test USD-related calculations (inflows, balances, total return).
    """
    profits_df = td.prepare_profits_data(sample_transfers_df, sample_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Check USD calculations for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    initial_investment_wallet1 = 10 * 20000  # 10 BTC * $20,000
    second_investment_wallet1 = 5 * 21000    # 5 BTC * $21,000
    total_investment_wallet1 = initial_investment_wallet1 + second_investment_wallet1
    expected_balance_wallet1 = 15 * 23000    # 15 BTC * $23,000
    expected_total_return_wallet1 = 40000 / total_investment_wallet1

    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet1)

    # Check USD calculations for wallet2, ETH
    wallet2_eth = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'ETH')]
    initial_investment_wallet2 = 50 * 1500  # 50 ETH * $1,500
    second_investment_wallet2 = 25 * 1600  # 25 ETH * $1,600
    total_investment_wallet2 = initial_investment_wallet2 + second_investment_wallet2
    expected_balance_wallet2 = 65 * 1800  # 65 ETH * $1,800
    expected_total_return_wallet2 = 20000 / total_investment_wallet2

    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet2)

    # Check USD calculations for wallet3, MYRO
    wallet3_myro = result[(result['wallet_address'] == 'wallet3') & (result['coin_id'] == 'MYRO')]
    initial_investment_wallet3 = 1000 * 10  # 1000 MYRO * $10
    second_investment_wallet3 = 500 * 15   # 500 MYRO * $15
    total_investment_wallet3 = initial_investment_wallet3 + second_investment_wallet3
    expected_balance_wallet3 = 750 * 8  # 750 MYRO * $8
    current_value_wallet3 = 750 * 8  # Current holdings: 750 MYRO * $8
    sold_value_wallet3 = 750 * 12     # Sold tokens: 750 MYRO * $12
    profit_wallet3 = current_value_wallet3 + sold_value_wallet3 - total_investment_wallet3
    expected_total_return_wallet3 = profit_wallet3 / total_investment_wallet3

    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet3)


@pytest.fixture
def price_data_transfers_df():
    """
    Create a sample transfers DataFrame for testing interactions with price data.
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'MYRO', 'MYRO'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2'],
        'date': [
            '2023-03-01', '2023-04-01',  # BTC wallet1 buys during training period
            '2023-02-20', '2023-03-10'   # MYRO wallet2 buys during training period (before price data)
        ],
        'net_transfers': [10.0, -10.0, 1000.0, -1000.0],  # Buys and sells
        'balance': [10.0, 0.0, 1000.0, 0.0]  # Balance adjustments after buy and sell
    }
    df = pd.DataFrame(data)
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])
    return df

@pytest.fixture
def price_data_prices_df():
    """
    Create a sample prices DataFrame for testing interactions with price data.
    """
    data = {
        'date': ['2023-03-15', '2023-04-01', '2023-03-15', '2023-04-01'],
        'coin_id': ['BTC', 'BTC', 'MYRO', 'MYRO'],
        'price': [22000.0, 23000, 12, 15]  # Price data available starting from 2023-03-15
    }
    df = pd.DataFrame(data)
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])
    return df

@pytest.mark.unit
def test_price_data_interactions(price_data_transfers_df, price_data_prices_df):
    """
    Test interactions between wallet transfers and available price data.
    """
    profits_df = td.prepare_profits_data(price_data_transfers_df, price_data_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Test scenario: Buy during training period before price data, sell after price data
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    wallet1_btc_profits = (23000-22000) * 10
    assert wallet1_btc.iloc[0]['date'] == pd.Timestamp('2023-03-15')  # First row should reflect earliest price data
    assert wallet1_btc.iloc[0]['profits_cumulative'] == 0  # No profit on initial transfer in
    assert wallet1_btc.iloc[1]['profits_cumulative'] == wallet1_btc_profits  # Profitability calculation should be valid

    # Test scenario: Buy and sell before price data is available
    wallet2_myro = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'MYRO')]
    assert wallet2_myro.empty  # No rows should exist, as no price data was available for the transaction



# ---------------------------------------- #
# classify_shark_wallets() unit tests
# ---------------------------------------- #

@pytest.fixture
def sample_shark_wallets_shark_coins_df():
    """
    mock version of shark_coins_df to test calculation logic
    """
    data = {
        'wallet_address': ['wallet_1', 'wallet_1', 'wallet_2', 'wallet_2', 'wallet_3'],
        'coin_id': ['coin_1', 'coin_2', 'coin_1', 'coin_3', 'coin_4'],
        'is_shark': [True, False, True, True, False]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_shark_wallets_modeling_config():
    """
    mock of config.yaml to test calculation logic
    """
    return {
        'shark_wallet_type': 'is_shark',
        'shark_wallet_min_coins': 2,
        'shark_wallet_min_shark_rate': 0.5
    }

@pytest.mark.unit
def test_total_coin_calculation(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config):
    """
    Test 2: Verify total coins calculation for each wallet.
    """
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    total_coins = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_1']['total_coins'].values[0]
    assert total_coins == 2, f"Expected 2, got {total_coins}"

@pytest.mark.unit
def test_shark_coin_calculation(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config):
    """
    Test 3: Verify shark coins calculation for each wallet.
    """
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    shark_coins = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_2']['shark_coins'].values[0]
    assert shark_coins == 2, f"Expected 2, got {shark_coins}"

@pytest.mark.unit
def test_shark_rate_calculation(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config):
    """
    Test 4: Verify shark rate calculation for each wallet.
    """
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    shark_rate = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_2']['shark_rate'].values[0]
    assert shark_rate == 1.0, f"Expected 1.0, got {shark_rate}"

@pytest.mark.unit
def test_megashark_classification(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config):
    """
    Test 5: Verify megashark classification based on minimum coins and shark rate thresholds.
    """
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    is_shark = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_2']['is_shark'].values[0]
    assert is_shark, "Expected wallet_2 to be classified as megashark"

@pytest.mark.unit
def test_non_shark_wallet_handling(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config):
    """
    Test 6: Verify handling of non-shark wallets (shark_coins = 0, shark_rate = 0).
    """
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    shark_coins = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_3']['shark_coins'].values[0]
    shark_rate = shark_wallets_df[shark_wallets_df['wallet_address'] == 'wallet_3']['shark_rate'].values[0]
    assert shark_coins == 0, f"Expected 0, got {shark_coins}"
    assert shark_rate == 0, f"Expected 0, got {shark_rate}"

@pytest.mark.parametrize("min_coins, min_shark_rate, expected_sharks", [
    (2, 0.6, ['wallet_2']),  # Test case where wallet_2 is a megashark
    (1, 0.5, ['wallet_1', 'wallet_2']),  # Test case where wallet_1 and wallet_2 are megasharks
])
@pytest.mark.unit
def test_varying_inputs(sample_shark_wallets_shark_coins_df, min_coins, min_shark_rate, expected_sharks):
    """
    Test 7: Verify classification with varying inputs for min_coins and min_shark_rate.
    """
    sample_shark_wallets_modeling_config = {
        'shark_wallet_type': 'is_shark',
        'shark_wallet_min_coins': min_coins,
        'shark_wallet_min_shark_rate': min_shark_rate
    }
    shark_wallets_df = td.classify_shark_wallets(sample_shark_wallets_shark_coins_df, sample_shark_wallets_modeling_config)
    classified_sharks = shark_wallets_df[shark_wallets_df['is_shark']]['wallet_address'].tolist()
    assert classified_sharks == expected_sharks, f"Expected {expected_sharks}, got {classified_sharks}"



# ---------------------------------------- #
# classify_shark_coins() unit tests
# ---------------------------------------- #

@pytest.fixture
def sample_shark_coins_profits_df():
    """
    Sample DataFrame for testing classify_shark_coins function
    """
    data = {
        'coin_id': ['coin_1', 'coin_1', 'coin_2', 'coin_2', 'coin_3'],
        'wallet_address': ['wallet_1', 'wallet_2', 'wallet_1', 'wallet_3', 'wallet_2'],
        'date': ['2024-02-15', '2024-02-20', '2024-02-18', '2024-02-25', '2024-02-22'],
        'usd_inflows_cumulative': [5000, 15000, 8000, 20000, 5000],
        'profits_cumulative': [3000, 8000, 6000, 9000, 4000],
    }
    df = pd.DataFrame(data)
    df['total_return'] = df['profits_cumulative'] / df['usd_inflows_cumulative']
    return df

@pytest.fixture
def sample_shark_coins_modeling_config():
    """
    Sample configuration for testing classify_shark_coins function    
    """
    return {
        'modeling_period_start': '2024-03-01',
        'shark_coin_minimum_inflows': 10000,
        'shark_coin_profits_threshold': 5000,
        'shark_coin_return_threshold': 0.5
    }

@pytest.mark.unit
def test_shark_coins_eligibility_filtering(sample_shark_coins_profits_df, sample_shark_coins_modeling_config):
    """
    Test 1: Ensure wallets are filtered correctly based on shark eligibility criteria (inflows).
    """
    shark_coins_df = td.classify_shark_coins(sample_shark_coins_profits_df, sample_shark_coins_modeling_config)
    eligible_wallets = shark_coins_df['wallet_address'].unique()
    assert 'wallet_1' not in eligible_wallets, "Wallet_1 should be excluded due to insufficient inflows."
    assert 'wallet_2' in eligible_wallets, "Wallet_2 should be included."
    assert 'wallet_3' in eligible_wallets, "Wallet_3 should be included."

@pytest.mark.unit
def test_shark_coins_profits_classification(sample_shark_coins_profits_df, sample_shark_coins_modeling_config):
    """
    Test 2: Verify that wallets are classified as profits sharks correctly.
    """
    shark_coins_df = td.classify_shark_coins(sample_shark_coins_profits_df, sample_shark_coins_modeling_config)
    is_profits_shark = shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_2']['is_profits_shark'].values[0]
    assert is_profits_shark, "Wallet_2 should be classified as a profits shark."

@pytest.mark.unit
def test_shark_coins_returns_classification(sample_shark_coins_profits_df, sample_shark_coins_modeling_config):
    """
    Test 3: Verify that wallets are classified as returns sharks correctly.
    """
    shark_coins_df = td.classify_shark_coins(sample_shark_coins_profits_df, sample_shark_coins_modeling_config)
    is_returns_shark_w2 = shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_2']['is_returns_shark'].values[0]
    is_returns_shark_w3 = shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_3']['is_returns_shark'].values[0]
    assert is_returns_shark_w2, "Wallet_2 should be classified as a returns shark."
    assert not is_returns_shark_w3, "Wallet_3 should not be classified as a returns shark."

@pytest.mark.unit
def test_shark_coins_combined_shark_classification(sample_shark_coins_profits_df, sample_shark_coins_modeling_config):
    """
    Test 4: Ensure wallets are classified as sharks if they meet either profits or returns criteria.
    """
    shark_coins_df = td.classify_shark_coins(sample_shark_coins_profits_df, sample_shark_coins_modeling_config)
    is_shark = shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_2']['is_shark'].values[0]
    assert is_shark, "Wallet_2 should be classified as a shark."

@pytest.mark.unit
def test_shark_coins_modeling_period_filtering(sample_shark_coins_profits_df, sample_shark_coins_modeling_config):
    """
    Test 5: Verify that aggregates in shark_coins_df exclude data from the modeling period.
    """
    # Run the classify_shark_coins function
    shark_coins_df = td.classify_shark_coins(sample_shark_coins_profits_df, sample_shark_coins_modeling_config)

    # Manually calculate expected values for wallet_2 and wallet_3 (both should exclude modeling period data)
    assert shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_2']['profits_cumulative'].values[0] == 8000, "Profits for wallet_2 should be 8000"
    assert shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_3']['profits_cumulative'].values[0] == 9000, "Profits for wallet_3 should be 9000"
    assert shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_2']['total_return'].values[0] == 8000 / 15000, "Return for wallet_2 should be 0.6"
    assert shark_coins_df[shark_coins_df['wallet_address'] == 'wallet_3']['total_return'].values[0] == 0.45, "Return for wallet_3 should be 0.45"





# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #


# ---------------------------------- #
# set up config and module-level variables
# ---------------------------------- #

def load_test_config():
    """
    loads tests/test_config.yaml to mimic production config setup
    """
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_test_config()

# Module-level variables
TRAINING_PERIOD_START = config['modeling']['training_period_start']
TRAINING_PERIOD_END = config['modeling']['training_period_end']
MODELING_PERIOD_START = config['modeling']['modeling_period_start']
MODELING_PERIOD_END = config['modeling']['modeling_period_end']


# -------------------------- #
# retrieve_transfers_data() production data quality checks
# -------------------------- #

@pytest.fixture(scope='session')
def transfers_df():
    """
    retrieves transfers_df for data quality checks
    """
    logger.info("Beginning integration testing...")
    logger.info("Generating transfers_df fixture from production data...")
    return td.retrieve_transfers_data(TRAINING_PERIOD_START, MODELING_PERIOD_START, MODELING_PERIOD_END)

@pytest.mark.integration
def test_transfers_data_quality(transfers_df):
    """
    Retrieves transfers_df and performs comprehensive data quality checks.
    """
    logger.info("Testing transfers_df from retrieve_transfers_data()...")
    transfers_df = transfers_df.copy(deep=False)  # Create a copy to avoid affecting subsequent tests

    # Test 1: No duplicate records
    # ----------------------------
    deduped_df = transfers_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
    logger.info(f"Original transfers_df length: {len(transfers_df)}, Deduplicated: {len(deduped_df)}")
    assert len(transfers_df) == len(deduped_df), "There are duplicate rows based on coin_id, wallet_address, and date"

    # Test 2: All coin-wallet pairs have a record at the end of the training period
    # ----------------------------------------------------------------------------
    transfers_df_filtered = transfers_df[transfers_df['date'] < MODELING_PERIOD_START]
    pairs_in_training_period = transfers_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
    period_end_df = transfers_df[transfers_df['date'] == TRAINING_PERIOD_END]

    logger.info(f"Found {len(pairs_in_training_period)} total pairs in training period with {len(period_end_df)} having data at period end.")
    assert len(pairs_in_training_period) == len(period_end_df), "Not all training data coin-wallet pairs have a record at the end of the training period"

    # Test 3: No negative balances
    # ----------------------------
    # the threshold is set to -0.1 to account for rounding errors from the dune ingestion pipeline
    negative_balances = transfers_df[transfers_df['balance'] < -0.1]
    logger.info(f"Found {len(negative_balances)} records with negative balances.")
    assert len(negative_balances) == 0, "There are negative balances in the dataset"

    # Test 4: Date range check
    # ------------------------
    min_date = transfers_df['date'].min()
    max_date = transfers_df['date'].max()
    expected_max_date = pd.to_datetime(MODELING_PERIOD_END)
    logger.info(f"Date range: {min_date} to {max_date}")
    assert max_date == expected_max_date, f"The last date in the dataset should be {expected_max_date}"

    # Test 5: No missing values
    # -------------------------
    missing_values = transfers_df.isna().sum()
    assert missing_values.sum() == 0, "There are missing values in the dataset"

    # Test 6: Balance consistency
    # ---------------------------
    transfers_df['balance_change'] = transfers_df.groupby(['coin_id', 'wallet_address'],observed=True)['balance'].diff()
    transfers_df['expected_change'] = transfers_df['net_transfers']

    # Calculate the difference between balance_change and expected_change
    transfers_df['diff'] = transfers_df['balance_change'] - transfers_df['expected_change']

    # Define a threshold for acceptable discrepancies (e.g., 1e-8)
    # currently set to 0.1 for coins with values e+13 and e+14 that are showing rounding issues
    threshold = 0.1

    # Find inconsistent balances, ignoring the first record for each coin-wallet pair
    # and allowing for small discrepancies
    inconsistent_balances = transfers_df[
        (~transfers_df['balance_change'].isna()) &  # Ignore first records
        (transfers_df['diff'].abs() > threshold)    # Allow small discrepancies
    ]

    if len(inconsistent_balances) > 0:
        logger.warning(f"Found {len(inconsistent_balances)} records with potentially inconsistent balance changes.")
        logger.warning("Sample of inconsistent balances:")
        logger.warning(inconsistent_balances.head().to_string())
    else:
        logger.info("No significant balance inconsistencies found.")

    # Instead of a hard assertion, we'll log a warning if inconsistencies are found
    # This allows the test to pass while still alerting us to potential issues
    assert len(inconsistent_balances) == 0, f"Found {len(inconsistent_balances)} records with potentially inconsistent balance changes. Check logs for details."

    # Test 8: Ensure all applicable wallets have records as of the training_period_end
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs as of the training_period_end
    training_transfers_df = transfers_df[transfers_df['date'] <= TRAINING_PERIOD_END]
    training_wallets_df = training_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the training_transfers_end date
    training_end_df = transfers_df[transfers_df['date'] == TRAINING_PERIOD_END]
    training_end_df = training_end_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(training_wallets_df) == len(training_end_df), "Some wallets are missing a record as of the training_period_end"

    # Test 9: Ensure all wallets have records as of the modeling_period_start
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs
    modeling_transfers_df = transfers_df[transfers_df['date'] <= MODELING_PERIOD_START]
    modeling_wallets_df = modeling_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the modeling_period_start
    modeling_start_df = transfers_df[transfers_df['date'] == MODELING_PERIOD_START]
    modeling_start_df = modeling_start_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(modeling_wallets_df) == len(modeling_start_df), "Some wallets are missing a record as of the modeling_period_start"

    # Test 9: Ensure all wallets have records as of the modeling_period_end
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs
    modeling_transfers_df = transfers_df[transfers_df['date'] <= MODELING_PERIOD_END]
    modeling_wallets_df = modeling_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the modeling_period_end
    modeling_end_df = transfers_df[transfers_df['date'] == MODELING_PERIOD_END]
    modeling_end_df = modeling_end_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(modeling_wallets_df) == len(modeling_end_df), "Some wallets are missing a record as of the modeling_period_end"

    # Test 10: Confirm no records exist prior to the training period start
    # ------------------------------------------------------------------------------------------
    assert len(transfers_df[transfers_df['date']<TRAINING_PERIOD_START]) == 0, "Records prior to training_period_start exist"

    logger.info("All transfers_df data quality checks passed successfully.")



# ---------------------------------------- #
# calculate_wallet_profitability() production data quality tests
# ---------------------------------------- #
# tests the data quality of the production data as calculated from the transfers_df() fixture

@pytest.fixture(scope='session')
def profits_df(transfers_df):
    """
    builds profits_df from production data for data quality checks
    """
    logger.info("Generating profits_df from production data...")

    # retrieve prices data
    prices_df = td.retrieve_prices_data()

    # fill gaps in prices data
    prices_df,_ = td.fill_prices_gaps(prices_df,max_gap_days=2)

    profits_df = td.prepare_profits_data(transfers_df, prices_df)
    profits_df = td.calculate_wallet_profitability(profits_df)

    return profits_df


@pytest.mark.integration
def test_profits_df_completeness(profits_df):
    """
    Checks if there are any NaN values in profits_df. 
    """
    missing_values = profits_df.isna().sum()
    assert missing_values.sum() == 0, "There are missing values in the dataset"


@pytest.mark.integration
def test_modeling_period_end_wallet_completeness(profits_df):
    """
    Checks if all of the coin-wallet pairs at the end of the training period
    have data at the end of the modeling period to esnure they can be analyzed
    for profitability. 
    """
    training_period_end = pd.to_datetime(MODELING_PERIOD_START) - pd.Timedelta(1, 'day')
    modeling_period_end = MODELING_PERIOD_END

    # Get all coin-wallet pairs at the end of the training period
    training_profits_df = profits_df[profits_df['date'] == training_period_end]
    training_profits_df = training_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

    # Get all coin-wallet pairs at the end of the modeling period
    modeling_profits_df = profits_df[profits_df['date']==modeling_period_end]
    modeling_profits_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

    # Check if there are any pairs at the end of the training period without records at the end of the modeling period
    missing_pairs = training_profits_df.merge(modeling_profits_df, on=['coin_id', 'wallet_address'], how='left', indicator=True)
    missing_pairs = missing_pairs[missing_pairs['_merge'] == 'left_only']

    # Assert that no pairs are missing
    assert missing_pairs.empty, "Some coin-wallet pairs in training_profits_df are missing from modeling_profits_df"



# ---------------------------------------- #
# classify_shark_coins() tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def shark_coins_df(profits_df):
    """
    builds shark_coins_df from production data for data quality checks
    """
    shark_coins_df = td.classify_shark_coins(profits_df, config['modeling'])
    return shark_coins_df


@pytest.mark.integration
def test_no_duplicate_coin_wallet_pairs(shark_coins_df):
    """
    Test to assert there are no duplicate coin-wallet pairs in the shark_coins_df
    returned by classify_shark_coins().
    """
    # Group by coin_id and wallet_address and check for duplicates
    duplicates = shark_coins_df.duplicated(subset=['coin_id', 'wallet_address'], keep=False)

    # Assert that there are no duplicates in sharks_df
    assert not duplicates.any(), "Duplicate coin-wallet pairs found in sharks_df"


# ---------------------------------------- #
# classify_shark_wallets() tests
# ---------------------------------------- #

@pytest.mark.integration
def test_no_duplicate_wallets(shark_coins_df):
    """
    Test to assert there are no duplicate wallet pairs in the shark_wallets_df
    returned by classify_shark_wallets().
    """
    shark_wallets_df = td.classify_shark_wallets(shark_coins_df,config['modeling'])
    # Group by coin_id and wallet_address and check for duplicates
    duplicates = shark_wallets_df.duplicated(subset=['wallet_address'], keep=False)

    # Assert that there are no duplicates in sharks_df
    assert not duplicates.any(), "Duplicate coin-wallet pairs found in sharks_df"
