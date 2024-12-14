"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0302 # over 1000 lines
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=E0401 # can't find import (due to local import)
# pyright: reportMissingModuleSource=false

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import wallet_features.transfers_features as wts

load_dotenv()
logger = dc.setup_logger()

# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------------- #
# calculate_average_holding_period() unit tests
# ------------------------------------------------- #

@pytest.mark.unit
def test_calculate_average_holding_period():
    """
    Test the wts.calculate_average_holding_period function using a sample DataFrame.

    Step-by-step calculation logic:
    1. 2024-01-01 buy 100:
       - Initial purchase, average age = 0.

    2. 2024-01-05 sell 50:
       - Days since last tx = 4
       - Previously 100 coins at age=0, now aged by 4 days → avg_age = 0+4=4
       - Sell 50 at avg_age=4, remaining 50 coins keep avg_age=4.

    3. 2024-01-10 buy 75:
       - Days since last tx = 5
       - Age existing 50 coins: avg_age=4+5=9
       - Buy 75 new at age=0
       - Weighted avg = (50*9 + 75*0)/125 = 450/125 = 3.6 days.

    4. 2024-01-15 sell 25:
       - Days since last tx = 5
       - Age existing 125 coins: avg_age=3.6+5=8.6
       - Sell 25 at avg_age=8.6, leaving 100 at avg_age=8.6.

    5. 2024-01-20 no transaction:
       - Days since last tx = 5
       - Age 100 coins: avg_age=8.6+5=13.6
       - Balance unchanged.

    6. 2024-01-29 no transaction:
       - Days since last tx = 9
       - Age 100 coins: avg_age=13.6+9=22.6
       - Balance unchanged.

    7. 2024-01-30 sell 100:
       - Days since last tx = 1
       - Age 100 coins: avg_age=22.6+1=23.6
       - Sell all, balance=0, avg_age resets effectively to 0 since no holdings remain.

    Expected average_holding_period by date:
    2024-01-01: 0
    2024-01-05: 4
    2024-01-10: 3.6
    2024-01-15: 8.6
    2024-01-20: 13.6
    2024-01-29: 22.6
    2024-01-30: 0 (no holdings)
    """

    df = pd.DataFrame({
        'coin_id': ['BTC'] * 7,
        'wallet_address': ['wallet1'] * 7,
        'date': [
            '2024-01-01', '2024-01-05', '2024-01-10',
            '2024-01-15', '2024-01-20', '2024-01-29',
            '2024-01-30'
        ],
        'net_transfers': [100, -50, 75, -25, 0, 0, -100]
    })
    # Expected results from the calculation logic above
    expected = np.array([0, 4, 3.6, 8.6, 13.6, 22.6, 0])

    result = wts.calculate_average_holding_period(df)

    # Compare expected and actual using np.allclose for floating-point tolerance
    assert np.allclose(result['average_holding_period'].values,
                       expected, rtol=1e-05, equal_nan=True)

