'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long

import pandas as pd
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()

