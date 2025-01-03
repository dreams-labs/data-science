# # pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# # pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# # pylint: disable=unused-argument
# # pyright: reportMissingModuleSource=false

# import sys
# from datetime import timedelta
# from pathlib import Path
# from dataclasses import dataclass
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# import pytest
# from dreams_core import core as dc

# # pyright: reportMissingImports=false
# sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
# import wallet_features.trading_features as wtf
# from wallet_modeling.wallets_config_manager import WalletsConfig

# load_dotenv()
# logger = dc.setup_logger()

# config_path = Path(__file__).parent.parent / 'test_config' / 'test_wallets_config.yaml'
# wallets_config = WalletsConfig.load_from_yaml(config_path)


# # ===================================================== #
# #                                                       #
# #                 U N I T   T E S T S                   #
# #                                                       #
# # ===================================================== #


# # ------------------------------------------------ #
# # split_training_window_dfs() unit tests
# # ------------------------------------------------ #
