"""
Orchestrates groups of functions to generate wallet model pipeline
"""
from typing import Dict,Tuple
import logging
import pandas as pd

# Local module imports
import wallet_insights.model_evaluation as wime
import wallet_insights.wallet_model_reporting as wmr


# Set up logger at the module level
logger = logging.getLogger(__name__)

