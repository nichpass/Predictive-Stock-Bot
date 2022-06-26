
import matplotlib.pyplot as plt
import pandas as pd

from constants.constants import *
from execute.execute_forest import ExecuteForest
from ml.data_organization.dataset_util import create_kfold_data, create_kfold_datasets, create_kfold_dataloaders
from ml.training.normalizer import Normalizer
from stats.basic_stats import CommandLineStats
from stats.model_result import ModelResult
from strategy.nn_base import NeuralNetworkStrategyBase
from util.combo import ComboGenerator
from util.stock_base import StockData
from util.technical_indicators import TechnicalIndicator

# NOTE: THIS FILE IS A WORK IN PROGRESS AND CURRENTLY DOES NOT WORK

api_config = {
    'key': ALPHA_VANTAGE_KEY, # Get api key here if lose: https://www.alphavantage.co/support/#api-key
    'symbol': 'CFLT',
    'interval': MINUTE_15,
    'time_period': MONTH_5,
    'outputsize': 'full',
    'outputformat': 'csv',
    'key_adjusted_close': '5. adjusted close',
}

general_config = {
    'bband_ma_window': 20,
    'init_balance': 10000,
    'persistent_attributes': [],
    'variable_attributes': [EMA_5, EMA_20, UPPER_BBAND_STD2, LOWER_BBAND_STD2, TIME_NORM, CLOSE, RSI],
    'vars_per_combo': 5,
    'num_best_models': 10,
    'kfolds': [['tr', 'tr', 'v', 'te']]
}

# Get stock data from api or saved csv file
sd = StockData(api_config)
df_origin = sd.get_data()

window = 3
input_cols = [LFILTER + '_' + str(window)]

df_pa = TechnicalIndicator.compute_tis(df_origin, input_cols)

# df_pa = TechnicalIndicator.loc_max_min(df_pa, LFILTER, window)

# https://blog.quantinsti.com/price-action-trading/ for the savgol filter

plt.style.use("dark_background")
plt.plot(df_pa.index, df_pa[CLOSE], color='blue', zorder=4)
plt.plot(df_pa.index, df_pa[LFILTER], color='orange', zorder=5)
# plt.scatter(df_pa.index, df_pa['lmax'], color='green', zorder=6)
# plt.scatter(df_pa.index, df_pa['lmin'], color='red', zorder=6)

plt.show()

'''
so issues that we are seeing:
- need to smooth the data, but this requires a delay / horizontal shift --> unclear what to do at the frontier
- NEED to rewrite the code so that these smoothings are calculated as it walks forward, but can deal with that later
- for now need to see what it can identify --> just looking for dtops and dbots

'''