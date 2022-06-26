
import matplotlib.pyplot as plt

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


display_config = {
    'train_stats': False,
    'result_stats': True,
    'best_auc_roc_charts': True
}

forest_config = {
    'n_estimators': 100,
    'max_features': 5,
    'max_depth': 12,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Get stock data from api or saved csv file
sd = StockData(api_config)
df_origin = sd.get_data()

# Create combo generator --> gets the input cols for each neural net
cg = ComboGenerator(general_config['persistent_attributes'], general_config['variable_attributes'], r=general_config['vars_per_combo'])
num_combos = cg.get_num_combos()
print('NUMBER OF COMBOS:', num_combos)

# Create objects to store the results
model_result_list = []
auc_roc_values = []
roc_best_thresholds = []

best_model_result = ModelResult(None)
best_model_result.set_test_auc_roc_avg(-1)

# test out each combo to see how well it does
for i in range(num_combos):
    combo = cg.get_next_combo()
    if not cg.combo_is_valid(combo):
        continue

    print('current combo: ', combo)

    # compute all cols for the neural net, then train the model
    df_combo = TechnicalIndicator.compute_tis(df_origin, combo)
    input_cols = combo

    es = ExecuteForest(df_combo, input_cols, forest_config, display_config, general_config['kfolds'])
    es.execute()

