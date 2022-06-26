
import matplotlib.pyplot as plt

from constants.constants import *
from execute.execute_single import ExecuteSingle
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
    'variable_attributes': [EMA_5, EMA_20, TIME_NORM, CLOSE, RSI],
    'vars_per_combo': 5,
    'num_best_models': 10,
    'kfolds': [['tr', 'tr', 'v', 'te'],
               ['te', 'tr', 'tr', 'v'],
               ['v', 'te', 'tr', 'tr'],
               ['tr', 'v', 'te', 'tr']]
}


display_config = {
    'train_stats': False,
    'result_stats': True,
    'best_auc_roc_charts': True
}

# TODO: IN THE FUTURE experiment with shuffling after kfolds have been established
nn_config = {
    'patience_limit': 40,
    'profit_pct': 2,
    'stop_loss_pct': 1,
    'trade_timeout': 60, # for 15min interval, this is 1/4 hour * 60 = 15 hours before trade is considered a loss from stagnation (money that isn't making profits is money wasted)
    'transforms': [Normalizer.fit_transform],
    'batch_size': BATCH_SIZE,
    'epochs': 80,
    'learning_rate': 0.0001
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
    print('current combo: ', combo)

    if not cg.combo_is_valid(combo):
        continue

    # compute all cols for the neural net, then train the model
    df_combo = TechnicalIndicator.compute_tis(df_origin, combo)
    input_cols = combo

    es = ExecuteSingle(df_combo, input_cols, nn_config, display_config, general_config['kfolds'])
    model_result = es.execute()

    if display_config['result_stats']:
        model_result.describe()

    if model_result.get_test_auc_roc_avg() > best_model_result.get_test_auc_roc_avg():
        best_model_result = model_result


# set up test dataset for running the feed-forward stock alg (buy and sell when the model says to)
kfold = general_config['kfolds'][0]
_, _, test_data = create_kfold_data(df_origin, kfold)
input_cols = best_model_result.get_input_cols()
df_best_combo = TechnicalIndicator.compute_tis(test_data, input_cols)

# makes the assumption that the best threshold is the first kfold's best threshold (definitely wrong but serves its purpose for now)
nn_base = NeuralNetworkStrategyBase(best_model_result.get_model(), best_model_result.get_test_best_thresholds()[0], input_cols=input_cols)

balance, balance_hist, position_hist = nn_base.event_based_eval(df_best_combo)
CommandLineStats.event_stat_summary(position_hist, balance_hist)
