'''
INPUTS:
-stock name
-stock interval
-length of history
-input params


OUTPUTS:
- stats on how well it did? --> stats object
    - sharpe ratio
    - std
    - numerator of sharpe ratio
    - returns vs risk free returns
    - something showing returns over time
    - k-fold results

'''

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))


import numpy as np

from constants.constants import *
from ml.data_organization.dataloader import Dataset_VariablePercentOddsRatio
from ml.data_organization.dataset_util import create_kfold_data, create_kfold_datasets, create_kfold_dataloaders
from ml.model_architectures.basenetwork import BaseNetwork
from ml.training.normalizer import Normalizer
from ml.training.train import BaseModelTrain
from stats.basic_stats import CommandLineStats
from stats.stat_util import get_roc_data
from strategy.nn_base import NeuralNetworkStrategyBase
from util.crypto_data_pull import CryptoDataPull
from util.stock_base import StockData
from util.technical_indicators import TechnicalIndicator
from stats.model_result import ModelResult


np.random.RandomState(1234)


class ExecuteSingle:
    def __init__(self, df, input_cols, nn_config, display_config, kfolds):
    # def __init__(self, stock, input_cols,  interval='15min', time_period='5month', init_balance=10000, stat_display=False):
        self.df = df
        self.input_cols = input_cols
        self.nn_config = nn_config
        self.display_config = display_config

        self.kfolds = kfolds


        self.model_result = ModelResult(input_cols)


    def execute(self):
        for kfold in self.kfolds:
            print('-- executing fold: ', kfold)

            # generate data / datasets / dataloaders
            train_data, valid_data, test_data = create_kfold_data(self.df, kfold)
            train_dataset, valid_dataset, test_dataset = create_kfold_datasets(self.input_cols, self.nn_config, train_data, valid_data, test_data)
            train_loader, valid_loader, test_loader = create_kfold_dataloaders(self.nn_config, train_dataset, valid_dataset, test_dataset)

            # Actual training / testing
            network = BaseNetwork(inputs=len(self.input_cols))
            bmt = BaseModelTrain(network, self.nn_config, self.input_cols, train_loader, valid_loader, test_loader, loss_ratio=train_dataset.get_neg_pos_ratio())
            bmt.main()

            # bmt.plot_loss()

            # get relevant output stats for determining overfitting (train and test should be about as successful)
            train_loss, train_auc_roc, train_best_threshold = bmt.train(eval=True)
            valid_loss, valid_auc_roc, valid_best_threshold = bmt.validate(eval=True)
            test_loss, test_auc_roc, test_best_threshold = bmt.test()

            self.model_result.append_train(train_loss, train_auc_roc, train_best_threshold)
            self.model_result.append_valid(valid_loss, valid_auc_roc, valid_best_threshold)
            self.model_result.append_test(test_loss, test_auc_roc, test_best_threshold)

            self.model_result.compute_averages()
            self.model_result.set_model(network)

        return self.model_result
