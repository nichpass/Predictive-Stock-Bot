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

from sklearn.metrics import confusion_matrix

sys.path.insert(1, os.path.join(sys.path[0], '..'))


import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from constants.constants import *
from ml.data_organization.dataloader import Dataset_VariablePercentOddsRatio
from ml.data_organization.dataset_util import create_kfold_data, create_kfold_datasets_forest
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


class ExecuteForest:
    def __init__(self, df, input_cols, forest_config, display_config, kfolds):
        self.df = df
        self.input_cols = input_cols
        self.forest_config = forest_config
        self.display_config = display_config

        self.kfolds = kfolds
        self.model_result = ModelResult(input_cols)


    def execute(self):
        for kfold in self.kfolds:
            print('-- executing fold: ', kfold)

            # generate data / datasets / dataloaders
            train_data, valid_data, test_data = create_kfold_data(self.df, kfold)
            train_dataset, valid_dataset, test_dataset = create_kfold_datasets_forest(self.input_cols, train_data, valid_data, test_data)

            train_x, train_y = ExecuteForest.format_data_loader(train_dataset)
            valid_x, valid_y = ExecuteForest.format_data_loader(valid_dataset)
            test_x, test_y = ExecuteForest.format_data_loader(test_dataset)

            # training / testing
            model = RandomForestClassifier(
                                            n_estimators=self.forest_config['n_estimators'],
                                            max_features=self.forest_config['max_features'],
                                            max_depth=self.forest_config['max_depth'],
                                            min_samples_split=self.forest_config['min_samples_split'],
                                            min_samples_leaf=self.forest_config['min_samples_leaf'])
            model.fit(train_x, train_y)

            train_pred = model.predict(train_x)
            valid_pred = model.predict(valid_x)
            test_pred = model.predict(test_x)

            train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_y, train_pred).ravel()
            valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(valid_y, valid_pred).ravel()
            test_tn, test_fp, test_fn, test_tp = confusion_matrix(test_y, test_pred).ravel()


            # precision = percent of positive predictions that you got right --> TP / (TP + FP)
            # recall = percent of all true positives that you predicted right --> TP / (TP + FN)
            # should first sort by precision and then by recall
            train_precision, train_recall = ExecuteForest.compute_precision_recall(train_fp, train_fn, train_tp)
            valid_precision, valid_recall = ExecuteForest.compute_precision_recall(valid_fp, valid_fn, valid_tp)
            test_precision, test_recall = ExecuteForest.compute_precision_recall(test_fp, test_fn, test_tp)

            print(f'---- train: [precision = {train_precision}, recall = {train_recall}]')
            print(f'---- valid: [precision = {valid_precision}, recall = {valid_recall}]')
            print(f'---- test: [precision = {test_precision}, recall = {test_recall}]')


    def format_data_loader(data_loader):
        return data_loader.get_inputs(), data_loader.get_labels()


    def compute_precision_recall(fp, fn, tp):
        precision = round(tp / (tp + fp), 3) if (tp + fp) else -999
        recall = round(tp / (tp + fn), 3) if (tp + fp) else -999
        return precision, recall