import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from constants.constants import *
from torch.utils.data import Dataset


class Dataset_VariablePercentOddsRatio(Dataset):
    
    def __init__(self, df, input_cols, timeout=10, profit_pct=2, stop_loss_pct=1, transforms=[], logs_enabled=False):
        df = df.reset_index(drop=True)

        r = df.index[-2] - df.index[0]
        c = len(input_cols)
        
        self.logs_enablead = logs_enabled

        self.dataset_inputs = np.empty((r, c), float)
        self.dataset_labels = np.empty((r, 1), float)
        
        self.input_cols = input_cols
        self.timeout = timeout
        self.profit_pct = profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        self.bad_data_idxs = []

        for idx in range(df.index[0], df.index[-2]):
            input = np.array(df.loc[idx, input_cols], float)
            label = np.array([self.__evaluate_trade(df, idx)], float)
            self.dataset_inputs[idx] = input
            self.dataset_labels[idx] = label
            idx += 1
                

        self.remove_inconclusive_data()

        for transform in transforms:
            self.dataset_inputs = transform(self.dataset_inputs)

        counts = np.unique(self.dataset_labels, return_counts=True)
        self.neg_pos_ratio = float(counts[1][0]) / counts[1][1] 

        if logs_enabled:
            print("pos cases: ", counts[1][1], ", neg cases: ", counts[1][0], ', inconc. cases: ', len(df.index) - counts[1][0] - counts[1][1] - 2)
    

    def get_neg_pos_ratio(self):
        return self.neg_pos_ratio


    def __getitem__(self, idx):
        return self.dataset_inputs[idx], self.dataset_labels[idx]
    
    
    def __len__(self):
        return len(self.dataset_inputs)


    def __evaluate_trade(self, df, idx):
        cur = idx
        take_profit_price = df['close'][idx] * (1 + self.profit_pct / 100)
        stop_loss_price = df['close'][idx] * (1 - self.stop_loss_pct / 100)
        
        while cur < min(idx + self.timeout, len(df.index)):
            cur_price = df['close'][cur]
            high_price, low_price = df['high'][cur], df['low'][cur]
            
            if high_price >= take_profit_price and low_price <= stop_loss_price:
                self.bad_data_idxs.append(idx)
                return 'nan'
            if cur_price >= take_profit_price or high_price >= take_profit_price:
                return 1
            elif cur_price <= stop_loss_price or low_price <= stop_loss_price:
                return 0
            
            cur += 1

        return 0
    

    def remove_inconclusive_data(self):
        self.dataset_inputs = np.delete(self.dataset_inputs, self.bad_data_idxs, axis=0)
        self.dataset_labels = np.delete(self.dataset_labels, self.bad_data_idxs)


    def get_input_shape(self):
        return self.dataset_inputs.shape


    def get_inputs(self):
        return self.dataset_inputs


    def get_labels(self):
        return self.dataset_labels