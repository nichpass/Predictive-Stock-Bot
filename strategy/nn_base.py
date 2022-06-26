'''
The 'strategy' implemented here is simply however the neural network is trained. This file relies on the neural net output to make trades
'''
import numpy as np
import pandas as pd

from .action_history import ActionHistory
from constants.constants import *
from ml.training.normalizer import Normalizer
from stats.position import Position
from torch import tensor
from util.stock_funcs import StockFunctions


class NeuralNetworkStrategyBase:

    # def __init__(self, network, threshold, input_cols=[]):
    #     self.network = network.double()
    #     self.threshold = threshold
    #     self.input_cols = input_cols


    # def implement(self, df):
    #     action_hist = ActionHistory()
    #     state = NO_POSITION
        
    #     stop_loss = -1
    #     take_profit = -1

    #     win_pct = 2
    #     win_loss_ratio = 2

        
    #     for i in range(len(df.index)):

    #         cur = df.iloc[i, df.columns.get_loc('close')]
    #         action_hist.append_nan()

    #         if state == NO_POSITION:

    #             inputs = df.iloc[i, [df.columns.get_loc(c) for c in self.input_cols]]
    #             inputs = Normalizer.fit_transform(inputs)
    #             input_tensor = tensor(inputs).double()
    #             pred = self.network(input_tensor)

    #             if pred > self.threshold:
    #                 stop_loss = StockFunctions.calc_buy_stop_loss(cur, win_pct, win_loss_ratio)
    #                 take_profit = StockFunctions.calc_buy_take_profit(cur, win_pct)
    #                 action_hist.update_last(buy_price=cur, state_hist=BUY_STOCK, stop_loss=stop_loss)
    #                 state = HOLDING_STOCK
    #             else:                                      
    #                 action_hist.update_last(state_hist=DO_NOTHING)

    #         elif state == HOLDING_STOCK:

    #             if cur >= take_profit:
    #                 action_hist.update_last(sell_price=cur, state_hist=SELL_STOCK)
    #                 state = NO_POSITION
    #             elif cur <= stop_loss:
    #                 action_hist.update_last(sell_price=cur, state_hist=SELL_STOCK)
    #                 state = NO_POSITION
    #             else:
    #                 action_hist.update_last(state_hist=DO_NOTHING)

    #     df_buy_sell = pd.DataFrame({'buy': action_hist.buy_price, 'sell': action_hist.sell_price,
    #                                 'short_sell': action_hist.short_sell_price, 'short_buy': action_hist.short_buyback_price,
    #                                 'state_hist': action_hist.state_hist, 'stop_loss_hist': action_hist.stop_loss_hist})
        

    #     df_buy_sell.index += df.index[0]
    #     df = df.drop(['buy', 'sell'], axis=1, errors='ignore')
    #     df_out = pd.concat([df, df_buy_sell], axis=1)
    #     df = df.reset_index(drop=True)

    #     return df_out


    def __init__(self, network, threshold, input_cols=[]):
        self.network = network.double()
        self.threshold = threshold
        self.input_cols = input_cols
        self.active_positions = []
        self.closed_positions = []

        
    def event_based_eval(self, df, win_pct=2, win_loss_ratio=2, max_positions=5, init_balance=10000):
        '''
        so this gives us a lot of good stuff, but it doesn't really give us buy / sell stuff. That will be harder to plot and understand
        
        '''
        balance = init_balance
        balance_hist = [balance]
        position_hist = []

        for i in range(len(df.index)):
            cur = df.iloc[i, df.columns.get_loc('close')]

            if len(self.active_positions) < max_positions:
                inputs = df.iloc[i, [df.columns.get_loc(c) for c in self.input_cols]]
                inputs = Normalizer.fit_transform(inputs)
                input_tensor = tensor(inputs).double()
                pred = self.network(input_tensor)

                if pred > self.threshold:
                    stop_loss = StockFunctions.calc_buy_stop_loss(cur, win_pct, win_loss_ratio)
                    take_profit = StockFunctions.calc_buy_take_profit(cur, win_pct)
                    position = Position(0, cur, take_profit, stop_loss, i, pred, self.threshold)
                    self.active_positions.append(position)
                    
            if len(self.active_positions) > 0:
                temp_pos = []
                for pos in self.active_positions:
                    if self.can_close_position(pos, cur, i):
                        balance += pos.get_net_profit() / 5
                        balance_hist.append(balance)
                        position_hist.append(position)
                    else:
                        temp_pos.append(pos)

                self.active_positions = temp_pos

        return balance, balance_hist, position_hist


    def can_close_position(self, pos, cur_price, i):
        if cur_price > pos.get_take_profit() or cur_price <= pos.get_stop_loss():
            pos.close(cur_price, i)
            self.closed_positions.append(pos)
            return True
        else:
            return False