'''
output of any strategy should be a buy column and a sell column
'''
import numpy as np
import pandas as pd

from .action_history import ActionHistory
from constants.constants import *


class BBandsStrategy_TrendRide:
    
    def implement(df, ema):
        ema_str = 'ema_' + str(ema)
        if ema_str not in df:
            print("EMA column not found, exiting")
            exit()

        istart, iend = ema, len(df.index)   
        data = df.iloc[istart:iend]
        
        action_hist = ActionHistory()
        state = NO_POSITION
        
        '''
        trend riding strategy:
        if the ema20 > ema50 and goes outside std, then we chill
        '''
        
        stop_loss = -1
        hit_count = 0
        hold_type = NO_HOLD
        
        for i in range(istart, iend):
            cur = data['close'][i]
            upper_std1 = data[UPPER_BBAND_STD1][i]
            upper_std2 = data[UPPER_BBAND_STD2][i]
            lower_std1 = data[LOWER_BBAND_STD1][i]
            lower_std2 = data[LOWER_BBAND_STD2][i]
            
            ema_50 = data[EMA_50][i]
            ema_20 = data[EMA_20][i]

            action_hist.append_nan()
  
            if state == NO_POSITION:
                ema_20_slope = BBandsStrategy_TrendRide.__calc_ema_slope(df, EMA_20, 20, i)
                ema_50_slope = BBandsStrategy_TrendRide.__calc_ema_slope(df, EMA_50, 50, i)
                
                can_buy_upper = cur >= upper_std1 and ema_20 > ema_50 and ema_20_slope > 50
                can_short_sell_lower = cur <= lower_std1 and ema_20 < ema_50 and ema_20_slope < -50
                can_act = can_buy_upper or can_short_sell_lower

                if can_act:
                    hit_count += 1
                    if hit_count == 2:
                        if can_short_sell_lower:
                            stop_loss = BBandsStrategy_TrendRide.__calc_stop_loss(cur, upper_std1, upper_std2, lower_std1, lower_std2, is_upper=False)
                            action_hist.update_last(short_sell=cur, state_hist=SHORT_SELL_STOCK, stop_loss=stop_loss)
                        elif can_buy_upper:
                            stop_loss = BBandsStrategy_TrendRide.__calc_stop_loss(cur, upper_std1, upper_std2, lower_std1, lower_std2)
                            action_hist.update_last(buy_price=cur, state_hist=BUY_STOCK, stop_loss=stop_loss)
                        hold_type = HOLDING_LOWER_BB if can_short_sell_lower else HOLDING_UPPER_BB
                        state = HOLDING_STOCK
                        hit_count = 0
                else:                                      
                    action_hist.update_last(state_hist=DO_NOTHING)

            elif state == HOLDING_STOCK:
                stop_loss = BBandsStrategy_TrendRide.__calc_stop_loss(cur, upper_std1, upper_std2, lower_std1, lower_std2)
                action_hist.update_last(stop_loss=stop_loss)

                can_short_buyback_lower = (cur >= stop_loss) and hold_type == HOLDING_LOWER_BB
                can_sell_upper = (cur <= stop_loss) and hold_type == HOLDING_UPPER_BB
                can_act = can_sell_upper or can_short_buyback_lower
                
                if can_act:
                    if can_short_buyback_lower:
                        action_hist.update_last(short_buy=cur, state_hist=SHORT_BUY_STOCK)
                    elif can_sell_upper:
                        action_hist.update_last(sell_price=cur, state_hist=SELL_STOCK)
                        
                    state = NO_POSITION
                    hold_type = NO_HOLD
                else:
                    action_hist.update_last(state_hist=DO_NOTHING)

        df_buy_sell = pd.DataFrame({'buy': action_hist.buy_price, 'sell': action_hist.sell_price,
                                    'short_sell': action_hist.short_sell_price, 'short_buy': action_hist.short_buyback_price,
                                    'state_hist': action_hist.state_hist, 'stop_loss_hist': action_hist.stop_loss_hist})
        df_buy_sell.index += istart
        df_out = pd.concat([df, df_buy_sell], axis=1)
        
        return df_out
    
    
    def __calc_stop_loss(cur, ustd1, ustd2, lstd1, lstd2, is_upper=True):
        if is_upper:
            return (ustd1 + lstd1) / 2

        else:
            return (ustd1 + lstd1) / 2
        
    def __calc_ema_slope(df, ema, window, cur_index):
        if cur_index >= window:
            return (df[ema][cur_index] - df[ema][cur_index - window]) / window
        else:
            return -1
