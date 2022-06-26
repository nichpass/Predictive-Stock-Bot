'''
output of any strategy should be a buy column and a sell column
'''
import numpy as np
import pandas as pd

from .action_history import ActionHistory
from constants.constants import *


class BBandsStrategy_Basic:
    
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
        issue with upper bound: need to be shorting jesus christ
        
        - so buying and selling is reversed, since we sell and then buy back later now
        - need to separate these actions out, but in same loop since only one position active at once
        '''
        
        stop_loss = -1
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
                can_short_sell_upper = cur >= upper_std2 and ema_20 < ema_50
                can_buy_lower = cur < lower_std2 and ema_20 > ema_50
                
                if can_short_sell_upper or can_buy_lower:  
                            
                    if can_short_sell_upper:
                        stop_loss = BBandsStrategy_Basic.__calc_stop_loss(cur, upper_std1, upper_std2)
                        action_hist.update_last(short_sell=cur, state_hist=SHORT_SELL_STOCK, stop_loss=stop_loss)
                    else:
                        stop_loss = BBandsStrategy_Basic.__calc_stop_loss(cur, lower_std1, lower_std2, is_upper=False)
                        action_hist.update_last(buy_price=cur, state_hist=BUY_STOCK, stop_loss=stop_loss)

                    hold_type = HOLDING_UPPER_BB if can_short_sell_upper else HOLDING_LOWER_BB
                    state = HOLDING_STOCK
                else:                                       # do nothing, no position
                    action_hist.update_last(state_hist=DO_NOTHING)

            elif state == HOLDING_STOCK:
                
                can_short_buyback_upper = (cur >= stop_loss or cur <= upper_std1) and hold_type == HOLDING_UPPER_BB
                can_sell_lower = (cur <= stop_loss or cur >= lower_std1) and hold_type == HOLDING_LOWER_BB
                
                if can_short_buyback_upper or can_sell_lower:
                
                    if can_short_buyback_upper:
                        action_hist.update_last(short_buy=cur, state_hist=SHORT_BUY_STOCK)
                    elif can_sell_lower:
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
    
    
    def __calc_stop_loss(cur, std1, std2, is_upper=True): # not most efficient equation but is intuitive which I like better
        if is_upper:
            return cur + (std2 - std1) / 2
        else:
            return cur - (std1 - std2) / 2
