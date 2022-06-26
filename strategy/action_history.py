import numpy as np

class ActionHistory:
    
    def __init__(self):
        self.buy_price = []
        self.sell_price= []
        self.short_sell_price = []
        self.short_buyback_price = []
        self.state_hist = []
        self.stop_loss_hist = []
        
        
    def append_nan(self):
        self.buy_price.append(np.nan)
        self.sell_price.append(np.nan)
        self.short_sell_price.append(np.nan)
        self.short_buyback_price.append(np.nan)
        self.state_hist.append(np.nan)
        self.stop_loss_hist.append(np.nan)
        
        
    # TODO: make a function for each, this is so inefficient
    def update_last(self, buy_price=None, sell_price=None, short_buy=None, 
                    short_sell=None, state_hist=None, stop_loss=None):
        
        self.buy_price[-1] = buy_price if buy_price is not None else self.buy_price[-1]
        self.sell_price[-1] = sell_price if sell_price is not None else self.sell_price[-1]
        self.short_sell_price[-1] = short_sell if short_sell is not None else self.short_sell_price[-1]
        self.short_buyback_price[-1] = short_buy if short_buy is not None else self.short_buyback_price[-1]
        self.state_hist[-1] = state_hist if state_hist is not None else self.state_hist[-1]
        self.stop_loss_hist[-1] = stop_loss if stop_loss is not None else self.stop_loss_hist[-1]
        
        