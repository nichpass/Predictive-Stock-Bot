class StockFunctions:

    def calc_buy_stop_loss(cur, win_pct, win_loss_ratio):
        return cur * (1 - (win_pct / win_loss_ratio) / 100)

    def calc_buy_take_profit(cur, win_pct):
        return cur * (1 + win_pct / 100)

    def calc_short_stop_loss(cur, win_pct, win_loss_ratio):
        return cur * (1 + (win_pct / win_loss_ratio) / 100)