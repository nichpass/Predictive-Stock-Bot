class Position:

    def __init__(self, id, buy_price, take_profit, stop_loss, tick_num, net_prediction, threshold):
        self.id = id
        self.buy_price = buy_price
        self.sell_price = None

        self.take_profit = take_profit
        self.stop_loss = stop_loss

        self.net_profit = None
        self.tick_enter = tick_num
        self.tick_exit = None

        # might be useful information later
        self.net_pred = net_prediction
        self.threshold = threshold

    
    def close(self, sell_price, tick_num):
        self.sell_price = sell_price
        self.tick_exit = tick_num
        self.net_profit = self.sell_price - self.buy_price
        self.net_profit_pct = (self.sell_price - self.buy_price) / self.buy_price


    def get_id(self):
        return self.id


    def get_buy_price(self):
        return self.buy_price

        
    def get_take_profit(self):
        return self.take_profit


    def get_stop_loss(self):
        return self.stop_loss


    def get_net_profit(self):
        return self.net_profit


    def get_net_profit_pct(self):
        return self.net_profit_pct