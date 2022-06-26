class StatSummary:
    def __init__(self):
        self.predictive_vars = []
        self.starting_balance = -1

        self.final_balance = -1
        self.risk_free_final_balance = -1

        self.returns = -1
        self.risk_free_returns = -1
        self.std = -1
        self.sharpe_ratio = -1

        self.hit_count = -1
        self.num_wins = -1
        self.num_losses = -1
        self.avg_win_return = -1
        self.avg_loss_return = -1
        

    def calc_sharpe_ratio(self):
        self.sharpe_ratio = (self.returns - self.risk_free_returns) / self.std