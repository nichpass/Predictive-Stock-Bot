from re import S
from typing import final
from constants.constants import *
class CommandLineStats:


    def create_and_display(df, init_balance=10000, partitions=5):
        df = df.reset_index(drop=True)
        trade_profit_history = []
        trade_profit_pct_history = []
        cur_hold = 'e'
        cur = 0
        state_hist = df['state_hist']
        while cur < len(state_hist):
                        
            if state_hist[cur] == BUY_STOCK:
                cur_hold = df['buy'][cur]

            elif state_hist[cur] == SHORT_SELL_STOCK:
                cur_hold = df['short_sell'][cur]
                
            elif state_hist[cur] == SELL_STOCK or state_hist[cur] == SHORT_BUY_STOCK:
                
                if state_hist[cur] == SELL_STOCK:
                    profit = df['sell'][cur] - cur_hold
                    print(f'buy-sell profit at {cur} is {profit}')
                elif state_hist[cur] == SHORT_BUY_STOCK:
                    profit = cur_hold - df['short_buy'][cur]
                    print(f'short profit at {cur} is {profit}')

                profit_pct = profit / cur_hold
                cur_hold = 'e' # breaks if not reset properly
                
                trade_profit_history.append(profit)
                trade_profit_pct_history.append(profit_pct)
            
            cur += 1
    
        final_balance = 10000
        for pct_change in trade_profit_pct_history:
            if pct_change != 0:
                final_balance += final_balance / partitions * pct_change   # risk 1% of 1/5 of money on each trade
            
        wins = [x for x in trade_profit_pct_history if x > 0]
        losses = [x for x in trade_profit_pct_history if x <= 0]

        buy_and_hold_profit = round(df['close'].iloc[-1] - df['close'].iloc[0], 3)
        buy_and_hold_profit_pct = round(buy_and_hold_profit / df['close'].iloc[0] * 100, 3)
        
        num_trades = len(trade_profit_history)
        num_wins = len(wins)
        num_losses = len(losses)

        if num_wins == 0 and num_losses == 0:
            win_rate = -1
        else:
            win_rate = round(num_wins / (num_wins + num_losses) * 100.0, 3)


        if num_wins == 0:
            avg_win_returns = -1
        else:
            avg_win_returns = round(sum(wins) / len(wins) * 100.0, 3)

        if num_losses == 0:
            avg_loss_returns = -1
        else:
            avg_loss_returns = round(sum(losses) / len(losses) * 100.0, 3)
        
        init_balance = round(init_balance, 3)
        final_balance = round(final_balance, 3)

        overall_profit_returns = round(final_balance - init_balance, 3)
        overall_profit_pct_returns = round(final_balance / (init_balance) - 1, 3)

        print(trade_profit_pct_history)

        print("---TRADING SUMMARY---    ")
        print(f"# of trades:              {num_trades}")
        print(f"# of wins:                {num_wins}")
        print(f"# of losses:              {num_losses}")
        print(f"OVERALL win rate:         {win_rate} %")
        print("---------------------    ")
        print(f"avg win % return:         {round(avg_win_returns, 3)} %")
        print(f"avg loss % return:        {round(avg_loss_returns, 3)} %")
        print("----------------------   ")
        print(f"starting balance:        ${init_balance}")
        print(f"final balance:           ${round(final_balance, 3)}")
        print(f"Overall profit returns:  ${round(overall_profit_returns, 3)}")
        print(f"Overall percent returns:  {round(overall_profit_pct_returns, 3)} %")
        print("----------------------   ")
        print(f"Profit if just held:     ${round(buy_and_hold_profit, 3)}")
        print(f"Pct return if just held:  {round(buy_and_hold_profit_pct, 3)} %")
                

    def event_stat_summary(closed_positions, balance_hist,):
        num_trades = len(closed_positions)
        winning_trades = [p for p in closed_positions if p.get_net_profit_pct() > 0]
        losing_trades = [p for p in closed_positions if p.get_net_profit_pct() < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        if (num_wins + num_losses) > 0:
            win_rate = num_wins / (num_wins + num_losses) * 100.0
        else:
            win_rate = -1

        avg_win_returns = 0 if num_wins == 0 else sum([p.get_net_profit_pct() for p in winning_trades]) / num_wins
        avg_loss_returns = 0 if num_losses == 0 else sum([p.get_net_profit_pct() for p in losing_trades]) / num_losses

        net_profits = balance_hist[-1] - balance_hist[0]
        net_return = (balance_hist[-1] - balance_hist[0]) / balance_hist[0]

        print("---TRADING SUMMARY---    ")
        print(f"# of trades:              {num_trades}")
        print(f"# of wins:                {num_wins}")
        print(f"# of losses:              {num_losses}")
        print(f"OVERALL win rate:         {round(win_rate, 3)} %")
        print("---------------------    ")
        print(f"avg win % return:         {round(avg_win_returns, 3)} %")
        print(f"avg loss % return:        {round(avg_loss_returns, 3)} %")
        print("----------------------   ")
        print(f"starting balance:        ${round(balance_hist[0], 3)}")
        print(f"final balance:           ${round(balance_hist[-1], 3)}")
        print(f"Overall profit returns:  ${round(net_profits, 3)}")
        print(f"Overall percent returns:  {round(net_return, 3)} %")
        # print("----------------------   ")
        # print(f"Profit if just held:     ${buy_and_hold_profit}")
        # print(f"Pct return if just held:  {buy_and_hold_profit_pct} %")

'''
so two different types of implementations:
- vectorized: single buy-sell top down (useful for visualizing)
- event based: store event objects --> can also visualize, just might be a tad tricker with the colors going on


you'll have the information you need (I think), so it should be fine

NEED TO FIND A WAY TO ITERATE OVER THE EVENTS WHILE TRACKING CURRENT BALANCE
- balance loses 1/5 of itself whenever I buy, then gains back that much * net_profit_pct whenever I sell
- so balance needs to be updated based on the indexes
- should be a good way to combine all of the buy indexes and then the sell indexes then iterate over both at the same time
    - like that alg where you pick the least of both sides


'''