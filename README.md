# **Predictive Stock Bot**

This application executes a pipeline to create and train a model that can predict when to buy and sell stocks. It will likely never make you any money, BUT it is still a pretty cool project!

## The idea: Probability = Profits

How can we tell if an algorithm will make us any money? In my opinion, there are 3 important data points:
- **Hit Rate**: how often your algorithm sees the pattern its's looking for
- **Success Rate**: how often the algorithm makes winning trades vs losing trades using this pattern
- **Risk:Reward Ratio**: how much you win on good trades, vs how much you lose on bad trades

**Here's an example:** Suppose you have a bot that finds the pattern it's looking for once a day. It buys the stock at the current price, and then sets exit conditions as well. If the stock price increases 2%, or decreases 1%, it will sell. This means our risk-to-reward ratio is 1:2 since on bad trades we lose 1%, and on good trades we win 2%.

**How often do we have to be right to make a profit?** Well, assuming that the broker we use is super nice and there are no broker fees, as long as we win more than 33% of the trades, we will make a profit! This is because 1 win equals 2 losses, so winning 1/3 and losing 2/3 trades cancels each other out.

**Now, let's look at some actual profit numbers**: Suppose the bot is right 50% of the time. That means that on average, it will win one in two trades. We said above that our bot finds one pattern to trade on each day. That means, every two days we will win 2% and lose 1%, netting us 1% profits. Suppose this bot ran for a year, then we would make: 1.01 ^ (365/2) * 100% = 615 % returns! In other words, **if we started with $10,000, we would end with $61,500!**

This of course is just wishful thinking. Most live stock data APIs are slow, algorithm success rates change over time, and hit rate can provide its own challenges as well. Additionally, there are fees for trades, and the buy price you execute at will be worse than the actual price of the stock, which is a result of a concept called the **bid-ask spread**. (That's a whole other story though).

So in essence, this project creates a trading bot that has a set risk-to-reward ratio, and it searches for patterns based on the technical indicators it receives (volume, RSI, bollinger bands, moving averages, etc.). It demonstrates the process of creating a trading bot, but it's a bot that will probably make you go bankrupt :)


## Model Design
There are two main scrips in this repo. One for using a neural net as the model type, and one for using a random forest. They are named `main_multi.py` and `main_forest.py` respectively. Each main file has a variety of config values that can be changed, but the actual model inputs are those list in the next section. Each model outputs a probability value between 0 and 1, representing the probability that buying in that point in time will return a profit. Selling occurs automatically, with the stop-loss value set at 0.5% below the buy price, and the take-profit price set at 1% above the buy price. (1:2 risk:reward ratio)

## Inputs 

1. stock ticker
2. timeline on which to backtest
3. technical indicators


## Program Flow
There are a variety of steps that this application executes, which occur in the following order:

1. **Get the test data**: pull historical stock data using the Alpha Vantage API
2. **Format the data**: Format api response into dataframe, where columns are `OPEN|CLOSE|HIGH|LOW`, representing the stock's starting, ending, highest, and lowest price in the past 15 minutes (by default, can change the interval size)
3. **Add TI columns**: Compute technical indicators based on the stock price data (exponential moving average, Bollinger Bands, etc.) and insert as columns in dataframe
4. **Build the training set**: For each row, see if the price in the near future would decrease below the stop loss or increase above take profit price. Depending on the answer, the label for this row of inputs would be 0 (a loss) or a 1 (a profit)
4. **Train the model**: For each possible combo of technical indicators (with at least 5 in each selection), execute a full training / testing flow with a 5-fold cross validation.
5. **Evaluate the model**: Use the newly trained model on the test data and evalulate the results

## Example Run

inputs:
- ticker: `CFLT`
- technical indicators: `[EMA_5, EMA_20, UPPER_BBAND_STD2, LOWER_BBAND_STD2, TIME_NORM, CLOSE, RSI]`
- timeline: `5 months` (15 minute intervals of data, so 5 months * 30 days * 24 hours * 60 min / 15 min = 14,400 data points)

For each possible combo of technical indicators (choose 5), training will be performed:

```
NUMBER OF COMBOS: 3
current combo:  ['ema_5', 'ema_20', 'time_norm', 'close', 'rsi']
-- executing fold:  ['tr', 'tr', 'v', 'te']
--- starting training...
--- training complete (3.91 s)
-- executing fold:  ['te', 'tr', 'tr', 'v']
--- starting training...
--- training complete (3.81 s)
-- executing fold:  ['v', 'te', 'tr', 'tr']
--- starting training...
--- training complete (3.77 s)
-- executing fold:  ['tr', 'v', 'te', 'tr']
--- starting training...
--- training complete (3.77 s)
...
...

        -- results:
        -----
        --- train auc_roc per fold: [0.602, 0.526, 0.622, 0.598]
        --- valid auc_roc per fold: [0.46, 0.506, 0.499, 0.536]
        --- test  auc_roc per fold: [0.475, 0.519, 0.474, 0.518]
        -----
        --- avg train auc_roc:      0.587
        --- avg valid auc_roc:      0.5
        --- average test auc_roc:   0.497
        -----
        --- avg train loss:         0.038
        --- avg valid loss:         0.039
        --- avg test loss:          0.039
        -----
        --- best test thresholds:   [0.234, 0.303, 0.204, 0.37]
        -----

---TRADING SUMMARY---    
# of trades:              13
# of wins:                6
# of losses:              7
OVERALL win rate:         46.154 %
---------------------
avg win % return:         0.035 %
avg loss % return:        -0.024 %
----------------------
starting balance:        $10000
final balance:           $9999.794
Overall profit returns:  $-0.206
Overall percent returns:  -0.0 %
```

As we can see, the model pretty heavily overfits in each trial run. There are many parameters to tune, both hyperparameters and strategy oriented ones such as risk-to-reward ratio, etc.

## Future Work

There are a lot of ways this project could be improved. I've listed below some of the tasks that I'll hopefully have time to implement in the future:
- save models and reload best model after training --> this way training doesn't have to occur every time the script is run
- save top 10 models
- save ranking of all models for visualization
- run the simulation on the top 10 models to see their results

