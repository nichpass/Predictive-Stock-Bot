'''
This class is used for plotting basic stock sequences
'''

import matplotlib.pyplot as plt

# color_map = {'close': 'black', TI_BBAND_UP: 'blue', TI_BBAND_LOW: 'purple',}

class BasePlot:
    
    def lineplot(df, cols=['close'], colors=['black'], alphas=[1.]):
        _, ax = plt.subplots(figsize=(12,7))
        for i in range(len(cols)):
            df[cols[i]].plot(ax=ax, color=colors[i], alpha=alphas[i], grid=True)


    def fill_between(df, line1, line2, color, alpha):
        plt.fill_between(df.index.values, df[line1].values, df[line2].values,
                         where=(df[line1] >= df[line2]), color=color, alpha=alpha)
    
    def buy_sell_scatter(df, buy_col, sell_col, colors=['green', 'red'], symbols=['^','v']):
        plt.scatter(df.index, df[buy_col], marker=symbols[0], color=colors[0], s=200, zorder=5)
        plt.scatter(df.index, df[sell_col], marker=symbols[1], color=colors[1], s=200, zorder=5)

    def stop_loss_scatter(df, stop_loss_col):
        plt.scatter(df.index, df[stop_loss_col], marker = '_', color='purple', s=200, zorder=5)

    def show():
        plt.show()
