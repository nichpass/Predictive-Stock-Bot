import copy
import numpy as np
import pandas as pd

from constants.constants import *
from scipy.signal import lfilter, argrelextrema

class TechnicalIndicator:
    
    def compute_tis(df, input_cols, bband_ma_window=20):
        for col in input_cols:
            if TechnicalIndicator.__no_calc_needed(col):
                pass

            elif TechnicalIndicator.__is_time(col):
                df = TechnicalIndicator.add_timenorm(df)

            elif TechnicalIndicator.__is_ema(col):
                df = TechnicalIndicator.add_emas(df, [TechnicalIndicator.__parse_ema(col)])

            elif TechnicalIndicator.__is_bband(col):
                bband_num = TechnicalIndicator.__parse_bband(col)
                df = TechnicalIndicator.add_bbands(df, col, window=bband_ma_window, num_std=bband_num)

            elif TechnicalIndicator.__is_rsi(col):
                df = TechnicalIndicator.add_rsi(df)

            elif TechnicalIndicator.__is_lfilter(col):
                df = TechnicalIndicator.add_lfilter(df, TechnicalIndicator.__parse_lfilter(col))

            elif col == 'test_10':
                df = TechnicalIndicator.add_lfilter_smooth(df, TechnicalIndicator.__parse_lfilter(col))

            else:
                print("Technical indicator not recognized, exiting")
                exit()

        df = df.dropna()
        return df


    def loc_max_min(df, col, window):
        if col not in df.columns:
            print(df.columns, col)
            print("Column specified for local maxima / minima is not present in dataframe, exiting")
            exit()
        n = window//2
        df['lmax'] = df.iloc[argrelextrema(df[col].values, np.greater_equal, order=window)[0]][col]
        df['lmin'] = df.iloc[argrelextrema(df[col].values, np.less_equal, order=window)[0]][col]

        return df


    def add_bbands(df, col, ema=20, window=20, num_std=2):
        ema_str = 'ema_' + str(ema)

        # we add both upper and lower whenever one of them pops up, the just skip the other one
        if col in df.columns:
            return df

        if ema_str not in df:
            df = TechnicalIndicator.add_emas(df, [ema])
        
        std = df['close'].rolling(window).std()
        upper_bound = df[ema_str] + std * num_std
        lower_bound = df[ema_str] - std * num_std
        
        upper_col_name = TI_BBAND_UP + '_std' + str(num_std)
        lower_col_name = TI_BBAND_LOW + '_std' + str(num_std)
        df[upper_col_name], df[lower_col_name] = upper_bound, lower_bound
        return df
    

    def add_emas(df, ema_list):
        for ema in ema_list:
            df["ema_" + str(ema)] = round(df.loc[:, 'close'].ewm(span=ema, adjust=False, min_periods=ema).mean(), 2)
        return df
    

    def add_std(df, window_list):
        for window in window_list:
            df['std_' + str(window)] = df['close'].rolling(window=window).std()
        return df


    def add_timenorm(df):
        df[TIME_NORM] = np.sin(2 * np.pi * (df[TIME].dt.hour + df[TIME].dt.minute / 60) / 24)
        return df


    def add_rsi(df):
        delta = df[CLOSE].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down
        df[RSI] = 100 - (100/(1 + rs))
        return df


    def add_lfilter(df, window):
        y = df['close']
        n = window  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        df['lfilter'] = lfilter(b,a,y)
        return df


    def add_lfilter_smooth(df, window):
        y = df['close']
        n = window  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        filtered = lfilter(b,a,y)
        smoothed = pd.Series(filtered).ewm(span=20).mean()
        df['test'] = smoothed
        return df


    def __is_lfilter(ti):
        return ti.split('_')[0] == 'lfilter'


    def __parse_lfilter(ti):
        return int(ti.split('_')[-1])
        

    def get_sma(df, window):
        pass


    def __no_calc_needed( ti):
        return ti in (TIME, OPEN, HIGH, LOW, CLOSE, VOLUME)


    def __is_time(ti):
        return ti == TIME_NORM


    def __is_ema(ti):
        return ti.split('_')[0] == 'ema'


    def __parse_ema(ema):
        return int(ema.split('_')[-1])


    def __is_bband(ti):
        return 'bband' in ti.split('_')


    def __parse_bband(ti):
        return int(ti[-1])

    
    def __is_rsi(ti):
        return ti == RSI

    