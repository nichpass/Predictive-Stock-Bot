'''
This file contains utility functions for pulling data from alpha vantage
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import constants.constants as constants
import pandas as pd
pd.set_option('display.max_columns', None)

from alpha_vantage.cryptocurrencies import CryptoCurrencies

class CryptoDataPull:

    # TODO: add a reset so '15min_2month" data can be the most recent data available

    def get_daily(self, config):
        self.config = config
        self.__create_filepath()

        if not os.path.isfile(self.filepath):
            self.__req_and_save_currency()

    
        df = pd.read_csv(self.filepath, index_col=0)
        df = df.reset_index(drop=True)
        return df


    def __create_filepath(self):
        # filename = self.config['symbol'] + '_' + self.config['interval'] + '_' + self.config['slice'] + ".csv"
        filename = self.config['symbol'] + '_daily_2021-2020_' + ".csv"
        self.filepath = os.path.join(constants.BASE_PATH, "data", filename)
        

    def __req_and_save_currency(self):
        cc = CryptoCurrencies(key=self.config['key'], output_format=self.config['outputformat'])
        data = cc.get_digital_currency_daily(self.config['symbol'], market='USD')
        df = pd.DataFrame(list(data[0]))
        df = self.__format_df_currency(df)
        df.to_csv(self.filepath)        

    
    def __format_df_currency(self, df):
        # grab desired colums and make the oldest timestamp have the first index (api sends reverse)
        df = df.iloc[1:, [0, 1, 2, 3, 4, 9, 10]]
        df = df.iloc[::-1]
        df = df.reset_index(drop=True)
        
        # name / cast all of the columns
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'market_cap']
        df['open'] = df['open'].astype('float32')
        df['high'] = df['high'].astype('float32')
        df['low'] = df['low'].astype('f')
        df['close'] = df['close'].astype('f')
        df['volume'] = df['volume'].astype('f')
        return df
