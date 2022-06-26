'''
This file contains utility functions for pulling data from alpha vantage
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pandas as pd
pd.set_option('display.max_columns', None)


import csv
import constants.constants as constants
import requests

from datetime import datetime


class StockData:

    # TODO: add a reset so '15min_2month" data can be the most recent data available
    url_daily = f''
    url_intraday = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice={}&apikey={}'


    def __init__(self, config):
        self.config = config


    def get_data(self):
        filepath = self.__create_filepath()

        if not os.path.isfile(filepath):
            df = self.__pull_data()
            df = self.__format_data(df)
            self.__save_data_to_csv(df, filepath)
            return df
        else: 
            return self.__read_from_csv(filepath)


    def __create_filepath(self):
        filename = self.config['symbol'] + '_' + self.config['interval'] + '_' + self.config['time_period'] + '_' +datetime.today().strftime('%Y-%m-%d') + ".csv"
        return os.path.join(constants.BASE_PATH, "data", '15min', filename)
        

    def __pull_data(self):
        data_format = self.config['interval']

        if data_format in ('5min', '15min', '30min'):
            return self.__pull_data_intraday()
        elif data_format == 'day':
            return self.__pull_data_day()
        else:
            print("Data interval not recognized, exiting")
            exit()


    def __pull_data_intraday(self):
        slices = self.__generate_slices()
        df = pd.DataFrame()
        for slice in slices:
            with requests.Session() as session:
                url = StockData.url_intraday.format(self.config['symbol'], self.config['interval'], slice, constants.ALPHA_VANTAGE_KEY)
                download = session.get(url)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                df_slice = pd.DataFrame(list(cr))
                df = pd.concat([df, df_slice])
        return df


    def __generate_slices(self):
        n_months = int(self.config['time_period'].split('month')[0])
        slices = []
        for i in range(1, n_months+1):
            slices.append(f'year1month{i}')
        return slices


    def __pull_data_day(self):
        pass


    def __format_data(self, df):
        if df.empty:
            print("Dataframe is null when it should be populated, exiting")
            exit()
            
        # grab desired colums and make the oldest timestamp have the first index (api sends reverse)
        df = df.iloc[::-1]
        df= df.reset_index(drop=True)
        df = df.drop(df.tail(1).index)

        # name / cast all of the columns
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df.dropna()
        df = self.__convert_dtypes(df)
        return df


    def __convert_dtypes(self, df):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        return df

    def __save_data_to_csv(self, df, filepath):
        if not df.empty:
            df.to_csv(filepath)
        else:
            print("Dataframe is empty / Null at save point, exiting")
            exit()


    def __read_from_csv(self, filepath):
        df = pd.read_csv(filepath, index_col=0)
        df = self.__convert_dtypes(df)
        return df


