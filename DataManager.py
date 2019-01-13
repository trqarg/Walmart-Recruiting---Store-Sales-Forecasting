#!/usr/bin/python
# -*- coding: utf-8 -*-
from dateutil.relativedelta import relativedelta
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
import pandas as pd
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class DataManager(object):

    def __init__(self, input_table, output_table):

        self.input_table = input_table
        self.output_table = output_table
        self.basePath = os.path.abspath(os.path.join(os.path.dirname(__file__)))


    def load_input(self,):
        ''' load input from Database'''
        df_train = pd.read_csv(os.path.join(self.basePath, 'train.csv'))

        df_test = pd.read_csv(os.path.join(self.basePath, 'test.csv'))

        return df_train, df_test


    def getPeriods(self, start, forecast_horizon, period_type):
        '''Build periods to predict from starting date'''
        start = datetime.datetime.strptime(start, '%Y-%m-%d %I:%M')
        if period_type == 'days':
            end = start + relativedelta(days=+(forecast_horizon - 1) * 1)
        elif period_type == 'hours':
            end = start + relativedelta(hours=+(forecast_horizon - 1) * 24)
        elif period_type == 'minutes':
            end = start + relativedelta(minutes=+(forecast_horizon - 1) * 1440)
        cur_date = start
        periods = []
        periods.append(start)
        while cur_date < end:
            if period_type == 'days':
                cur_date += relativedelta(days=1)
            elif period_type == 'hours':
                cur_date += relativedelta(hours=1)
            elif period_type == 'minutes':
                cur_date += relativedelta(minutes=1)
            periods.append(cur_date)
        return periods


    def save_forecast(self, prediction, output_table):
        '''Save Model on Database'''
        self.dao.upload_from_dataframe(prediction, output_table, if_exists='append')


