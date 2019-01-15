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

    def __init__(self,):

        self.basePath = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.inputPath = os.path.join(self.basePath, 'Data')


    def load_input(self,):
        ''' load input'''
        df_train = pd.read_csv(os.path.join(self.inputPath, 'train.csv'))

        df_test = pd.read_csv(os.path.join(self.inputPath, 'test.csv'))

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