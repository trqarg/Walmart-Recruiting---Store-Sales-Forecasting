#!/usr/bin/python
# -*- coding: utf-8 -*-
from DataManager import DataManager
import Forecaster
import numpy as np
import pandas as pd



if __name__ == '__main__':

    print('=== STARTING PROCESS ===')

    print('=== MODEL: LSTM ===')


    dm = DataManager()

    print('=== LOADING INPUT ===')

    df_train, df_test = dm.load_input()

    depts = df_test['Dept'].unique()

    for dep in depts:

        try:

            df_train_fu = pd.DataFrame(df_train.loc[df_train['Dept'] == dep])
            df_test_fu = pd.DataFrame(df_test.loc[df_test['Dept'] == dep])

            dates = df_test_fu['Date']

            df_test_fu.drop(['Weekly_Sales'], 1, inplace=True)
            df_test_fu.drop(['Dept'], 1, inplace=True)
            df_train_fu.drop(['Dept'], 1, inplace=True)

            cf = getattr(Forecaster, 'LSTMForecaster')(df_train=df_train_fu, df_test=df_test_fu)

            print('=== BUILDING MODEL ===')

            model = cf.build_model()

            print('=== TRAINING MODEL ===')

            trained_model = cf.train_model(model)

            print('=== RUNNING FORECAST ===')

            prediction = cf.run_forecast(trained_model)

            df_test_fu['Dept'] = dep
            df_test_fu['prediction'] = prediction
            df_test_fu['Date'] = dates
            df_test_fu['lag'] = np.arange(len(df_test_fu)) + 1

            df_test_fu['prediction'][df_test_fu['prediction'] < 0] = 0

            print('=== SAVING FORECAST ===')

            dm.save_forecast(df_test_fu, dm.output_table)

        except Exception:
            print('Not enough info')
            pass

    print('=== PROCESS COMPLETED ===')