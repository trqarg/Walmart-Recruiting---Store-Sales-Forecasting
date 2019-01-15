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

    stores = df_train['Store'].unique()
    depts = df_train['Dept'].unique()

    for store in stores:

        for dep in depts:

            print('Running Store:' + str(store) + 'Dept: ' + str(dep))

            train = df_train.loc[(df_train['Store'] == store) & (df_train['Dept'] == dep)]
            test = df_test.loc[(df_test['Store'] == store) & (df_test['Dept'] == dep)]

            print(train)

            cf = getattr(Forecaster, 'LSTMForecaster')(df_train=train, df_test=test)

            print('=== BUILDING MODEL ===')

            model = cf.build_model()

            print('=== TRAINING MODEL ===')

            trained_model = cf.train_model(model)

            print('=== RUNNING FORECAST ===')

            prediction = cf.run_forecast(trained_model)

            # df_test_fu['Dept'] = dep
            # df_test_fu['prediction'] = prediction
            # df_test_fu['Date'] = dates
            # df_test_fu['lag'] = np.arange(len(df_test_fu)) + 1
            #
            # df_test_fu['prediction'][df_test_fu['prediction'] < 0] = 0

            print('=== SAVING FORECAST ===')

    print('=== PROCESS COMPLETED ===')