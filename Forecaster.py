#!/usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from keras import backend as be
from keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import confusion_matrix
import json
import numpy as np


class Forecaster(ABC):
    """
    Abstract Forecasting Algorithms Class
    """
    def __init__(self, df_train, df_test):
        self.df_train = df_train

        df_train["Date"] = df_train["Date"].astype('category')
        self.fecha_cat_test = dict(enumerate(df_train['Date'].cat.categories))
        df_train["Date"] = df_train["Date"].cat.codes

        self.x_values = self.df_train.iloc[:, 0: 1].values
        self.y_values = self.df_train['Weekly_Sales']

        self.df_test = df_test

        df_test["Date"] = df_test["Date"].astype('category')
        self.fecha_cat_train = dict(enumerate(df_test['Date'].cat.categories))
        df_test["Date"] = df_test["Date"].cat.codes

        self.x_values_ahead = self.df_test.iloc[:, 0: 1].values

        '''SCALING DATA'''
        self.scalerX = MinMaxScaler(feature_range=(0, 1)).fit(self.x_values)

        '''RESHAPE INTO [samples, time steps, features] '''
        self.x_values = self.x_values.reshape((self.x_values.shape[0], 1, self.x_values.shape[1]))
        self.x_values_ahead = self.x_values_ahead.reshape((self.x_values_ahead.shape[0], 1, self.x_values_ahead.shape[1]))

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, classifier, bayesian):
        pass

    @abstractmethod
    def run_forecast(self, classifier):
        pass

class LSTMForecaster(Forecaster):

    def build_model(self, hidden_layer_sizes=None, activation=None, solver=None, max_iter=None):
        ''' Builds the Long Short Term Memory Neural Net'''
        classifier = Sequential()
        classifier.add(LSTM(32,
                       activation='relu',
                       input_shape=(self.x_values.shape[1], self.x_values.shape[2]),
                       stateful=False,
                       return_sequences=True))
        classifier.add(Dropout(0.1))
        classifier.add(LSTM(64,
                       activation='relu',
                       input_shape=(self.x_values.shape[1], self.x_values.shape[2]),
                       stateful=False,
                       return_sequences=False))
        classifier.add(Dropout(0.1))
        classifier.add(Dense(1, activation='sigmoid'))
        classifier.compile(loss='mse', optimizer='adam')
        return classifier

    def train_model(self, classifier, bayesian=False):
        ''' Trains the Long Short Term Memory Neural Net'''
        classifier.fit(self.x_values, self.y_values, epochs=1000, batch_size=1, verbose=0, shuffle=False)
        return classifier

    def run_forecast(self, classifier):
        ''' Makes Long Short Term Memory Neural Net Regression prediction'''
        prediction = classifier.predict(self.x_values_ahead, 1)
        return prediction

