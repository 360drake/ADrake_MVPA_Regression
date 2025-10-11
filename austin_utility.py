# Utility functions for AFIS regression analysis

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.model_selection import GridSearchCV, LeaveOneOut# type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import sklearn.linear_model # type: ignore


def gridsearch_scoring(train, test, param_grid, model = SVR(), cv = 5, score = 'r2'):
    '''Takes feature matrix and targets, splits them by current K-fold split, and
    applies GridSearchCV() over the dictionary of parameters for the specified
    model. Then fits a model based on the desired metric, and returns the 
    R2, MAE and MSE scores of the model.'''


    X_train, y_train = train
    grid_model = GridSearchCV(model, param_grid = param_grid, verbose = 0, cv = cv, refit = False, scoring = score)
    grid_model.fit(X_train, y_train)

    BEST_PARAMS = grid_model.best_params_

    best_model = model.set_params(**BEST_PARAMS)
    best_model.fit(X_train, y_train)

    X_test, y_test = test
    y_pred = best_model.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    
    return [R2, MAE, MSE], BEST_PARAMS

def loo_scoring(inputs, targets, param_grid, model = SVR(), gscv = 5, score = 'r2'):
    '''Takes feature matrix and targets, splits them by current K-fold split, and
    applies GridSearchCV() over the dictionary of parameters for the specified
    model. Then fits a model based on the desired metric, and returns the 
    R2, MAE and MSE scores of the model.'''

    outer_loo = LeaveOneOut()

    all_pred = []
    best_params = []

    for gs_idx, test_idx in outer_loo.split(inputs):
        X_train = inputs.loc[gs_idx]
        y_train = targets.loc[gs_idx]

        grid_model = GridSearchCV(model, param_grid = param_grid, verbose = 0, cv = gscv, refit = False, scoring = score)
        grid_model.fit(X_train, y_train)

        params = grid_model.best_params_
        best_params.append(params)

        best_model = model.set_params(**params)
        best_model.fit(X_train, y_train)

        X_test = inputs.loc[test_idx]
        y_pred = best_model.predict(X_test)
        all_pred.append(y_pred)

    R2 = r2_score(targets, all_pred)
    MAE = mean_absolute_error(targets, all_pred)
    MSE = mean_squared_error(targets, all_pred)

    return [R2, MAE, MSE], best_params