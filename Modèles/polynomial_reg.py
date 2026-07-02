import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline

# Dans ce fichier, contrairement au fichier ACP.py où l'on a fait des modèles sur des paramètres bien définis, 
# on utilise grid_search pour être sûr d'avoir le meilleur ridge et degré


def polyreg(X, Y) :
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

    pipeline = Pipeline([('poly_features', preprocessing.PolynomialFeatures(include_bias=False)), ('ridge', linear_model.Ridge())])

    param_grid = {
        'poly_features__degree': [1,2,3,4,5],
        'ridge__alpha' : [0.001, 0.01, 0.1, 1.0]
    }

    grid_search = model_selection.GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = 5,
        scoring = 'neg_mean_squared_error',
        n_jobs = -1
    )

    grid_search.fit(X_train, Y_train)
    polyreg_ridge = grid_search.best_estimator_
    y_pred = polyreg_ridge.predict(X_test)

    mae = metrics.mean_absolute_error(Y_test, y_pred)
    rmse = metrics.root_mean_squared_error(Y_test, y_pred)
    r2 = metrics.r2_score(Y_test, y_pred)
    
    return (polyreg_ridge, mae, rmse, r2) #model puis MAE puis RMSE puis r2


