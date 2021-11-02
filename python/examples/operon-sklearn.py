# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import pearsonr

from operon import RSquared
from operon.sklearn import SymbolicRegressor

from pmlb import fetch_data, dataset_names, classification_dataset_names, regression_dataset_names
#print(regression_dataset_names)

X, y = fetch_data('1027_ESL', return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=1234)

reg = SymbolicRegressor(
        allowed_symbols='add,sub,mul,div,constant,variable',
        offspring_generator='basic',
        local_iterations=10,
        n_threads=4,
        error_metric = ['r2', 'shape'],
        random_state=1234
        )

reg.fit(X_train, y_train)
print(reg.get_model_string(2))
print(reg._stats)

y_pred_train = reg.predict(X_train)
print('r2 train (sklearn.r2_score): ', r2_score(y_train, y_pred_train))
# for comparison we calculate the r2 from _operon and scipy.pearsonr
print('r2 train (operon.rsquared): ', RSquared(y_train, y_pred_train))
r = pearsonr(y_train, y_pred_train)[0]
print('r2 train (scipy.pearsonr): ', r * r)

# crossvalidation
sc = make_scorer(RSquared, greater_is_better=True)
scores = cross_val_score(reg, X, y, cv=5, scoring=sc)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
