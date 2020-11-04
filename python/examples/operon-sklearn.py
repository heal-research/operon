import operon
import _operon
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from pmlb import fetch_data, dataset_names, classification_dataset_names, regression_dataset_names
#print(regression_dataset_names)

X, y = fetch_data('1027_ESL', return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

reg = operon.SymbolicRegressor(
        allowed_symbols='add,sub,mul,div,constant,variable',
        offspring_generator='basic',
        local_iterations=0,
        n_threads=24,
        seed=None,
        )

reg.fit(X_train, y_train, show_model=True)
print(reg._stats)

y_pred_train = reg.predict(X_train)
print('r2 train (sklearn.r2_score): ', r2_score(y_train, y_pred_train))
# for comparison we calculate the r2 from _operon and scipy.pearsonr
print('r2 train (operon.rsquared): ', _operon.RSquared(y_train, y_pred_train))
r = pearsonr(y_train, y_pred_train)[0]
print('r2 train (scipy.pearsonr): ', r * r)

# crossvalidation
scores = cross_val_score(reg, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
