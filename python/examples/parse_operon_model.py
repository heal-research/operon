import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from pmlb import fetch_data, dataset_names, classification_dataset_names, regression_dataset_names
from operon.sklearn import SymbolicRegressor

import seaborn as sns
import matplotlib.pyplot as plt

from sympy import parse_expr, symbols, lambdify

# fetch data
X, y = fetch_data('192_vineyard', return_X_y=True, local_cache_dir='./data/')

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True)

# do a regression
reg = SymbolicRegressor()
reg.fit(X_train, y_train)


model_str = reg.get_model_string(20)

variables = [symbols(v) for v in df.columns if v != 'Y']
print(variables)

expr = parse_expr(model_str)
print(expr)

X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()
# print(X.head)
f = lambdify(variables, expr)
y_pred = f(*X.T)

# do linear scaling
A = np.vstack([y_pred, np.ones(len(y_pred))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
y_pred = y_pred * m + c

print('r2:', mean_squared_error(y_pred, y))

fig, ax = plt.subplots(figsize=(12, 6))
xs = range(len(y_pred))
sns.lineplot(ax=ax, x=xs, y=y_pred)
sns.lineplot(ax=ax, x=xs, y=y)

