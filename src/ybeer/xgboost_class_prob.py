'''
Coding Just for Fun
Created by burness on 16/1/25.
'''
from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
import xgb_1 as xgb


train_result = pd.DataFrame.from_csv("train_result.csv")
train_result_xgb = np.array(train_result['Response'] - 1)
# print train_result_xgb

train = pd.DataFrame.from_csv("train_dummied_v2.csv")
train.fillna(9999)
train_arr = np.array(train)
col_list = list(train.columns.values)

test = pd.DataFrame.from_csv("test_dummied_v2.csv")
test.fillna(9999)
test_arr = np.array(test)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# for i in range(train_arr.shape[1]):
#     print col_list[i], chi2_params[0][i]

# Standardizing
stding = StandardScaler()
train_arr = stding.fit_transform(train_arr)
test_arr = stding.transform(test_arr)

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [3], 'num_class': [8], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [7], 'num_round': [180],
              'subsample': [0.75]}

for params in ParameterGrid(param_grid):
    print params

    print 'start CV'

    # CV
    cv_n = 4
    kf = StratifiedKFold(np.array(train_result).ravel(), n_folds=cv_n, shuffle=True)
    metric = []
    meta_estimator_xgboost = np.zeros((train_arr.shape[0], 8))
    best_meta_estimator_xgboost = np.zeros((train_arr.shape[0], 8))
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
        y_train, y_test = train_result_xgb[train_index].ravel(), train_result_xgb[test_index].ravel()

        # train machine learning
        xg_train = xgboost.DMatrix(X_train, label=y_train)
        xg_test = xgboost.DMatrix(X_test, label=y_test)

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = params['num_round']
        xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

        # predict
        class_pred = xgclassifier.predict(xg_test)
        class_pred = class_pred.reshape(y_test.shape[0], 8)

        meta_estimator_xgboost[test_index, :] = class_pred

        # evaluate
        # print log_loss(y_test, class_pred)
        metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
        best_meta_estimator_xgboost = meta_estimator_xgboost
print 'The best metric is:', best_metric, 'for the params:', best_params

best_meta_estimator_xgboost = pd.DataFrame(best_meta_estimator_xgboost)
best_meta_estimator_xgboost.to_csv('xgboost_train_probabilities.csv')


# train machine learning
xg_train = xgboost.DMatrix(train_arr, label=train_result_xgb)
xg_test = xgboost.DMatrix(test_arr)

watchlist = [(xg_train, 'train')]

num_round = params['num_round']
xgclassifier = xgboost.train(best_params, xg_train, num_round, watchlist);

# predict
predicted_results = xgclassifier.predict(xg_test)
predicted_results = predicted_results.reshape(test.shape[0], 8)

predicted_results = pd.DataFrame(predicted_results)
predicted_results.to_csv('xgboost_test_probabilities.csv')
