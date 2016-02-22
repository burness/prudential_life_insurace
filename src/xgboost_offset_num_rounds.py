#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/1/15.
'''

__author__ = 'burness'
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import sys

def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return 'score',quadratic_weighted_kappa(yhat, y)

def get_params(objective="reg:linear",eta=0.05,min_child_weight=50,subsample=0.5,colsample_bytree=0.30,silent=1,max_depth=9):

    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.06
    params["min_child_weight"] = 50
    params["subsample"] = 0.4
    params["colsample_bytree"] = 0.25
    params["silent"] = 1
    params["max_depth"] = 8
    # plst = list(params.items())
    return params

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response','Split']
# columns_to_drop = ['Id', 'Response','Response_2']
xgb_num_rounds = 500
num_classes = 8

print("Load the data using pandas")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# combine train and test
all_data = train.append(test)

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]

print('Eliminate missing values')
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)
# all_data['Response_2'] = all_data['Response'].astype(int)/(8*1.0)
# print all_data['Response_2'].head()
# Provide split column
all_data['Split'] = np.random.randint(3, size=all_data.shape[0])

# split train and test
train = all_data[all_data['Response']>0].copy()
# test数据和train在一起,因为之前有fillna(-1)
test = all_data[all_data['Response']<1].copy()

# split train data to train_0 add validation set
train_0 = train[train['Split']>0]
train_val = train[train['Split']==0]


# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)


# get the parameters for xgboost
plst = get_params()
print(plst)

'''
Here is work for cv
'''
xgb.cv(plst,xgtrain,nfold=3,num_boost_round=xgb_num_rounds,show_progress=True)
# test = xgb.cv(plst,xgtrain,nfold=3,num_boost_round=3)
# model = xgb.train(plst,xgtrain,xgb_num_rounds)
# train_preds = model.predict(xgtrain_val, ntree_limit=model.best_iteration)
# print "Train params is:"
# print(plst)
# print('Train score is:', eval_wrapper(train_preds, train_val['Response']))
# test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
# offsets = np.ones(num_classes) * -0.5
# offset_train_preds = np.vstack((train_preds, train_preds, train_val['Response'].values))
# for j in range(num_classes):
#     train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
#     offsets[j] = fmin_powell(train_offset, offsets[j])
#
# # apply offsets to test
# data = np.vstack((test_preds, test_preds, test['Response'].values))
# for j in range(num_classes):
#     data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
#
# final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
#
# preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
# preds_out = preds_out.set_index('Id')
# # csv_name = 'xgb_offset_0109_'+str(j)+'submission.csv'
# preds_out.to_csv('../data/result_split/%s/submission%d.csv'%(folder_name,jj))
# print('../data/result_split/%s/submission%d.csv'%(folder_name,jj))
