'''
Coding Just for Fun
Created by burness on 16/1/24.
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import random
import sys

random.seed(23)

def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)

def get_params(objective="reg:linear",eta=0.05,min_child_weight=50,subsample=0.5,colsample_bytree=0.30,silent=1,max_depth=9):

    params = {}
    params["objective"] = objective
    params["eta"] = eta
    params["min_child_weight"] = min_child_weight
    params["subsample"] = subsample
    params["colsample_bytree"] = colsample_bytree
    params["silent"] = silent
    params["max_depth"] = max_depth
    plst = list(params.items())

    return plst

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response','Split']
xgb_num_rounds = 500
num_classes = 8

# print("Load the data using pandas")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# combine train and test
all_data = train.append(test)

# create any new variables
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[1]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[2]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]


# print('Eliminate missing values')
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# Provide split column
all_data['Split'] = np.random.randint(5, size=all_data.shape[0])

# split train and test
train = all_data[all_data['Response']>0].copy()
train0 = train[train['Split'] > 0]
train_validation = train[train['Split'] == 0]
# test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
xgtrain0 = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
# xgtrain0 = xgb.DMatrix(train0.drop(columns_to_drop, axis=1), train0['Response'].values)
# xgtrain_validation = xgb.DMatrix(train_validation.drop(columns_to_drop, axis=1), train_validation['Response'].values)
# xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)

# get the parameters for xgboost
file_name = sys.argv[1]
folder_name = file_name.split('.')[0]
with open('../data/params_0119/'+file_name) as fread:
    jj=0
    for line in fread.readlines():
        params = line.split(',')
        params[-1] = params[-1].strip()
        jj+=1
        # get the parameters for xgboost
        plst = get_params(eta=params[0], min_child_weight=params[1],subsample=params[2],colsample_bytree=params[3],max_depth=params[4])

        # print 'file: '+file_name+",params: "+str(jj)

        # train model
        model = xgb.cv(plst, xgtrain0,  num_boost_round=xgb_num_rounds,metrics=['auc'],show_progress=True,show_stdv=True)

        # get preds
        # train_validation_preds = model.predict(xgtrain_validation, ntree_limit=model.best_iteration)
        # print 'file: '+file_name+",params: "+str(jj)+' Train score is:', eval_wrapper(train_validation_preds, train_validation['Response'])
