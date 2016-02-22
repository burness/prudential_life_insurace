#-*-coding:utf-8-*-
'''
这个脚本用skearn的gridsearch  太慢了
'''
__author__ = 'burness'
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa

def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)

def get_params():

    params = {}
    params["objective"] = "reg:linear"
    # params["objective"] = "reg:logistic"
    params["eta"] = 0.08
    params["min_child_weight"] = 80
    params["subsample"] = 0.75
    params["colsample_bytree"] = 0.30
    params["silent"] = 1
    params["max_depth"] = 9
    plst = list(params.items())

    return plst

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response','Split']
# columns_to_drop = ['Id', 'Response','Response_2']
xgb_num_rounds = 250
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
all_data['Split'] = np.random.randint(5, size=all_data.shape[0])

# split train and test
train = all_data[all_data['Response']>0].copy()
# test数据和train在一起,因为之前有fillna(-1)
test = all_data[all_data['Response']<1].copy()

# split train data to train_0 add validation set
train_0 = train[train['Split']>0]
train_val = train[train['Split']==0]


# convert data to xgb data structure
xgtrain0 = xgb.DMatrix(train_0.drop(columns_to_drop, axis=1), train_0['Response'].values)
xgtrain_val = xgb.DMatrix(train_val.drop(columns_to_drop, axis=1),train_val['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)

# xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response_2'].values)
# xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response_2'].values)

# get the parameters for xgboost
# plst = get_params()
# print(plst)

# train model
param = {'colsample_bytree': [0.3,0.2,0.4], 'silent':[1],'min_child_weight':[80],
         'subsample':[0.75], 'n_estimators':[100,150,200,250],'learning_rate': [0.08,0.10,0.12,0.14,0.2],  'max_depth': [9]}
print 'xgb_model 初始化'
xgb_model = xgb.XGBClassifier( objective='reg:linear')

print 'model training'
clf = GridSearchCV(xgb_model, param,n_jobs=4, cv=2)
# model = xgb.train(plst, xgtrain0, xgb_num_rounds)

# xgb_model.fit(train_0.drop(columns_to_drop, axis=1), train_0['Response'].values)
clf.fit(train_0.drop(columns_to_drop, axis=1), train_0['Response'].values)
best_param = clf.best_params_
for para_name in sorted(best_param.keys()):
    print para_name
    print best_param[para_name]


# get preds
train_preds = clf.predict(train_val.drop(columns_to_drop, axis=1))
xgb_model_best = xgb.XGBClassifier(best_param)

# train_preds = xgb_model_best.predict(train_val.drop(columns_to_drop, axis=1))
# train_preds = xgb_model.predict(train_val.drop(columns_to_drop, axis=1))
print('Train score is:', eval_wrapper(train_preds, train_val['Response']))
# test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
# train_preds = np.clip(train_preds, -0.99, 8.99)
# test_preds = np.clip(test_preds, -0.99, 8.99)

# train offsets
# offsets = np.ones(num_classes) * -0.5
# offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
# offset_train_preds = np.vstack((train_preds, train_preds, train['Response_2'].values))

# for j in range(num_classes):
#     train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
#     offsets[j] = fmin_powell(train_offset, offsets[j])

# apply offsets to test
# data = np.vstack((test_preds, test_preds, test['Response'].values))
# for j in range(num_classes):
#     data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
#
# final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
#
# preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
# preds_out = preds_out.set_index('Id')
# preds_out.to_csv('../data/xgb_logistic_offset_submission.csv')
