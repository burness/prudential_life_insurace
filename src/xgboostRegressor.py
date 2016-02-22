'''
Coding Just for Fun
Created by burness on 16/1/27.
'''
import xgboost as xgb
import numpy as np
from ml_metrics import rmse
class XGBoostRegressor():
    def __init__(self, num_boost_round=1900, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'reg:linear'})

    def fit(self, X, y, num_boost_round=None):

        params = {
            'booster':'gbtree',
            'max_depth':8,
            'min_child_weight':4,
            'eta':0.0125,#0.03
            'silent':1,
            'objective':'reg:linear',
            'eval_metric':'rmse',
            "seed":2,
            'subsample':0.7,
            "colsample_bytree":0.7,
            # "num_parallel_tree":1,
            "base_score":0.5,
            "alpha":0,
            "max_delta_step":1,
            "lambda":0,
            'lambda_bias':0,
            'num_boost_round':1900,
        }
        num_boost_round = num_boost_round or self.num_boost_round
        num_boost_round = 1900
        dtrain = xgb.DMatrix(X, label=y)
        # params = self.params
        if X.shape[1]==1:
            params.update({'colsample_bytree': 1.0})
        self.clf = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
        )
        self.fscore = self.clf.get_fscore()

        # bb=np.zeros(dtrain.num_col())
        bb = {}

        for ftemp, vtemp in self.fscore.items():
            # bb[int(ftemp[1:])]=vtemp
            bb[ftemp] = vtemp

        # bb=bb/float(bb.max())
        i = 0
        cc = np.zeros(dtrain.num_col())
        for feature, value in  bb.items():
            cc[i] = value
            i+=1
        self.coef_= cc

    def predict(self, X):
        dX = xgb.DMatrix(X)
        y = self.clf.predict(dX)
        return y
    def score(self, X, y):
        Y = self.predict(X)
        return self.rmse_loss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

    def rmse_loss(self,y,y_pred):
        return rmse(y,y_pred)