import pandas as pd
import numpy as np

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
dtrain_val = pd.read_csv("../data/sample_submission.csv")

#We transform categorical values to dummies 0/1

categorical = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41','Medical_History_1', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']

from sklearn.feature_extraction import DictVectorizer
from ml_metrics import quadratic_weighted_kappa
def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

train_ohd,_,_=one_hot_dataframe(train,categorical,replace=True)
test_ohd,_,_=one_hot_dataframe(test,categorical,replace=True)

# train_ohd = train_ohd.drop(["Insurance_History_5","Family_Hist_2","Family_Hist_3",
#                     "Family_Hist_4","Family_Hist_5","Medical_History_10","Medical_History_24","Medical_History_32"],axis=1)
# test_ohd = test_ohd.drop(["Insurance_History_5","Family_Hist_2","Family_Hist_3",
#                            "Family_Hist_4","Family_Hist_5","Medical_History_10","Medical_History_24","Medical_History_32"],axis=1)

features=train_ohd.columns.tolist()
features.remove("Id")
features.remove("Response")
# train_features=train_ohd[features]
# test_features=test_ohd[features]

# for f in prudential_df.columns:
#     if f == "Response": continue
#     if prudential_df[f].dtype == 'float64':
#         prudential_df[f].fillna(prudential_df[f].mean(), inplace=True)
#         test_df[f].fillna(test_df[f].mean(), inplace=True)
#     else:
#         prudential_df[f].fillna(prudential_df[f].median(), inplace=True)
#         test_df[f].fillna(test_df[f].median(), inplace=True)

train_features=train_ohd[features]
test_features=test_ohd[features]
train_count = train_features.count()
train_count.to_csv('../data/train_count.txt')
for f in train_features.columns:
    if f=='Response': continue
    if train_features[f].dtype == 'float64':
        train_features[f].fillna(train_features[f].mean(),inplace=True)
        test_features[f].fillna(test_features[f].mean(),inplace=True)
    else:
        train_features[f].fillna(train_features[f].median(),inplace=True)
        test_features[f].fillna(test_features[f].median(),inplace=True)
# train_features=train_features.fillna(-9999)
# test_features=test_features.fillna(-9999)
# print(train_features)

y_train = train["Response"].values

import xgb_1 as xgb

#You can improve parameters if you want, this solution is not perfect at all

param = {'max_depth':6, 'eta':10**-1, 'silent':1, 'min_child_weight':3, 'subsample' : 0.7 ,"early_stopping_rounds":10,
         "objective": "multi:softmax","num_class":8,'eval_metric': 'mlogloss','colsample_bytree':0.65}



num_round=700
print y_train-1
print y_train
dtrain=xgb.DMatrix(train_features,label=y_train-1)
dtest=xgb.DMatrix(test_features)

mask = np.random.choice([False, True], len(train_features), p=[0.66, 0.34])
not_mask = [not i for i in mask]
# dtrain = xgb.DMatrix(train_features[mask],label=y_train[mask]-1)
# dtrain_val  = xgb.DMatrix(train_features[not_mask],label=y_train[not_mask]-1)
watchlist  = [(dtrain,'train')]

bst = xgb.train(param, dtrain, num_round, watchlist)

print("Training the model")
y_test_bst=bst.predict(dtest)
y_test_bst=y_test_bst+1
# for i in y_test_bst:
#     print i

#we need integers to fit the model

def output_function(x):
    return int(x)

y_test_bst_result=[output_function(y) for y in y_test_bst]

#write results

ids=test.Id.values.tolist()
n_ids=len(ids)
import csv

prediction_file = open("../data/xgb_2015-1227.csv", "w")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["Id","Response"])
for i in range(0,n_ids):
    prediction_file_object.writerow([ids[i],y_test_bst_result[i]])
