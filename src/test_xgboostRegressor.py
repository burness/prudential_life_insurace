'''
Coding Just for Fun
Created by burness on 16/1/27.
'''
from xgboostRegressor import XGBoostRegressor
import pandas as pd
import numpy as np
columns_to_drop = ['Id', 'Response', 'Medical_History_1']
xgb_num_rounds = 1200
num_classes = 8
eta_list = [0.05] * 200
eta_list = eta_list + [0.02] * 1000

print("Load the data using pandas")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# combine train and test
all_data = train.append(test)

# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
# create any new variables
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

all_data['Split'] = np.random.randint(3, size=all_data.shape[0])


print('Eliminate missing values')
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train = all_data[all_data['Response']>0].copy()
train0  = train[train['Split'] == 0]
train_validation = train[train['Split'] > 0]
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
# xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
# xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)

#---validation parse
# trainFeatures = train.drop(columns_to_drop, axis=1)
# trainLabels = train['Response'].values
# validationFeatures = train_validation.drop(columns_to_drop,axis=1)
# validationLabels = train_validation['Response'].values
# xgbRegression = XGBoostRegressor()
# xgbRegression.fit(trainFeatures, trainLabels)
# print xgbRegression.score(validationFeatures, validationLabels)

trainFeatures = train.drop(columns_to_drop,axis=1)
trainLabels = train['Response'].values
testFeatures = test.drop(columns_to_drop,axis=1)
xgbRegression = XGBoostRegressor()
xgbRegression.fit(trainFeatures,trainLabels)
testLabels = xgbRegression.predict(testFeatures)



test_preds = np.clip(testLabels, -0.99, 8.99)

# train offsets
# offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
#
#
# apply offsets to test
# data = np.vstack((test_preds, test_preds, test['Response'].values))
# for j in range(num_classes):
#     data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
#
# final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
#
preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('xgboost_qianqian.csv')