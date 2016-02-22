__author__ = 'burness'
from itertools import product
eta = [0.05,0.06,0.07,0.08,0.09,0.10,0.12]
min_child_weight = [50,60,70]
subsample = [0.5,0.55,0.6,0.65]
colsample_bytree = [0.25,0.30,0.35,0.40]
max_depth = [7,8,9]

with open('../data/params_0119.txt','a') as fwrite:
    for i in product(eta,min_child_weight,subsample,colsample_bytree,max_depth):
        line = ','.join([str(i) for i in list(i)])
        fwrite.write(line+'\n')
