# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold

def onehotencoding(data, features='all'):
    new_data = pd.DataFrame(pd.get_dummies(data[cat_columns],columns=data[cat_columns].columns,sparse=False))
    return new_data

train = pd.read_csv('../data/train.csv', sep=';')
test = pd.read_csv('../data/test.csv', sep=';')
# replace NaN values with zero
train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)
    # prepare data for label encoding
# first save target in a separate variable and drop index, id, codepostal
target_train = pd.DataFrame(train['prime_tot_ttc'])
train.drop(['id','codepostal','prime_tot_ttc'], axis = 1, inplace=True)
test.drop(['id','codepostal'], axis = 1, inplace=True)
train.reset_index(drop=True)
test.reset_index(drop=True)
# stack train and test sets for label encoding
stack = pd.concat([train,test])
#find categorical variables for encoding
cat_columns = list(stack.select_dtypes(include=['category','object']))


# apply ohe function
stack = pd.concat([stack, onehotencoding(stack, cat_columns)], axis=1)
stack.drop(cat_columns, axis = 1, inplace=True)
print stack.head()
# devide back new data frame on train and test sets
# you should get train set shape is (300000, 266) and test set shape is (30000, 266)
train = stack.head(300000)
test = stack.tail(3000)

## Modeling and CV
#initializing kfold for our cross validation
kf = KFold(len(train.index),n_folds=3)


# metric mape
def mape(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean([np.abs(i - j)/j for i,j in zip(y_true,y_pred)]) *100
print mape([1,2,3],[1.1,2.2,3.3])

# apply cv and train RandomForestRegressor model
regressor = RandomForestRegressor()
print 'fitting'
regressor.fit(train, target_train.values.ravel())
print 'predicting'
predicted_train = regressor.predict(train.iloc[:30000])
print 'predicted'
print mape(target_train.iloc[:30000], predicted_train)


