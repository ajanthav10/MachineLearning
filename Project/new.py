# importing the libs
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
#from scipy.stats import pointbiserialr,spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data
train=pd.read_csv("./train_final.csv",names=['age', 'workclass', 'fnlwgt', 'education', 'education.num',
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
       'income>50K'])
test=pd.read_csv("./test_final.csv",names=['age', 'workclass', 'fnlwgt', 'education', 'education.num',
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
       'income>50K'])
print(train.shape)
train.info()
X=['age','workclass', 'fnlwgt', 'education', 'education.num',
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
Y=['income>50K']
train_x=train[X]
train_y=train[Y]
labelProcessor = preprocessing.LabelEncoder()
for i in range(14):
    train.iloc[:,i] = labelProcessor.fit_transform(train.iloc[:,i])

train_x=train[X]
train_y=train[Y]
print(train_x)
print(train_y)

'''
col_names = train_x.columns
print(col_names)
num_data = train_x.shape[0]
for c in col_names:
  missing=train_x[c].isin(["?"]).sum()
  if missing>0:
    print(c,missing)'''

clf = tree.DecisionTreeClassifier(max_depth=10)
clf=clf.fit(train_x,train_y)
y_pred=clf.predict(test)
print(accuracy_score,(test_y,y_pred))
