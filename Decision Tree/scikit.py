import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
test=pd.read_csv('/home/ajantha/Desktop/Repos/MachineLearning/Decision Tree/car/test.csv')
train=pd.read_csv('/home/ajantha/Desktop/Repos/MachineLearning/Decision Tree/car/train.csv')
Y_test=np.array(test.iloc[:,-1])
Y_train=np.array(train.iloc[:,-1])
X_test=np.array(test.iloc[:,:-1])
X_train=np.array(train.iloc[:,:-1])
clf=DecisionTreeClassifier(criterion='entropy',max_depth=6)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
score=accuracy_score(y_pred,Y_test)
print(score)
