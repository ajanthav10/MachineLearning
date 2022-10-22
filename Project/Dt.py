import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


train = './train_final.csv'
test = './test_final.csv'


df = pd.read_csv(train)

Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

labelProcessor = preprocessing.LabelEncoder()

for i in range(14):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])



Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

decTree = tree.DecisionTreeClassifier(max_depth=6)
decTree = decTree.fit(X,Y)

test_predictions = decTree.predict(X)
print(accuracy_score(Y, test_predictions))

testDF = pd.read_csv(test)

for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    

predictions = decTree.predict(testDF.iloc[:,1:15])


predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_1.csv', index=False, header=['Prediction','ID'])