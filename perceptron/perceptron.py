#importing libraries
import pandas as pd
import numpy as np
import csv

'''DATA PREPROCESSING
INPUT- training and testing data
added bais col to X feature and in Y label convert "0" to "-1"'''
training_data=pd.read_csv('./bank-note/train.csv')
X_train = training_data.iloc[:,:-1].to_numpy()
bais_train=np.ones((871, 1))
X_train = np.append(X_train, bais_train, axis=1)
Y_train=training_data.iloc[:,4].to_numpy()
Y_train[Y_train == 0] = -1
#_____Preprocessing test data
testing_data=pd.read_csv('./bank-note/test.csv')
X_test = testing_data.iloc[:,:-1].to_numpy()
bais_test=np.ones((499, 1))
X_test = np.append(X_test, bais_test, axis=1)
Y_test=testing_data.iloc[:,4].to_numpy()
Y_test[Y_test == 0] = -1

###########################Standard perceptron###################################

def std_perceptron(X,Y,r,t):
    num_records, num_features = X.shape
    weight = np.append(np.zeros(num_features-1),1)
    #weight=np.append(weight, 1)
    print(weight)
    for a in range(t):
        indices = np.arange(Y.shape[0])
        np.random.shuffle(indices)
        Xshuff = X[indices]
        yshuff = Y[indices]
        for i in range(num_records):
            xi = Xshuff[i,:]
            yi = yshuff[i]
            if (yi*(np.dot(weight, xi)) <= 0):
                weight = weight + r * (yi*xi)
        y_pred=np.sign(np.dot(X, weight))
        error=np.sum(y_pred!=Y) / Y.shape[0]
        #print("for epoch")
        print("For epoch {} ,the training error is {}".format((a+1), error))
    print("Learnt Weight is",weight)
    return weight

def std_predict(X_test,Y_test,std_weight):
    y_pred=np.sign(np.dot(X_test, std_weight))
    error=np.sum(y_pred!=Y_test) / Y_test.shape[0]
    print("Testing Error is",error)

###########################Average perceptron###################################
def avg_perceptron(X,Y,r,t):
    num_records, num_features = X.shape
    weight = np.append(np.zeros(num_features-1),1)
    avg_weight = np.append(np.zeros(num_features-1),1)
    #weight=np.append(weight, 1)
    print(weight)
    count=0
    for a in range(t):
        indices = np.arange(Y.shape[0])
        np.random.shuffle(indices)
        Xshuff = X[indices]
        yshuff = Y[indices]
        for i in range(num_records):
            xi = Xshuff[i,:]
            yi = yshuff[i]
            if (yi*(np.dot(weight, xi)) <= 0):
                weight = weight + r * (yi*xi)
            avg_weight=avg_weight+weight
            count+=1
        y_pred=np.sign(np.dot(X, avg_weight))
        error=np.sum(y_pred!=Y) / Y.shape[0]
        
        #print("for epoch")
        print("For epoch {} ,the training error is {}".format((a+1), error))
    #print(count)
    #avg_weight=avg_weight/count
    print("Learnt Weight is",avg_weight)
    return avg_weight

def avg_predict(X_test,Y_test,avg_weight):
    y_pred=np.sign(np.dot(X_test, avg_weight))
    error=np.sum(y_pred!=Y_test) / Y_test.shape[0]
    print("Testing Error is",error)


########################### Voted perceptron###################################

def voted_perceptron(X,Y,r,t):
    num_records, num_features = X.shape
    weight = [np.append(np.zeros(num_features-1),1)]
    m=0
    C=[0]
    for a in range(t):
        indices = np.arange(Y.shape[0])
        np.random.shuffle(indices)
        Xshuff = X[indices]
        yshuff = Y[indices]
        for i in range(num_records):
            xi = Xshuff[i,:]
            yi = yshuff[i]
            if (yi*(np.dot(weight[m], xi)) <= 0):
                weight[m] += r * (yi*xi)
                weight.append(weight[m].copy())
                m=m+1
                C.append(1)
            else:
                C[m]+=1
    votes = np.array(list(zip(weight, C)), dtype=object)
    with open('weights.csv', 'w', newline='') as f:
        wrt = csv.writer(f)
        wrt.writerow(['b', 'x1', 'x2', 'x3', 'x4', 'C'])
        for w in votes:
            row = w[0]
            row = np.append(row, w[1])
            wrt.writerow(row)
    #print(len(W),len(C))
    #print(W)
    #return (W, C,weight)
    return votes

def voted_predict(X,Y,votes):
    y_pred=np.zeros(len(Y))
    for a in range(len(Y)):
        i=0
        for w,c in votes:
            i+=c*np.sign(np.dot(w,X[a]))
        y_pred[a]=np.sign(i)
    error=np.sum(y_pred!=Y) / Y.shape[0]
    print("Error is",error)




print("__________Training Voted perceptron________________")
votes=voted_perceptron(X_train, Y_train, r=0.01,t=10)
voted_predict(X_train,Y_train,votes)
print("_______testing Voted perceptron__________")
voted_predict(X_test,Y_test,votes)


print("__________Training Standard perceptron________________")
std_weight=std_perceptron(X_train, Y_train, r=0.01,t=10)
std_predict(X_test,Y_test,std_weight)

print("__________Training Average perceptron________________")
avg_weight=avg_perceptron(X_train, Y_train, r=0.01,t=10)
avg_predict(X_test,Y_test,avg_weight)

