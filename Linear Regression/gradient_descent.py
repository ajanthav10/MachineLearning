import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from numpy import linalg as lin
#import csv
#csv needed to get values for both the test and training accuracy for different iterations

'''def mean_squared_error(X,Y,W):

    squared_error=0
    for i in range(len(X)):
        squared_error+=(Y[i] - np.dot(W, X[i]))**2 
    return squared_error*0.5'''

def mean_squared_error(X,Y,W):
    
    squared_error=0
    y_pred=np.zeros(Y.shape)
    for i in range(len(X)):
        y_pred[i]=np.dot(W,X[i])
        squared_error=(Y[i] -y_pred[i])**2 
    return squared_error*0.5

'''def mse(w,xy):
    (x,y) = xy
    
    # Compute output
    # keep in mind that wer're using mse and not mse/m
    # because it would be relevant to the end result
    o = np.sum(x*w,axis=1)
    mse = np.sum((y-o)*(y-o))
    mse = mse/2
    return mse    '''

def gradient_descent(X,Y,learning_rates,threshold):#=10e-6'''):
    
    n = len(X) 
    costs = []
    current_weight = np.zeros(X.shape[1])
    delta = 0
    new_weight=np.zeros(X.shape[1])
    while(delta>threshold):
        for i in range(len(X[0])):
            grad_vector=0
            for j in range(len(X)):
                grad_vector+=X[i][j] * (Y[i]-np.dot(current_weight,X[i]))
            new_weight[j] = -(grad_vector)
        updated_weight=current_weight-learning_rates*new_weight
        delta = lin.norm(-new_weight+updated_weight)
        costs.append(mean_squared_error(X,Y,W))
        current_weight=updated_weight

    return current_weight,costs


def main():
    '''DATA PREPROCESSING'''
    train = pd.read_csv('./concrete/train.csv',names=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr','output'])
    test = pd.read_csv('./concrete/test.csv', names=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr','output'])

    #split the X features and Y in training and testing 
    features=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr']
    X_train=train[features]
    X_test=test[features]
    Y_train=train['output']
    Y_test=test["output"]
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_test.shape)
    #print(Y_train.shape)

    print("********** Part 4(a) **********")
    print("Batched gradient descent experiment")

    r = 0.01
    W, costs = gradient_descent(X_train, Y_train,r,10e-6)
    test_cost_value = mean_squared_error(X_test, Y_test, W)
    print("Learning rate: ", r)
    print("The learned weight vector: ", W)
    print("Test data cost function value: ", test_cost_value)
    fig1 = plt.figure()
    plt.plot(costs)
    fig1.suptitle('Gradient Descent ', fontsize=20)
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Cost Function Value', fontsize=16)
    plt.show()
    fig1.savefig("BGD_cost_function.png")
    print("Figure has been saved!")

if __name__ == "__main__":
    main()