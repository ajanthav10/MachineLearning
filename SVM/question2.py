from random import sample
import numpy as np
import pandas as pd


def primal_SVM(data, weights, epochs, C, learning_rate, d = 1/3)->np.ndarray:
    
    for epoch in range(epochs):
        l_r = learning_rate/(1+epoch*learning_rate/d)
        shuffled_data = sample(list(data), k = len(data))
        shuffled_data = np.array(shuffled_data)
        for i in shuffled_data:
            inputs = i[:-1]
            pred = np.dot(inputs,weights)
            if 1-i[-1]*pred >= 0:
                weights = (1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
                weights += l_r*C*len(data)*i[-1]*inputs
            else:
                weights = (1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
    return weights
def primal_SVM1(data, weights, epochs, C, learning_rate)->np.ndarray:
    
    for epoch in range(epochs):
        l_r = learning_rate/(1+epoch)
        shuffled_data = sample(list(data), k = len(data))
        shuffled_data = np.array(shuffled_data)
        for i in shuffled_data:
            inputs = i[:-1]
            pred = np.dot(inputs,weights)
            if 1-i[-1]*pred >= 0:
                weights =(1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
                weights += l_r*C*len(data)*i[-1]*inputs
            else:
                weights = (1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
    return weights
def predict(entry, weight):
    y_label = 0
    for i in range(len(entry)-1):
        y_label += entry[i]*weight[i]
    if y_label >= 0:
        return 1
    else:
        return -1

def error_test(data, weight):
    error = 0
    for i in data:
        y_pred = predict(i, weight)
        if i[-1] != y_pred:
            error += 1
    #error/len(data)
    return error/len(data)

def main():
    '''DATA PREPROCESSING
    INPUT- training and testing data
    added bais col to X feature and in Y label convert "0" to "-1"
    '''
    train_data=pd.read_csv('./bank-note/train.csv')
    X_train = train_data.iloc[:,:-1].to_numpy()
    bais_train=np.ones((871, 1))
    X_train = np.append(X_train, bais_train, axis=1)
    y_train=train_data.iloc[:,4].to_numpy()
    y_train[y_train == 0] = -1
    trainset=np.column_stack((X_train,y_train))
    #_____Preprocessing test data
    test_data=pd.read_csv('./bank-note/test.csv')
    X_test = test_data.iloc[:,:-1].to_numpy()
    bais_test=np.ones((499, 1))
    X_test = np.append(X_test, bais_test, axis=1)
    y_test=test_data.iloc[:,4].to_numpy()
    y_test[y_test == 0] = -1
    testset=np.column_stack((X_test,y_test))
    weights=np.zeros(5,)
    C = [100/873, 500/873, 700/873]
    print("____________________________________Question 2A ___________________________________")
    for i in C:
        obtained_weight = primal_SVM(trainset, weights, 100, i, 0.01, d = 1/3)
        print("The value of C is",i)
        print("Weight vector is",obtained_weight)
        #print(w0)
        
        trainerror = error_test(trainset, obtained_weight)
        testerror = error_test(testset, obtained_weight)
        print("Training Error =",trainerror)
        print("Testing Error =",testerror)
        
    print("____________________________________Question 2B ___________________________________")
    for i in C:
        w1 = primal_SVM1(trainset, weights, 100, i, 0.01)
        print("The value of C is",i)
        print("Weight vector is",w1)
        #print(w0)
        
        trainerror = error_test(trainset, w1)
        testerror = error_test(testset, w1)
        print("Training Error =",trainerror)
        print("Testing Error =",testerror)
        

if __name__ == "__main__":
    main()
    