import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
#import csv
#csv needed to get values for both the test and training accuracy for different iterations

def mean_squared_error(X,Y,W):
    for i in range(len(X)):
        squared_error=np.sum((Y[i]-np.dot(W,X[i])))/len(X)
    return squared_error

def gradient_descent(X,Y,learning_rates,threshold=10e-6)
    
    n = len(x) 
    costs = []
    current_weight = np.zeros(X.shape[1])
    delta = 0
    new_weight=np.zeros(X.shape[1])
    while(delta>threshold):
        

    # Estimation of optimal parameters
    for i in range(iterations):
         
        # Making predictions
        y_predicted = (current_weight * x) + current_bias
         
        # Calculationg the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_weight)
         
        # Calculating the gradients
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")
     
     
    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
     
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

if __name__ == "__main__":
    main()