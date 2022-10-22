from operator import inv
import numpy as np 
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv    
import sys
import pandas as pd
from numpy import linalg as LA
import random


##parsing a list of learning rates [1,0.5,0.25,0.125,0.001]
# import the library
import argparse
# create the parser 
parser = argparse.ArgumentParser(description='Stochastic Gradient Descent')
# Add an argument
parser.add_argument('--r', type=float, required=True, help='learning rate' ,nargs='+')
parser.add_argument('--lr', type=float, default=0.001,required=False, help='learning rate' ,nargs='+')

# Parse the argument
args = parser.parse_args()
# Print "Rate value" + the user input argument
#print('Rate,', args.r)

def mean_square_error(X:np.ndarray, Y:np.ndarray, current_W:np.ndarray):
    square_error = 0 
    Y_pred = np.empty_like(Y)
	

    for i in range(len(X)):
        Y_pred[i]=np.dot(current_W, X[i])
        temp = (np.power((Y[i] -Y_pred[i]),2))
        square_error += temp 
    return 0.5*square_error

def gradient_descent(X:np.ndarray, Y:np.ndarray,r:float,threshold):
	costs = []  
	current_W = np.zeros(X.shape[1])
	#print(W.shape)
	#print(r)

	norm = math.inf

	while norm > threshold:
		updated_W = np.zeros(X.shape[1])
		
		for j in range(len(X[0])):
			temp = 0 
			for i in range(len(X)):
				temp += X[i][j] *(Y[i] - np.dot(current_W, X[i]))
			updated_W[j] = temp 

		updated_W = current_W + r*updated_W
		#r=r*0.5
		norm = LA.norm(current_W - updated_W)
		costs.append(mean_square_error(X, Y, current_W))
		#r=r*0.5
		current_W = updated_W

	#costs.append(mean_square_error(X, Y, W))
	return current_W, costs

def stochastic_gradient_descent(X, Y, lr,threshold):

	current_W = np.zeros(X.shape[1])
	norm = math.inf
	costs = [mean_square_error(X, Y, current_W)]
	while norm > threshold:
		i = random.randrange(len(X))
		running_w = np.zeros(X.shape[1])
		for j in range(len(X[0])): 
			running_w[j] = X[i][j] *(Y[i] - np.dot(current_W, X[i]))
		updated_W = current_W + lr*running_w
		current_W = updated_W
		new_cost = mean_square_error(X, Y, current_W) 
		norm = abs(new_cost - costs[-1])
		costs.append(new_cost)

	return current_W, costs

def main():
    '''DATA PREPROCESSING'''
    train = pd.read_csv('./concrete/train.csv',names=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr','output'])
    test = pd.read_csv('./concrete/test.csv', names=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr','output'])

    #split the X features and Y in training and testing 
    features=['cement','slag','Fly ash','water','SP','Coarse Aggr','Fine Aggr']
    X_train=train[features]
    one_train = np.ones(X_train.shape[0])
    D_train = np.column_stack((one_train, X_train))
    X_test=test[features]
    one_test = np.ones(X_test.shape[0])
    D_test = np.column_stack((one_test, X_test))
    Y_train=train['output']
    Y_test=test["output"]
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_test.shape)
    #print(Y_train.shape)
    
    print("********** Part 4(a) - Implementing batch Gradient Descent**********")
    r = args.r
    GD_Weights, GD_costs = gradient_descent(D_train, Y_train,r,10e-6)
    test_GD_cost_value = mean_square_error(D_test, Y_test, GD_Weights)
    print("Learning rate: ", r)
    print("GD_Learned weight vector: ", GD_Weights)
    print("Test data cost function value: ", test_GD_cost_value)
    dict={"GD Learnt Weight": GD_Weights}
    fig1 = plt.figure()
    plt.plot(GD_costs)
    fig1.suptitle('Batch Gradient Descent on Test data')
    plt.xlabel('Iterations')
    plt.ylabel('J (Cost Function Value)')
    plt.show()
    fig1.savefig("BGD_cost_function.png")

    print("********** Part 4(b) - Implementing stochastic_gradient_descent **********")
    lr = args.lr
    #print(lr)
    SGD_Weights, SGD_costs = stochastic_gradient_descent(D_train, Y_train,lr,10e-10)
    test_SGD_cost_value = mean_square_error(D_test, Y_test, SGD_Weights)
    print("Learning rate: ", lr)
    print("Learned weight vector: ",SGD_Weights)
    print("Test data cost function value: ", test_SGD_cost_value)
    #storing the results to csv
   

    fig1 = plt.figure()
    plt.plot(SGD_costs)
    fig1.suptitle('Stochastic Gradient Descent on Test data')
    plt.xlabel('Iterations')
    plt.ylabel('J (Cost Function Value)')
    plt.show()
    fig1.savefig("Stochastic_cost_functio.png")
    dict.__setitem__('SGD_Learnt Weights',SGD_Weights)
    #dict ={'Learnt Weights':SGD_Weights}
    df=pd.DataFrame(dict)
    df.to_csv('results.csv')
    print("********** Part 4(c)  Optimal weight vector with analytical form**********")

    #multiply X and Y
    #print(Y_train.shape)
    #multiplication of X and X.Tnew_D_train = D_train.T
    XX = np.matmul(D_train.T, D_train)
    inv_XX = inv(XX)
    analytical_w = np.matmul(np.matmul(inv_XX, D_train.T), Y_train)
    print(analytical_w.shape)
    test_cost_value = mean_square_error(D_test, Y_test, analytical_w)
    print("The learned weight vector: ", analytical_w)
    diffweight_GD = LA.norm(analytical_w - GD_Weights)
    diffweight_SGD = LA.norm(analytical_w - SGD_Weights)
    print("Comparing learned weights by Gradient Descent, Stochastic GD with Optimal weights",diffweight_GD,diffweight_SGD)
    print("Test data cost function value: ", test_cost_value)

if __name__ == "__main__":
    main()

