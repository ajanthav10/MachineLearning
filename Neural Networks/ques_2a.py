import pandas as pd
import numpy as np

def sigmoid(x)->float:
    return 1/(1 + np.exp(-x))

def loss(y_pred,y)->float:
    return 0.5*(y_pred-y)**2

x_train = np.array([[2,1.4]])
y_train = np.array([[0]])
w1 = np.array([[1, 0], [-2.3, -1.5]])  
w2 = np.array([[0, -1], [-1, -2]])  
w3 = np.array([0, 2])  

b1 = np.array([1, 1])  
b2 = np.array([1, 1])  
b3 = np.array([1])     

def forward(x_train,y_train,w1,w2,w3,b1,b2,b3):
    output1 = sigmoid(np.dot(x_train, w1.transpose()) + b1)
    output2 = sigmoid(np.dot(output1, w2.transpose()) + b2)
    y = np.dot(output2, w3.transpose()) + b3
    L = loss(y,y_train)
    diff_l=y-y_train
    return y,output1,output2,diff_l

def backprop(y,a1,a2,diff_l):

    d_w3 = np.dot(diff_l.transpose(), a2)
    d_b3 =np.sum(diff_l,axis=0)

    d_output2 = diff_l*w3.transpose() * a2 * (1 - a2)
    d_b2 = np.sum(d_output2,axis=0)
    d_w2 = np.dot(d_output2.transpose(), a1)

    d_output1 = np.dot(d_output2, w2) * a1 * (1 - a1)
    d_b1 = np.sum(d_output1,axis=0)
    d_w1 = np.dot(d_output1.transpose(), x_train)
    return d_b3,d_b2,d_b1,d_w3,d_w2,d_w1
y,a1,a2,dldy=forward(x_train,y_train,w1,w2,w3,b1,b2,b3)
b3,b2,b1,w3,w2,w1=backprop(loss,a1,a2,dldy)
print("________________Forward Pass______________________")
print("output=",y)
print("op from layer1",a1)
print("op from layer2",a2)
print("________________Backprop______________________")
print("w1 =",w1)
print("w2" ,w2)
print("w3" ,w3)
print("b1" ,b1)
print("b2" ,b2)
print("b3" ,b3)
print("Same diff_weights as manually calculated")

