references :-https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
#https://towardsdatascience.com/build-a-simple-neural-network-using-pytorch-38c55158028d
#https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
#https://analyticsindiamag.com/step-by-step-guide-to-build-a-simple-neural-network-in-pytorch-from-scratch/
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from IPython.display import clear_output

#data preprocessing
train_data = pd.read_csv("./bank-note/train.csv", header=None)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values

test_data = pd.read_csv("./bank-note/test.csv", header=None)
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values
X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32)
Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32)
from tqdm import trange

class NN(nn.Module):
    def __init__(self, weight_init_scheme=None, num_hidden_layers=3, num_hidden_size=5, act_func="RELU",
                 input_feature=32):
        super(NN, self).__init__()
        self.sequential = nn.Sequential().to(dtype=torch.float32)
        if act_func == "RELU":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        for i in range(num_hidden_layers - 1): 
            linear = nn.Linear(input_feature, num_hidden_size).to(dtype=torch.float32)
            if weight_init_scheme == "Xavier":
                nn.init.xavier_uniform_(linear.weight)
            else:
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.sequential.add_module('layer_norm{}'.format(i), linear)
            if act_func == "RELU":
                self.sequential.add_module('act{}'.format(i), self.act)
            input_feature = num_hidden_size

        self.sequential.add_module('layer_norm{}'.format(num_hidden_layers - 1), 
                                   nn.Linear(num_hidden_size, 1)) 

    def forward(self, input):
        ''' Implmenting forward pass with sequential model to ip 
        ip - dataset
        op - forward pass after sigmoid '''
        temp = self.sequential(input)
        x = torch.sigmoid(temp)
        return x


def training(network,criterion, optimizer, X, y): 
    ''' implementing training of 
    ip - network: a PyTorch neural network model
         criterion: a PyTorch loss function
         optimizer: a PyTorch optimizer ,x and y
    op - loss
    '''
    network.train()
    optimizer.zero_grad()
    output = network(X)
    y = torch.unsqueeze(y,1)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss

def minibatch(ip, T, size, shuffle=True):
    '''Iterates over the datasets in batches
    ip- dataset,targetdataset, size of minibatch
    Trains on  mini-batches of data 
    '''
    assert len(ip) == len(T)
    if shuffle:
        indices = np.random.permutation(len(ip))
    for a in range(0, len(ip) - size + 1, size):
        if shuffle:
            i = indices[a:a + size]
        else:
            i = slice(a, a + size)
        yield ip[i], T[i]


def predict(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    network.eval()
    sign = np.sign(network(X))
    return sign




def main():
    print("___________________TANH Activation Func_______________________")
    num_hidden_layers = [3,5,9]
    width = [5,10,25,50,100]
    learning_rate= 1e-5
    alpha = 5
    input_shape = X_train.shape[-1]
    print("Hidden_layer| Width |Train_error| Test_error")
    for num_hidden_layer in num_hidden_layers:
        for hidden_layer_size in width:
            network = NN(weight_init_scheme="Xavier",num_hidden_layers=num_hidden_layer, num_hidden_size=hidden_layer_size,act_func="Tanh",input_feature=input_shape)
            L_train = []
            L_test = []
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            for epoch in range(25):
                for x_batch, y_batch in minibatch(X_train, Y_train, size=32, shuffle=True):
                    training(network, criterion, optimizer, x_batch, y_batch)

                network.eval()
                acc_train = np.mean((network(X_train).reshape(-1).detach().numpy().round() != Y_train.numpy()))
                L_train.append(acc_train)
                clear_output()
            acc_test = np.mean((network(X_test).reshape(-1).detach().numpy().round() != Y_test.numpy()))
            print(num_hidden_layer,"|",hidden_layer_size,"|",L_train[-1],"|",acc_test)

    print("___________________RELU Activation Func_______________________")
    print("Hidden_layer| Width |Train_error| Test_error")

    for num_hidden_layer in num_hidden_layers:
        for hidden_layer_size in width:
            network = NN(weight_init_scheme="He",num_hidden_layers=num_hidden_layer, num_hidden_size=hidden_layer_size,act_func="RELU",input_feature=input_shape)
            L_train = []
            L_test = []
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            for epoch in range(25):
                for x_batch, y_batch in minibatch(X_train, Y_train, size=32, shuffle=True):
                    training(network, criterion, optimizer, x_batch, y_batch)

                network.eval()
                acc_train = np.mean((network(X_train).reshape(-1).detach().numpy().round() != Y_train.numpy()))
                L_train.append(acc_train)
                clear_output()
            acc_test = np.mean((network(X_test).reshape(-1).detach().numpy().round() != Y_test.numpy()))
            print(num_hidden_layer,"|",hidden_layer_size,"|",L_train[-1],"|",acc_test)

if __name__ == "__main__":
    main()

