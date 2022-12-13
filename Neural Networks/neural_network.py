#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Preprocessing
train = pd.read_csv('./bank-note/train.csv')
test = pd.read_csv('./bank-note/test.csv')

# helper functions 
def sigmoid(x)->float:
    return 1/(1 + np.exp(-x))

def diff_sig(x)->float:
    return sigmoid(x)*(1 - sigmoid(x))   

def shuffle(train, labels)->np.ndarray:
    '''
    random shuffle of the train dataset 
    ip - Train dataset
    op - randomnly shuffled X features and Y Labels
    '''
    i = np.random.choice(np.arange(len(train)), len(train),
                               replace=False)
    rand_X = train[i,:]
    rand_Y = labels[i]
    return rand_X, rand_Y

def learning_rate_schd(lr0, d_init, t)->float:
    gamma= lr0/(1 + (lr0*t)/d_init)
    return gamma

class layer:
    '''To represent a single layer in a neural network
     fin - number of input features
     num - number of the layer
     fout - number of output features 
     layer_type - str to check if it is hidden or output layer
    '''
    def __init__(self, num, fin, fout, layer_type='hidden', winit='Gauss'):
        self.layer_type = layer_type
        self.layer_num = num 
        self.nodes = np.zeros(fout) # Zero array with shape as same
        if self.layer_type == 'hidden':
            self.s = np.zeros(fout)
        else:
            pass
        
        if winit == 'Gauss':
            np.random.seed(15)
            self.ws = np.random.randn(fin + 1, fout)                   
        else:
            self.ws = np.zeros((fin + 1, fout))
        self.dnodes = np.zeros_like(self.nodes)
        self.dws = np.zeros_like(self.ws)
                
class Neural_network:
    '''To represent a feedfoward neural network and to compute backprop using sgd
    '''
    def __init__(self, w, data, T=5, lr0=0.01, d_init=0.005, d=3, winit='Gauss'):
        self.w = w # no of weight/nodes in the layer
        self.d = d # num of layers in the network
        self.T = T # training epochs set to 80 
        self.train = data.iloc[:,:-1].values
        self.labels = (data.iloc[:,-1].values)*2 - 1
        self.N = len(self.train) # num of training samples 
        self.lr0 = lr0 # learning rate set to 0.01
        self.d_init = d_init # set to 0.5 --found by hypertuning 
        # layer creation
        if (winit == 'Gauss') or (winit == 'Zero'):
            L_input = [layer(1, self.train.shape[1], w, layer_type='hidden', winit=winit)]
            L_hidden = [layer(i, w, w, layer_type='hidden', winit=winit) for i in range(2,d,1)]
            L_output = [layer(d, w, 1, layer_type='output', winit=winit)]
            self.layers = L_input + L_hidden + L_output
        else:
            L_hidden = [layer(i, w, w, layer_type='hidden', winit=winit[i-1]) for i in range(1,d,1)]
            L_output = [layer(d, w, 1, layer_type='output', winit=winit[-1])]
            self.layers = L_hidden + L_output
        
    def forward(self, attribute):
        '''Feedforward operation on the NN
        '''
        for layer in self.layers:
            #iterating over all the layers of the NN
            n = layer.layer_num
            if len(attribute.shape) == 1:
                attribute = attribute.reshape(1,-1)
            #input layer- data augumented with bais term 
            if n == 1:
                bais = np.ones((attribute.shape[0], 1))
                fin = np.append(bais, attribute, axis=1)
            else:
                # not input layer so previous output+ bais augumented 
                bais = np.ones((self.layers[n-2].nodes.shape[0], 1))
                fin = np.append(bais, self.layers[n-2].nodes, axis=1)
            #output of the current layer obtained by dot product
            temp = fin.dot(layer.ws)
            #  save as node if it is output layer 
            if layer.layer_type == 'output':
                layer.nodes = temp
            else:
                # pass through activation func if not output layer
                layer.s = temp
                layer.nodes = sigmoid(temp)
                
        return self.layers[2].nodes
    
    def backprop(self, attribute, Y):
        '''Backpropgation on the NN
        ip-features,label 
        op -none'''
        #creating reverse_layers starting from the output & moving towards the input 
        reverse_layers = np.flip(self.layers.copy())
        for layer in reverse_layers:
            n = layer.layer_num
            down_layer = self.d - (n - 1) 
            #layer ==output so gradient of the loss function with respect to the output 
            if layer.layer_type == 'output':
                down_nodes = np.append(1, reverse_layers[down_layer].nodes)
                
                dL = layer.nodes-Y
                layer.dnodes = dL
                layer.dws = dL* down_nodes     # gradients of the weights of all layers        
            elif layer.layer_type == 'hidden':
                #gradients wrt outputs of the downstream ,w and diff_sigmoid
                if layer.layer_num == 1:
                    down_nodes = np.append(1, attribute)
                else:
                    down_nodes = np.append(1, reverse_layers[down_layer].nodes)
                     
                up_layer = self.d - (n + 1) 
                if up_layer == 0:
                    dUpward = reverse_layers[up_layer].dnodes
                    wsUp = reverse_layers[up_layer].ws
                    dSig = diff_sig(layer.s)
                    dN = dUpward.reshape(1,-1).dot(wsUp.T)
                else:
                    dUpward = reverse_layers[up_layer].dnodes[:,1:] 
                    wsUp = reverse_layers[up_layer].ws 
                    dSig = diff_sig(layer.s)
                    dSigUp = diff_sig(reverse_layers[up_layer].s) 
                    dN = (dUpward*dSigUp).reshape(1,-1).dot(wsUp.T)
                
                layer.dnodes = dN
                dNodes = dN[:,1:]
                #dot product of the gradient wrt op and ip layer 
                layer.dws = down_nodes.reshape(-1,1).dot((dNodes*dSig).reshape(1,-1))

    def prediction(self, test)->float:
        '''Predicitng the labels on test using the trained neural network
        ip - test 
        op - error 
        incorrect = predict != Y_test
        error = sum(incorrect)/len(incorrect)
        '''
        #splitting the test dataset into features and labels
        X_test = test.iloc[:,:-1].values
        Y_test = test.iloc[:, -1].values.reshape(-1,1)
        # converting so that the labels are -1 and 1 instead of 1 and 0
        Y_test = Y_test*2 - 1 

        predict = (self.forward(X_test) >= 0)*2 - 1
        error=np.sum(predict!=Y_test) / Y_test.shape[0]
        
        return error
    
    def training(self, train, test):
        '''Trains the dataset by iterating over a number of epochs 
        ip- dataset
        op- error
        '''
        train_error = np.zeros(self.T)
        test_error = np.zeros(self.T)
        for t in range(self.T):
            gamma = learning_rate_schd(self.lr0, self.d_init, t)
            #shuffle the dataset over training example
            data, labels = shuffle(self.train, self.labels)
            #perform feedforward and backpropgation on each sample
            for i in range(self.N):
                X = data[i,:]
                Y = labels[i]
                self.forward(X)
                self.backprop(X, Y)
                for layer in self.layers:
                    if layer.layer_type == 'output':
                        #updating the weight of layer using SGD 
                        # gamma calc using learning_rate_schd()
                        layer.ws -= gamma*layer.dws.T
                    else:
                        layer.ws -= gamma*layer.dws
            #predicting error via calling prediction() 
            error = self.prediction(train)
            train_error[t] = error
            error = self.prediction(test)
            test_error[t] = error
                          
        return self, train_error, test_error
  

def main():
    epochs = 100
    lr0 = 0.01 # values of lr and d after testing for a while
    d_init = 0.5
    widths = [5, 10, 25, 50, 100]
    Train_error = []
    Test_error = []
    Train_error_2c =[]
    Test_error_2c =[]

    print('___________________________________________________Question 2b_____________________________________________________________')
    for w in widths:
        Neural_net = Neural_network(w, train, T=epochs, lr0=lr0, d_init=d_init, winit='Gauss')
        trained_net, train_err, test_err = Neural_net.training(train, test)
        Train_error.append(train_err)
        Test_error.append(test_err)
    #create a new list where each element is the last element from the corresponding sub-list in Train_error and Test_error
    trerr_val=[method[-1] for method in Train_error]
    testerr_val=[method[-1] for method in Test_error]
    for i in range(5):
        print("Width =",widths[i],"Training Error",trerr_val[i],"Testing Error",testerr_val[i])
    

    print("____________________________________________________Question 2c_________________________________________________________")
    for w in widths:
        nnet = Neural_network(w, train, T=epochs, lr0=lr0, d_init=d_init, winit='Zero')
        trained_net, train_err, test_err = nnet.training(train, test)
        Train_error_2c.append(train_err)
        Test_error_2c.append(test_err)


    trerr_val1=[method[-1] for method in Train_error_2c]
    testerr_val1=[method[-1] for method in Test_error_2c]
    for i in range(5):
        print("Width =",widths[i],"Training Error",trerr_val1[i],"Testing Error",testerr_val1[i])



    fig, ax = plt.subplots(figsize=(12,6))
    epoch = np.arange(epochs)
    c = ['yellow', 'red', 'blue', 'green', 'black']
    for i in range(len(Train_error)):
        plt.plot(epoch, Train_error[i],color=c[i])
    for i in range(len(Train_error_2c)):
        plt.plot(epoch, Train_error_2c[i],linestyle='--', color=c[i])
    plt.xlim([0,100])
    plt.ylim([0,0.1])
    width = ['w=5',
            'w=10',
            'w=25',
            'w=50',
            'w=100']
    plt.legend(width, fontsize=7)
    plt.xlabel('Epoch', fontsize=8)
    plt.ylabel('Error', fontsize=8)
    plt.title("Training Error for Gauss Vs Zero Weight Init")
    plt.savefig('Training_error.png')

    fig, ax = plt.subplots(figsize=(12,6))
    epoch = np.arange(epochs)
    c = ['yellow', 'red', 'blue', 'green', 'black']
    for i in range(len(Test_error)):
        plt.plot(epoch, Test_error[i],color=c[i])
    for i in range(len(Test_error_2c)):
        plt.plot(epoch, Test_error_2c[i],linestyle='--', color=c[i])
    plt.xlim([0,100])
    plt.ylim([0,0.1])
    width = ['w=5',
            'w=10',
            'w=25',
            'w=50',
            'w=100']
    plt.legend(width, fontsize=7)
    plt.xlabel('Epoch', fontsize=8)
    plt.ylabel('Error', fontsize=8)
    plt.title("Testing Error for Gauss Vs Zero Weight Init")
    plt.savefig('Testing_error.png')

if __name__ == "__main__":
    main()
