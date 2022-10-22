'''Created the Adaboost Algorithm and modified the DT 
# to enable it to take weight for calculating entropy'''
import numpy as np 
from new_ID3 import * #importing the Decision tree
import pandas as pd
import matplotlib.pyplot as plt # to plot the figure
import argparse

class AdaBoost:
    """
    Constructing the adaboost ensemble learning algorithm 
    with weights to be uniform distribution as 1/m

    """
    def __init__(self, train, T,  key=None):
        '''constructors to initialise the weights and depth 
        '''
        self.data = train
        self.labels = np.array(train.iloc[:,-1])# spliting the data only to have the labels 1 or 0
        self.weight = np.ones((len(train),))/len(train)# weights is 1/len(training ex)
        self.T = 500 # no of iterations 
        self.depth = 1 # decision stump
        self.weak_learner = []
        self.alpha = np.zeros((T,))
        self.error = np.zeros((T,))
        self.w_error= np.zeros((T,))
        self.key = key
    
    def plurality(self, stump_init, t:int, D:np.ndarray, numerical=False):
        '''calc the vote and find alpha_t to update the weight 
        '''
        #creating the object for applytree to apply Dt to test data
        obj_tree = applyTree(self.data, stump_init, weights=D, 
                             numerical=numerical)
        h_t, total_error = predict_ID3(obj_tree)
        self.w_error[t] = total_error
        self.error[t] = 1 - sum(h_t)/len(h_t)
        self.alpha[t] = 0.5*np.log((1 - total_error)/(total_error)) 
        return h_t
           
   
    def weight_update(self, D:np.ndarray, t, h_t:np.ndarray):
        y_i=h_t*2-1
        new_D_t = D*np.exp(-self.alpha[t]*y_i)
        Z_t = np.sum(new_D_t)
        return new_D_t/Z_t
                
    def predict(self, data:pd.DataFrame):
        predicts = []
        for t in range(self.T):
            tree_init = self.weak_learner[t]
                
            applyInit = applyTree(data, tree_init, 
                                weights=tree_init.weights, 
                                numerical=True)
            predict_ID3(applyInit)
            predicts.append(applyInit.predict)
        print('Finished Applying adaboost on Test set \n')
        return predicts
    
def train_adaboost(self):
    D = self.weight.copy()
    print('training the dataset for adaboost....')
    for t in range(self.T):
        decision_stump = decisionTree(self.data, numerical=True, 
                                        depth=self.depth, weights=D)
        run_ID3(decision_stump)
        self.weak_learner.append(decision_stump)
        h_t = self.plurality(decision_stump, t, D, numerical=True)
        new_D = self.weight_update(D, t, h_t)
        D = new_D   
    print('Adaboost training completed!\n')

   
    
def apply_adaBoost(self, data:pd.DataFrame):
    h_t = np.array(self.predict(data))
    h_t = (np.vectorize(self.key.get)(h_t)).T
    alpha = np.array(self.alpha)
    alpha_h = alpha*h_t
    error = np.zeros((self.T,))
    label = data.iloc[:,-1]
    label = np.array([self.key[label[i]] for i in range(len(data))])
    for t in range(self.T):
        H = np.sum(alpha_h[:,:t+1], axis=1) > 0
        H = H*2 - 1
        error[t] = sum(H != label)/len(label)
    return error


def main():
    '''creating the argument to take  dataset to be used for adaboost'''
    parser = argparse.ArgumentParser(description='Adaboost for both dataset')
    # Add an argument
    parser.add_argument('--dataset',choices=["credit", "bank"] , required=True, help='dataset for adaboost')
    args = parser.parse_args()
    dataset=args.dataset
    #print(args.dataset)
    if dataset=='bank':
        print("running adaboost for bank dataset..")
        cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'y']
        train = pd.read_csv('bank/train.csv', names=cols)
        test = pd.read_csv('bank/test.csv', names=cols)
    elif dataset=='credit':
        print("running adaboost for credit dataset")
        cols = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3',
                'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
                'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
                'PAY_AMT5','PAY_AMT6','default payment next month']					
        data = pd.read_csv('credit/credit.csv', names=cols,)
        #print(data.shape)
        #print(data)
        train=data.iloc[0:24000,:]
        print(train)
        #print(train.shape)
        test=data.iloc[24000:30001,:]
        #print(test.shape)
        
    T = 500  # for AdaBoost
    key = {'no': -1, 'yes': 1}
    #instantiating the class adaboost 
    obj_adaboost = AdaBoost(train, T, key=key)
    train_adaboost(obj_adaboost)
    err_AdaTrain = apply_adaBoost(obj_adaboost, train)
    #np.savetxt('ada_error.csv', [err_AdaTrain], delimiter=',', fmt='%d')
    stump_err_train = obj_adaboost.error  
    err_AdaTest = apply_adaBoost(obj_adaboost, test)
    stump_err_test = obj_adaboost.error
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(25, 10))
    ax1.plot(err_AdaTrain, color='blue')
    ax1.plot(err_AdaTest, color='green')  
    ax1.legend(['train', 'test'])
    ax1.set_title('First Figure')
    ax1.set_xlabel('Iteration', fontsize=8)
    ax1.set_ylabel('Error Rate', fontsize=8)
    
    ax2.plot(stump_err_train, color='blue')
    ax2.plot(stump_err_test, color='green')  
    ax2.legend(['train', 'test'])
    ax2.set_title('Second Figure')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('Error Rate', fontsize=14)
    f.savefig('adaboost.png') 
   

if __name__ == "__main__":
    main()

    

