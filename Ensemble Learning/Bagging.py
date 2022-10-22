import numpy as np 
from new_ID3 import * #importing the Decision tree
import pandas as pd
import matplotlib.pyplot as plt # to plot the figure
import argparse

class Bagging:
    """
    creating bagging ensemble learning algorithm by picking m' samples with replacement
    """
    def __init__(self, m, T, data, numerical=False, key=None, 
                 randForest=False, Gsize=6, verbose=True, 
                 global_override=False, globaldf = None):
        self.mp = m
        self.T = T
        self.data = data
        self.errs = np.zeros((T,))
        self.alpha = np.zeros((T,))
        self.treesInit = []
        self.numerical = numerical
        self.key = key
        if m < 0.5*len(data):
            self.small_sub = True
            self.globaldf = data
        else:
            self.small_sub = False
        self.randForest = randForest
        self.Gsize = Gsize
        self.verbose = verbose
        if global_override:
            self.small_sub = True
            self.globaldf = globaldf
        
    def sample_draw(self):
        idx = np.random.choice(np.arange(len(self.data)), self.mp)
        return self.data.iloc[list(idx)]
    
    
    def _calc_vote(self, tree_init, t, numerical=False):
        err_init = applyTree(self.data, tree_init, 
                             numerical=numerical)
        h_t, total_err = apply_ID3(err_init)
        self.errs[t] = total_err
        self.alpha[t] = 0.5*np.log((1 - total_err)/total_err)
        
    def _bagging_loop(self):
        for t in range(self.T):
            bootstrap = self.sample_draw()
            if self.small_sub:
                tree_init = decisionTree(bootstrap, numerical=self.numerical,
                                         small_sub=self.small_sub,
                                         globaldf=self.globaldf,
                                         randForest=self.randForest,
                                         Gsize=self.Gsize)
            else:
                tree_init = decisionTree(bootstrap, numerical=self.numerical,
                                         randForest=self.randForest,
                                         Gsize=self.Gsize)
            self.treesInit.append(tree_init)
            run_ID3(tree_init)
            self._calc_vote(tree_init, t, numerical=self.numerical)
        

    def _map2posneg(self, h, key):
        h_mapped = [key[i] for i in h]
        return np.array(h_mapped)    

    def _apply_bagging_loop(self, data):
        predicts = []
        for t in range(self.T):
            applyInit = applyTree(data, self.treesInit[t],
                                  numerical=self.numerical)
            apply_ID3(applyInit)
            predicts.append(applyInit.predict)   
        return predicts  
   
def run_bagging(self):
    self._bagging_loop()
    
    
def apply_bagging(self, data):
    h_t = np.array(self._apply_bagging_loop(data))
    h_t = (np.vectorize(self.key.get)(h_t)).T
    alpha = np.array(self.alpha)
    alpha_h = alpha*h_t
    err = np.zeros((self.T,))
    true_lab = data.iloc[:,-1]
    true_lab = np.array([self.key[true_lab[i]] for i in range(len(data))])
    for t in range(self.T):
        H = np.sum(alpha_h[:,:t+1], axis=1) > 0
        H = H*2 - 1
        err[t] = sum(H != true_lab)/len(true_lab)
    return err     

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
        data = pd.read_excel('credit/default of credit card clients.xls', names=cols,usecols="B:Y")
        #print(data.shape)
        #print(data)
        train=data.iloc[0:24000,:]
        #print(train)
        #print(train.shape)
        test=data.iloc[24000:30001,:]
        #print(test.shape)
    T =500  
    key = {'no': -1, 'yes': 1}
    m = 1000
    bagInit = Bagging(m, T, train, numerical=True, key=key)
    run_bagging(bagInit)
    # run_bagging_parallel(bagInit)

    err_bag_train = apply_bagging(bagInit, train)
    
    err_bag_test = apply_bagging(bagInit, test)
    f,(ax1) = plt.subplots(figsize=(10,5))
    ax1.plot(err_bag_train, 'b')
    ax1.plot(err_bag_test, 'r')  
    ax1.legend(['train', 'test'])
    ax1.set_title('bagging')
    ax1.set_xlabel('Number of Trees', fontsize=18)
    ax1.set_ylabel('Error ', fontsize=16)
    f.savefig('bagging.png')


if __name__ == "__main__":
    main()