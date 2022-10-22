import numpy as np 
from new_ID3 import * #importing the Decision tree
import pandas as pd
import matplotlib.pyplot as plt # to plot the figure
import argparse
#runns for a very long time 
class Bagging:
    """
    creating bagging ensemble learning algorithm by picking m' samples with replacement
    """
    def __init__(self, m, T, data, numerical=False, key=None): 
        self.samples = m
        self.T = T
        self.data = data
        self.error = np.zeros((T,))
        self.alpha = np.zeros((T,))
        self.treesInit = []
        self.numerical = numerical
        self.key = key
        
    def sample_draw(self):
        idx = np.random.choice(np.arange(len(self.data)), self.samples)
        return self.data.iloc[list(idx)]
    
    
    def plurality(self, tree_init, t, numerical=False):
        obj_tree = applyTree(self.data, tree_init, 
                             numerical=numerical)
        h_t, total_error = predict_ID3(obj_tree)
        self.error[t] = total_error
        self.alpha[t] = 0.5*np.log((1 - total_error)/total_error)
      

    def predict(self, data):
        predicts = []
        for t in range(self.T):
            applyInit = applyTree(data, self.treesInit[t],
                                  numerical=self.numerical)
            predict_ID3(applyInit)
            predicts.append(applyInit.predict)   
        return predicts  
   
def train_bagging(self):
    #self._bagging_loop()
    for t in range(self.T):
        bag = self.sample_draw()
        tree_init = decisionTree(bag, numerical=self.numerical)           
        self.treesInit.append(tree_init)
        run_ID3(tree_init)
        self.plurality(tree_init, t, numerical=self.numerical)
    
    
def apply_bagging(self, data):
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
        print("running bagging for bank dataset..")
        cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'y']
        train = pd.read_csv('bank/train.csv', names=cols)
        test = pd.read_csv('bank/test.csv', names=cols)
    elif dataset=='credit':
        print("running baggingbank for credit dataset")
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
    obj_bagging = Bagging(m, T, train, numerical=True, key=key)
    train_bagging(obj_bagging)
    # run_bagging_parallel(bagInit)

    err_bag_train = apply_bagging(obj_bagging, train)
    
    err_bag_test = apply_bagging(obj_bagging, test)
    f,(ax1) = plt.subplots(figsize=(10,5))
    ax1.plot(err_bag_train, 'blue')
    ax1.plot(err_bag_test, 'red')  
    ax1.legend(['train', 'test'])
    ax1.set_title('bagging')
    ax1.set_xlabel('Number of Trees', fontsize=18)
    ax1.set_ylabel('Error ', fontsize=16)
    f.savefig('bagging.png')


if __name__ == "__main__":
    main()