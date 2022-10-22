# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:09:17 2021

@author: Yoshihiro Obata
"""
import numpy as np
from ID3 import decisionTree, run_ID3, applyTree, apply_ID3
import matplotlib.pyplot as plt
import time
import pandas as pd

def var(data, m):
    s_sq = (1/(len(data.T) - 1))*np.sum((data.T - m)**2)
    return s_sq

# %% Change T here:
T = 50 # for AdaBoost
T_bag = 50 # for Bagging and Random Forest
key = {'no':-1, 'yes':1}
if T == 50:
    print('Running a smaller number of trees just for example of functionality. Results are created using T=50')

# %%
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']
train = pd.read_csv('bank/train.csv', names=cols)
test = pd.read_csv('bank/test.csv', names=cols)

# %% boosting
tic = time.perf_counter()

adaInit = AdaBoost(train, T, key=key)
run_adaBoost(adaInit)

err_AdaTrain = apply_adaBoost(adaInit, train)
stump_err_train = adaInit.errs
toc = time.perf_counter()
print('Time to train and apply AdaBoost with {} trees was {:0.4f} seconds.\n'.format(T, toc-tic))

tic = time.perf_counter()
err_AdaTest = apply_adaBoost(adaInit, test)
stump_err_test = adaInit.errs
toc = time.perf_counter()
print('Time to apply AdaBoost to test set with {} trees was {:0.4f} seconds.\n'.format(T, toc-tic))

# %% plotting adaboost
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(np.arange(T), err_AdaTrain, 'k', label='Training', linewidth=2)
ax.plot(np.arange(T), err_AdaTest, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
if T == 500:
    plt.savefig('P2AdaBoost500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2AdaBoost.png', dpi=150, bbox_inches='tight')
    
fig,ax = plt.subplots(figsize=(10,5))
plt.plot(np.arange(T), stump_err_train, 'k', label='Training', linewidth=2)
plt.plot(np.arange(T), stump_err_test, 'r--', label='Test', linewidth=2)
ax.tick_params(labelsize = 16, size = 10, width = 2)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(2)
plt.xlim([0,adaInit.T])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Stump Error', fontsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.grid(True)
if T == 500:
    plt.savefig('P2AdaBoostStump500.png', dpi=150, bbox_inches='tight')
else:
    plt.savefig('P2AdaBoostStump.png', dpi=150, bbox_inches='tight')

class AdaBoost:
    """
    
    """
    def __init__(self, train, T, depth=1, key=None):
        self.data = train
        self.labels = np.array(train.iloc[:,-1])
        self.D_init = np.ones((len(train),))/len(train)
        self.T = T
        self.depth = depth # decision stump
        self.learners_init = []
        self.alpha = np.zeros((T,))
        self.errs = np.zeros((T,))
        self.errs_w = np.zeros((T,))
        self.key = key
    
    def _calc_vote(self, stump_init, t, D, numerical=False):
        err_init = applyTree(self.data, stump_init, weights=D, 
                             numerical=numerical)
        h_t, total_err = apply_ID3(err_init)

        # total_err = 1 - total_acc
        if total_err > 0.5:
            print(f'Total error was {total_err}, which is greater than 50%')
        self.errs_w[t] = total_err
        self.errs[t] = 1 - sum(h_t)/len(h_t)
        self.alpha[t] = 0.5*np.log((1 - total_err)/(total_err))
        
        return h_t
    
    def _progress(self, t):
        percent = np.round(100*t/self.T)
        if len(self.learners_init) != self.T:        
            print(f'{percent}% done. {t} trees created...')
        else:
            print(f'{percent}% done. {t} trees applied...')
    
    def _map2posneg(self, h_t):
        return h_t*2 - 1
    
    def _update_weights(self, D, t, h_t):
        yh = self._map2posneg(h_t)
        D_tp1 = D*np.exp(-self.alpha[t]*yh)
        Z_t = np.sum(D_tp1)
        D_tp1 /= Z_t
        return D_tp1
    
    def _AdaLoop(self, D):
        print('Starting training...')
        for t in range(self.T):
            if (t)%np.round(self.T/10) == 0:
                self._progress(t)
            stump_init = decisionTree(self.data, numerical=True, 
                                      depth=self.depth, weights=D)
            run_ID3(stump_init)
            self.learners_init.append(stump_init)
            h_t = self._calc_vote(stump_init, t, D, numerical=True)
            Dtp1 = self._update_weights(D, t, h_t)
            D = Dtp1
        print('Done training\n')
            
    def _apply_AdaBoost(self, data):
        # h_t = []
        predicts = []
        for t in range(self.T):
            if (t)%np.round(self.T/10) == 0:
                self._progress(t)
            tree_init = self.learners_init[t]
            
            applyInit = applyTree(data, tree_init, 
                               weights=tree_init.weights, 
                               numerical=True)
            apply_ID3(applyInit)
            predicts.append(applyInit.predict)
        print('Done applying \n')
        return predicts
    
def run_adaBoost(self):
    D_init = self.D_init.copy()
    self._AdaLoop(D_init)
    
def apply_adaBoost(self, data):
    h_t = np.array(self._apply_AdaBoost(data))
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



# %% Change T here:
