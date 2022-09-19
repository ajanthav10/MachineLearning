''''
Author :Ajantha Varadharaaj
Date :sep 17th 2022

This code contains implementation of ID3 algorithm for car and bank dataset'''
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


'''TODO ---how to split the data as X and Y label and which ones belong to same cols ?'''

train_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/DecisionTree/car/train.csv",names=['buying','maint','doors','persons','lug_boot','safety,'label'])
test_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/DecisionTree/car/test.csv",names=['buying','maint','doors','persons','lug_boot','safety,'label'])



class select_attribute:
    def __init__(self):

    def entropy(target_col):
        ''' calculate the entropy 
        input :- Y (label) and train data
        output :- H = -(p_+)log_2(p_+) - (p_-)log_2(p_-)
        '''
        elements, counts = np.unique(target_col,return_counts = True)
        H = 0
        for i in range(len(elements)):    
            p_i = counts[i]/np.sum(counts)
            H += -p_i*np.log2(p_i)    
        return H


    def infogain()
        return IG
    def majority_error():
        return ME
    def giniindex():
        return GI

class tree:
    def __init__(self):

