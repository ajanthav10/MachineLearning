''''
Author :Ajantha Varadharaaj
Date :sep 17th 2022

This code contains implementation of ID3 algorithm for car and bank dataset'''
import pandas as pd
import numpy as np
from pprint import pprint

def entropy(X):
    H =0
    feature,count=np.unique(X,return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

    return H
