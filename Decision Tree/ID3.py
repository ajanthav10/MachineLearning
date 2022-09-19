''''
Author :Ajantha Varadharaaj
Date :sep 17th 2022

This code contains implementation of ID3 algorithm for car and bank dataset'''
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

##target_cols=indices
'''TODO ---how to split the data as X and Y label and which ones belong to same cols ? And might delete this as required '''

train_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/DecisionTree/car/train.csv",names=['buying','maint','doors','persons','lug_boot','safety,'label'])
test_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/DecisionTree/car/test.csv",names=['buying','maint','doors','persons','lug_boot','safety,'label'])



class bestsplit_attribute:
    def __init__(self,method="entropy"):#TODO add the depth what to do about user defined depth 

        #TODO any declartions of instances 
        self.attributes = np.array(df.iloc[:,:-1]) #to be here or in IG def ??
        self.labels=np.array(dataset.iloc[:,-1])  
        self.attrNames = np.array(df.columns[:-1]) # does it takes only name of feature only ???

       
    def entropy(self,idx):
        ''' calculate the entropy 
        input :- list of indices for a given attribute
        output :- H = -(p_+)log_2(p_+) - (p_-)log_2(p_-)
        '''
        elements, counts = np.unique(target_col,return_counts = True)
        H = 0
        for i in range(len(elements)):    
            p_i = counts[i]/np.sum(counts)
            H += -p_i*np.log2(p_i)    
        return H
    def majority_error(self,idx):
        '''calculate the majority error 
        input :- list of indices for a given attribute
        output :- ME = 1-(majority/sum_of_label)
        '''
        ME=1
        elements,counts = np.unique(target_col,return_counts = True)
        ME-=counts[np.argmax(counts)]/np.sum(counts)
        return ME
    def Gini_index(self,idx):
        ''' calculate the entropy 
        input :- list of indices for a given attribute
        output :- GI
        '''
        elements, counts = np.unique(target_col,return_counts = True)
        GI = 1
        for i in range(len(elements)):    
            GI-=pow((counts[i]/np.sum(counts)),2)
        return GI
    def IG(self,dataset,attributes)
        ## TODO common label ??
        '''calculates the IG of a single attribute
        input: all the indices required from dataset and also sub-dataset
        output: IGvalue ??'''
         # takes only the last column ie Y label
        H_total=entropy(label)
        for i in range(len(attrNames))
            attr_elements,atr_counts=np.unique(np.array(:,attributes),return_counts=True)
        weighted_entropy = 0 
        for i in range(len(values)):
            sub_data= data[data[attribute]==values[i]]
            weighted_entropy += counts[i]/np.sum(counts)*entropy(sub_data[label_name])

        InfoGain = total_entropy - weighted_entropy

        infoGain = self.gainMethod(idx) # total entropy
        # list of indices of an attribute
        attrVals = self.attributes[idx,attrID]
        #print(attrVals)
        w = sum(self.weights[idx])
        attrSet = list(set(attrVals)) # list of unique vals for attrVals
        #print(attrSet)
        infoGainAttr = 0
        # uses info gain method to calc info gain for the attr
        for value in range(len(attrSet)):
            idxloc = np.where(attrVals == attrSet[value])[0]
            #print(idxloc)
            attridx = list(idx[list(idxloc)])
            attr_w = sum(self.weights[attridx])
            gainSubset = self.gainMethod(attridx)
            infoGainAttr += attr_w/w*gainSubset
        infoGain -= infoGainAttr
        return IG
class tree:
    def __init__(self):


    def infogain()
        return IG
    def majority_error():
        return ME
    def giniindex():
        return GI

class tree:
    def __init__(self):

