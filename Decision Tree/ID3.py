'''
This code contains the implementation of Decision Tree ID3 algo for UNi of utah ML assignment. It was implemented by me to keep it as generic as possible
Date :21st sept 2022'''

#importing lib required
import pandas as pd
import numpy as np
from pprint import pprint # might use to print final decision tree
import matplotlib.pyplot as plt
# will i need matplotlib ??--> TODO


# begin with Input processing
# step 1 --load the csv into memory and store it to a data structure
# DS used is pandas dataframe as it is easier to access 

train_data=pd.read_csv("./car/train.csv",names=['buying','maint','doors','persons','lug_boot','safety','category'])
test_data=pd.read_csv("./car/test.csv",names=['buying','maint','doors','persons','lug_boot','safety','category'])
# the provided dataset has no names for the features/attributes naming it from desc-txt from zip file 
# naming the label as label

#step 2 split dataset into X and Y for computation purpose where x- features Y- label
#y - categorical label 
# storing the name of all features in list X_features
X_features=['buying','maint','doors','persons','lug_boot','safety'] # X featues 
Training_X=train_data['category'] # it is pd.series 
Training_Y=train_data[X_features] # pd.dataframe
#similarly splitting the X and Y from test dataset
Test_X=test_data['category']
Test_Y=test_data[X_features] 
print(type(Training_X))
print(type(Training_Y))
print(Training_X.shape)
print(Training_Y.shape)


class bestsplit_attribute:
    def __init__(self,method="entropy"):#TODO add the depth what to do about user defined depth 

        #TODO any declartions of instances 
          
    def entropy(self,feature_col): # TODO add in the args name correctly 
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
    def majority_error(self,feature_col): # TODO add in the args name correctly 
        '''calculate the majority error 
        input :- list of indices for a given attribute
        output :- ME = 1-(majority/sum_of_label)
        '''
        ME=1
        elements,counts = np.unique(target_col,return_counts = True)
        ME-=counts[np.argmax(counts)]/np.sum(counts)
        return ME
    def Gini_index(self,feature_col): # TODO add in the args name correctly 
        ''' calculate the entropy 
        input :- list of indices for a given attribute
        output :- GI # lazy to type the formula :P
        '''
        elements, counts = np.unique(target_col,return_counts = True)
        GI = 1
        for i in range(len(elements)):    
            GI-=pow((counts[i]/np.sum(counts)),2)
        return GI
    def IG_H(self,dataset,attributes)    

        return IG
    def IG_ME(self,dataset,attributes)          
        return IG_ME
    def IG_GI(self,dataset,attributes)          
        return IG_GI
    def most_common_category(self,data,category) # TODO args need Y label 
        '''to find the most common label among the dataset . Might be required when no of features =0 or 
        when all examples have same label
        input :- dataset which is pd.dataframe 
        output:- most_common_category (str) - as label are categorical'''
        X_features,attr_counts=np.unique(data[category],return_counts=True)
        #X_features and attr_counts - np array with same size 
        # attr_counts represents the count of each element in X_features
        #using argmax to get the index of label which is most repeated/common 
        most_common_category=X_features[np.argmax(attr_counts)]
        return most_common_category

    def ID3_H(self,depth,data,category,X_features)
        '''construct DT using Entropy as measure to get information gain
        Need to calc total entropy and entropy for each feature then subdata for the root node.
        input:data- pd dataframe, attributes- list and x_features - pd series , depth - to be max as no of attributes 
        output: create tree and store it in dict TODO - any other DS possible and how to draw the DT
        '''

        # step 1 calc the total entropy for whole dataset with respect to categorical label(Y)
        H_S=entropy(data[categorical])

        #Step 2 :- according to ID3, if all the attributes have same label then return leaf node with the label
        if len(np.unique(data[category]))<=1:
            return common_label # TODO np.unique(data[category])[0]  check common label or this 
        #step 3 : if attributes empty return a leaf node 
        else if len(X_features)==0:
            return common_label
        else:
            for features in X_features

        #
        return node

    def ID3_ME()
        return node
    
    def ID3_GI()
        return node