'''
This code contains the implementation of Decision Tree ID3 algo for Uni of utah ML assignment. 
Date :17th sept 2022'''

#importing lib required
import pandas as pd
import numpy as np
#from pprint import pprint # might use to print final decision tree
#from tabulate import tabulate
#import matplotlib.pyplot as plt
# will i need matplotlib ??--> TODO


# begin with Input processing
# step 1 --load the csv into memory and store it to a data structure
# DS used is pandas dataframe as it is easier to access 

train_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/Decision Tree/bank/train.csv",names=['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','category'])
test_data=pd.read_csv("/home/ajantha/Desktop/Repos/MachineLearning/Decision Tree/bank/test.csv",names=['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','category'])
# the provided dataset has no names for the features/attributes naming it from desc-txt from zip file 
# naming the label as label

#step 2 split dataset into X and Y for computation purpose where x- features Y- label
#y - categorical label 
# storing the name of all features in list X_features
X_features=['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','Label'] # X featues 
Training_Y=train_data['category'] # it is pd.series 
Training_X=train_data[X_features] # pd.dataframe
#similarly splitting the X and Y from test dataset
Test_Y=test_data['category']
Test_X=test_data[X_features] 
#print(type(Training_X))
#print(type(Training_Y))
#print(Training_X.shape)
#print(Training_Y.shape)


class car:


    #def __init__(self):#TODO add the depth what to do about user defined depth 

        #TODO any declartions of instances 
          
    def entropy(self,feature_col): # TODO add in the args name correctly 
        '''calculate the entropy 
        input :- list of indices for a given attribute
        output :- H = -(p_+)log_2(p_+) - (p_-)log_2(p_-)'''
        elements, counts = np.unique(target_col,return_counts = True)
        H = 0
        for i in range(len(elements)):    
            p_i = counts[i]/np.sum(counts)
            #print(here)
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
    def IG_H(self,dataset,X_features): 
        #total entropy is calculated using the whole dataset and all the labels 
        total_entropy=entropy(dataset[category])
        attr_values,attr_counts = np.unique(dataset[X_features],return_counts=True)
        # getting the values of each attributes and calc entropy of each
        attr_entropy= 0 
        for i in range(len(attr_values)):
            #getting subdata as pd.dataframe and calc entropy for the same 
            sub_data= dataset[dataset[X_features]==attr_values[i]]
            attr_Hv += attr_counts[i]/np.sum(attr_counts)*entropy(sub_data[category])
        IG_H = total_entropy - attr_entropy

        return IG_H

    def IG_ME(self,dataset,attributes):     
        total_ME = majority_error(dataset[category])
        attr_values,attr_counts = np.unique(dataset[X_features],return_counts=True)        
        attr_majorityerror= 0 
        for i in range(len(attr_counts)):
            sub_data= dataset[dataset[X_features]==attr_counts[i]]
            attr_majorityerror += attr_counts[i]/np.sum(attr_counts)*majority_error(sub_data[category])
        
        IG_ME = total_ME - attr_majorityerror

        return IG_ME
   
    def IG_GI(self,dataset,attributes):
        total_GI = Gini_index(dataset[category])
        attr_values,attr_counts = np.unique(dataset[X_features],return_counts=True)        
        attr_GI= 0 
        for i in range(len(attr_counts)):
            sub_data= dataset[dataset[X_features]==attr_counts[i]]
            attr_GI += attr_counts[i]/np.sum(attr_counts)*Gini_index(sub_data[category])
        
        IG_GI = total_GI - attr_GI

        return IG_GI


    def most_common_category(self,dataset,category):# TODO args need Y label 
        '''to find the most common label among the dataset . Might be required when no of features =0 or 
        when all examples have same label
        input :- dataset which is pd.dataframe 
        output:- most_common_category (str) - as label are categorical'''
        X_features,attr_counts=np.unique(dataset[category],return_counts=True)
        #X_features and attr_counts - np array with same size 
        # attr_counts represents the count of each element in X_features
        #using argmax to get the index of label which is most repeated/common 
        most_common_category=X_features[np.argmax(attr_counts)]
        return most_common_category



    def ID3_H(self,depth,data,category,X_features):
        '''construct DT using Entropy as measure to get information gain
        Need to calc total entropy and entropy for each feature then subdata for the root node.
        input:data- pd dataframe, attributes- list and x_features - pd series , depth - to be max as no of attributes 
        output: create tree and store it in dict TODO - any other DS possible and how to draw the DT
        '''
        #Step 1 :- according to ID3, if all the attributes have same label then return leaf node with the label
        if len(np.unique(dataset[category]))<=1:
            common_label=most_common_category(dataset,category)
            return common_label # TODO np.unique(data[category])[0]  check common label or this 
        #step 2 : if attributes empty return a leaf node 
        elif len(X_features)==0:       
             return common_label
        else:
            for features in X_features:
                X_featuresIG=[ IG_H(dataset,features,category) for features in X_features]
            best_split_attribute = X_featuresIG[np.argmax(attr_values)]
            # obtaning te highest ig value attribute 
            node = {best_split_attribute:{}} # storing the attribute and its vau in nested dict for creating a Decision tree

            for v in np.unique(dataset[best_split_attribute]):
                # now when the value of best_split_attribute is equal to each value then create subdata 
                # subdata - pd dataframe 
                sub_dataset= dataset[dataset[best_split_attribute]== v]
                # get the most common label for the subdata 
                sub_dataset_commonlabel= most_common_category(sub_dataset,label)

            if len(sub_dataset)==0 or depth==1:
                node[best_split_attribute][v] = sub_dataset_common_label

            else:
                subtree=ID3_H(depth-1,sub_dataset,X_features,category)
                node[best_split_attribute][v]=subtree
        return node


    def ID3_ME(self,depth,data,category,X_features):
        if len(np.unique(dataset[category]))<=1:
            common_label=most_common_category(dataset,category)
            return common_label # TODO np.unique(data[category])[0]  check common label or this 
        #step 2 : if attributes empty return a leaf node 
        elif len(X_features)==0:       
             return common_label
        else:
            for features in X_features:
                X_featuresIG=[ IG_ME(dataset,features,category) for features in X_features]
            best_split_attribute = X_featuresIG[np.argmax(attr_values)]
            # obtaning te highest ig value attribute 
            node = {best_split_attribute:{}} # storing the attribute and its vau in nested dict for creating a Decision tree

            for v in np.unique(dataset[best_split_attribute]):
                # now when the value of best_split_attribute is equal to each value then create subdata 
                # subdata - pd dataframe 
                sub_dataset= dataset[dataset[best_split_attribute]== v]
                # get the most common label for the subdata 
                sub_dataset_commonlabel= most_common_category(sub_dataset,label)

            if len(sub_dataset)==0 or depth==1:
                node[best_split_attribute][v] = sub_dataset_common_label

            else:
                subtree=ID3_ME(depth-1,sub_dataset,X_features,category)
                node[best_split_attribute][v]=subtree
        return node   

        
    def ID3_GI(self,depth,data,category,X_features):
        if len(np.unique(dataset[category]))<=1:
            common_label=most_common_category(dataset,category)
            return common_label # TODO np.unique(data[category])[0]  check common label or this 
        #step 2 : if attributes empty return a leaf node 
        elif len(X_features)==0:       
             return common_label
        else:
            for features in X_features:
                X_featuresIG=[ IG_GI(dataset,features,category) for features in X_features]
            best_split_attribute = X_featuresIG[np.argmax(attr_values)]
            # obtaning te highest ig value attribute 
            node = {best_split_attribute:{}} # storing the attribute and its vau in nested dict for creating a Decision tree

            for v in np.unique(dataset[best_split_attribute]):
                # now when the value of best_split_attribute is equal to each value then create subdata 
                # subdata - pd dataframe 
                sub_dataset= dataset[dataset[best_split_attribute]== v]
                # get the most common label for the subdata 
                sub_dataset_commonlabel= most_common_category(sub_dataset,label)

            if len(sub_dataset)==0 or depth==1:
                node[best_split_attribute][v] = sub_dataset_common_label

            else:
                subtree=ID3_GI(depth-1,sub_dataset,X_features,category)
                node[best_split_attribute][v]=subtree
        return node
#Predict
def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
               result = node[key][query[key]]
            except:
               return default

            result = node[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
def test(dataset,category,node):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(test_data)):
        predicted.loc[i,"predicted"] = predict(queries[i],node,1.0) 
        
    return np.sum(predicted["predicted"] == category)/len(dataset)

def print(self):

    for i in range(6):
        tree=ID3_H(i+1, dataset, Attributes,"labels")
        train_acc= test(data,Training_Label,tree)
        test_acc = test(test_data,Test_Label,tree)
        print("===========Accuracy of DT with Information Gain==========================================")
        print("Depth is",i+1 ,'Training acc is', train_acc, 'and testing_acc is' , test_acc)

    for i in range(6):
        tree=ID3_MI(i+1, dataset, Attributes,"labels")
        train_acc= test(data,Training_Label,tree)
        test_acc = test(test_data,Test_Label,tree)
        print("===========Accuracy of DT with Information Gain==========================================")
        print("Depth is",i+1 ,'Training acc is', train_acc ,'and testing_acc is' , test_acc)

    for i in range(6):
        tree=ID3_GI(i+1, dataset, Attributes,"labels")
        train_acc= test(data,Training_Label,tree)
        test_acc = test(test_data,Test_Label,tree)
        print("===========Accuracy of DT with Information Gain==========================================")
        print("Depth is",i+1, 'Training acc is', train_acc, 'and testing_acc is' , test_acc)

    return 0