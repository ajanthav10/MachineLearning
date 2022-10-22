'''
This code contains the implementation of Decision Tree ID3 algo for UNi of utah ML assignment. It was implemented by me to keep it as generic as possible
Date :17th sept 2022
References from Online : https://www.youtube.com/watch?v=K5QlpAqOtTE
https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f'''
#importing lib required
import numpy as np
import pandas as pd
from pprint import pprint
import csv


def H(feature_col):
    ''' calculate the entropy 
        input :- list of indices for a given attribute
        output :- H = -(p_+)log_2(p_+) - (p_-)log_2(p_-)
    '''
    elements, counts = np.unique(feature_col,return_counts = True)
    H_val = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        H_val+= -p_i*np.log2(p_i)    
    
    return H_val

def ME(feature_col): 
    '''calculate the majority error 
        input :- list of indices for a given attribute
        output :- ME = 1-(majority/sum_of_label)
    '''
    elements, counts = np.unique(feature_col,return_counts= True)
    m_err = 1-counts[np.argmax(counts)]/np.sum(counts)
    
    return m_err

def GI(feature_col):
    ''' calculate the entropy 
        input :- list of indices for a given attribute
        output :- GI # lazy to type the formula :P
    '''
    elements, counts = np.unique(feature_col,return_counts= True)
    GI=1
    
    for i in range(len(elements)):
        GI-=(counts[i]/np.sum(counts))**2
    
    return GI
    

def IG_H(data,attribute,label_name):
    '''calculating IG using Entropy
    input - total Entropy and indices of given attribute
    output - inforgain
    '''
    #total entropy is calculated using the whole dataset and all the labels
    total_entropy = H(data[label_name])
    values, counts = np.unique(data[attribute],return_counts=True)
    # getting the values of each attributes and calc entropy of each
    weighted_entropy = 0 
    for i in range(len(values)):
        #getting subdata as pd.dataframe and calc entropy for the same 
        sub_data= data[data[attribute]==values[i]]
        weighted_entropy += counts[i]/np.sum(counts)*H(sub_data[label_name])

    IG = total_entropy - weighted_entropy

    return IG

def IG_ME(data,attribute,label_name):
    '''calculating IG using majority error
    input - total majority error and indices of given attribute
    output - inforgain
    '''
    total_m_err = ME(data[label_name])
    values, counts = np.unique(data[attribute],return_counts=True)
    
    weighted_m_err = 0 
    for i in range(len(values)):
        sub_data= data[data[attribute]==values[i]]
        weighted_m_err += counts[i]/np.sum(counts)*ME(sub_data[label_name])
    
    IG = total_m_err - weighted_m_err
    print(IG)
    return IG
    
def IG_GI(data,attribute,label_name):

    total_GI = GI(data[label_name])
    values, counts = np.unique(data[attribute],return_counts=True)
    
    weighted_GI = 0 
    for i in range(len(values)):
        sub_data= data[data[attribute]==values[i]]
        weighted_GI += counts[i]/np.sum(counts)*GI(sub_data[label_name])
        
    IG = total_GI - weighted_GI

    return IG 

  
def Common_label(data, label):
    '''to find the most common label among the dataset . Might be required when no of features =0 or 
        when all examples have same label
        input :- dataset which is pd.dataframe 
        output:- most_common_category (str) - as label are categorical'''
    feature, f_count = np.unique(data[label],return_counts= True)
    #X_features and attr_counts - np array with same size 
    # attr_counts represents the count of each element in X_features
    #using argmax to get the index of label which is most repeated/common
    common_label = feature[np.argmax(f_count)]
    
    return common_label
    
def ID3_depth_entropy(depth, data,X_features,label):
    '''construct DT using Entropy as measure to get information gain
    Need to calc total entropy and entropy for each feature then subdata for the root node.
    input:data- pd dataframe, attributes- list and x_features - pd series , depth - to be max as no of attributes 
    output: create tree and store it in dict TODO - any other DS possible and how to draw the DT
    '''
    #Step 1 :- according to ID3, if all the attributes have same label then return leaf node with the label
    common_label=Common_label(data,label)
    
    #step 2 : if attributes empty return a leaf node
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
  
    elif len(X_features)==0:
        return common_label

    
    else:
       
        for f in X_features:
            item_values=[ IG_H(data,f,label) for f in X_features]
        # obtaning te highest ig value attribute 
        best_attribute_index = np.argmax(item_values)
        best_attribute = X_features[best_attribute_index]
        # storing the attribute and its vau in nested dict for creating a Decision tree
        tree = {best_attribute:{}}
        for value in np.unique(data[best_attribute]):
            # now when the value of best_split_attribute is equal to each value then create subdata 
            # subdata - pd dataframe 
            sub_data = data[data[best_attribute]== value]
            # get the most common label for the subdata 
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_entropy(depth-1,sub_data,X_features,label)
                tree[best_attribute][value]=subtree
        return tree

def ID3_depth_ME(depth, data,X_features,label):
    common_label=Common_label(data,label)
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
    elif len(X_features)==0:
        return common_label 
    else:    
        for f in X_features:
            item_values=[ IG_ME(data,f,label) for f in X_features]
        
        
        #print(item_values)
        best_attribute_index = np.argmax(item_values)
        best_attribute = X_features[best_attribute_index]
        
        print(best_attribute)
        tree = {best_attribute:{}}

        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_ME(depth-1,sub_data,X_features,label)
                tree[best_attribute][value]=subtree
        return tree

    
def ID3_depth_GI(depth, data,X_features,label="class"):
    common_label=Common_label(data,label)
  
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
        
    elif len(X_features)==0:
        return common_label

    else:
        for f in X_features:
            item_values=[ IG_GI(data,f,label) for f in X_features]

        best_attribute_index = np.argmax(item_values)
        best_attribute = X_features[best_attribute_index]
        
        tree = {best_attribute:{}}

        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_GI(depth-1,sub_data,X_features,label)
                tree[best_attribute][value]=subtree
        return tree

def predict(query,tree,default = 1):    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                #first lookup
                answer = tree[key][query[key]] 
            except:
                return default
  
            answer = tree[key][query[key]]
            #walking the tree 
            if isinstance(answer,dict):
                #implementing recursive func
                return predict(query,answer)
            else:
                return answer

def test(data,label,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    return np.sum(predicted["predicted"] == label)/len(data)

def main():    
    # begin with Input processing
    # step 1 --load the csv into memory and store it to a data structure
    # DS used is pandas dataframe as it is easier to access 
    data=pd.read_csv('./car/train.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])
    test_data=pd.read_csv('./car/test.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])
    # the provided dataset has no names for the features/X_features naming it from desc-txt from zip file 
    # naming the label as label
    #step 2 split dataset into X and Y for computation purpose where x- features Y- label
    #y - categorical label 
    # storing the name of all features in list X_features
    
    X_features= ['buying','maint','doors','persons','lug_boot','safety']
    Training_Label= data['labels']
    Training_Data= data[X_features]

    Test_Label= test_data['labels']
    Test_Data= test_data[X_features]

    
    print("Information Gain")
    print("--------------------------------------------------")
    for i in range(6):
        tree=ID3_depth_entropy(i+1, data, X_features,"labels")
        training_err=1- test(data,Training_Label,tree)
        testing_err =1- test(test_data,Test_Label,tree)
        print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
        
    print("Majority Error")
    print("--------------------------------------------------")
    #depth=6
    tree=ID3_depth_ME(6, data, X_features,"labels")
    training_err=1- test(data,Training_Label,tree)
    testing_err =1- test(test_data,Test_Label,tree)
    print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
    print("Gini Index")
    print("--------------------------------------------------")
    for i in range(6):
        tree=ID3_depth_GI(i+1, data, X_features,"labels")
        training_err=1- test(data,Training_Label,tree)
        testing_err =1- test(test_data,Test_Label,tree)
        print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
    
if __name__ == "__main__":
    main()