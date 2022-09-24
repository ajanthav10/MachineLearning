import numpy as np
import pandas as pd
from pprint import pprint
import csv

# Define gain methods
def H(feature_col):
    elements, counts = np.unique(feature_col,return_counts = True)
    H_val = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        H_val+= -p_i*np.log2(p_i)    
    
    return H_val

def ME(feature_col): 
    elements, counts = np.unique(feature_col,return_counts= True)
    m_err = 1-counts[np.argmax(counts)]/np.sum(counts)
    
    return m_err

def GI(feature_col):
    elements, counts = np.unique(feature_col,return_counts= True)
    GI=1
    
    for i in range(len(elements)):
        GI-=(counts[i]/np.sum(counts))**2
    
    return GI
    
#Compute Information gain and fidn the best attribute
def IG_H(data,attribute,label_name):


    total_entropy = H(data[label_name])
    values, counts = np.unique(data[attribute],return_counts=True)

    weighted_entropy = 0 
    for i in range(len(values)):
        sub_data= data[data[attribute]==values[i]]
        weighted_entropy += counts[i]/np.sum(counts)*H(sub_data[label_name])

    IG = total_entropy - weighted_entropy

    return IG

def IG_ME(data,attribute,label_name):
    total_m_err = ME(data[label_name])
    values, counts = np.unique(data[attribute],return_counts=True)
    
    weighted_m_err = 0 
    for i in range(len(values)):
        sub_data= data[data[attribute]==values[i]]
        weighted_m_err += counts[i]/np.sum(counts)*ME(sub_data[label_name])
    
    IG = total_m_err - weighted_m_err

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

# Find the common label which will be the leaf node.  
def Common_label(data, label):
    
    feature, f_count = np.unique(data[label],return_counts= True)
    common_label = feature[np.argmax(f_count)]
    
    return common_label
    

# ID3_depending on the depth
def ID3_depth_entropy(depth, data,Attributes,label):
    common_label=Common_label(data,label)
    

    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        for f in Attributes:
            item_values=[ IG_H(data,f,label) for f in Attributes]

        best_attribute_index = np.argmax(item_values)
        best_attribute = Attributes[best_attribute_index]
        
        tree = {best_attribute:{}}

        
        # grow a branch under the root node
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_entropy(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree

#ID3 with option to limit depth
def ID3_depth_ME(depth, data,Attributes,label):
    common_label=Common_label(data,label)
    

    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        for f in Attributes:
            item_values=[ IG_ME(data,f,label) for f in Attributes]

        best_attribute_index = np.argmax(item_values)
        best_attribute = Attributes[best_attribute_index]
        
        tree = {best_attribute:{}}

        
        # grow a branch under the root node
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_ME(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree

    #ID3 with option to limit depth
def ID3_depth_GI(depth, data,Attributes,label="class"):
    common_label=Common_label(data,label)
    

    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        for f in Attributes:
            item_values=[ IG_GI(data,f,label) for f in Attributes]

        best_attribute_index = np.argmax(item_values)
        best_attribute = Attributes[best_attribute_index]
        
        tree = {best_attribute:{}}

        
        # grow a branch under the root node
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_GI(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree

# Apply the ID3 and predict test dataset.

def predict(query,tree,default = 1):    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

def test(data,label,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    return np.sum(predicted["predicted"] == label)/len(data)



def main():    
    # import "car" data 
    # problem2. 
    data=pd.read_csv('./car/train.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])
    test_data=pd.read_csv('./car/test.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])

    # Split the data and its labels.
    Attributes= ['buying','maint','doors','persons','lug_boot','safety']
    Training_Label= data['labels']
    Training_Data= data[Attributes]

    Test_Label= test_data['labels']
    Test_Data= test_data[Attributes]

    
    print("Information Gain")
    print("--------------------------------------------------")
    for i in range(6):
        tree=ID3_depth_entropy(i+1, data, Attributes,"labels")
        training_err=1- test(data,Training_Label,tree)
        testing_err =1- test(test_data,Test_Label,tree)
        print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
        
    print("Majority Error")
    print("--------------------------------------------------")
    for i in range(6):
        tree=ID3_depth_ME(i+1, data, Attributes,"labels")
        training_err=1- test(data,Training_Label,tree)
        testing_err =1- test(test_data,Test_Label,tree)
        print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
    print("Gini Index")
    print("--------------------------------------------------")
    for i in range(6):
        tree=ID3_depth_GI(i+1, data, Attributes,"labels")
        training_err=1- test(data,Training_Label,tree)
        testing_err =1- test(test_data,Test_Label,tree)
        print("The depth =", i+1, "and training error = ","{:.3f}".format(training_err) ,"and testing error =" ,"{:.3f}".format(testing_err))
    
if __name__ == "__main__":
    main()