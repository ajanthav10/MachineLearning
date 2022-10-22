'''
This code contains the implementation of Decision Tree ID3 algo for UNi of utah ML assignment. It was implemented by me to keep it as generic as possible
Date :30th sept 2022
References from Online : https://www.youtube.com/watch?v=K5QlpAqOtTE
https://python-course.eu/machine-learning/decision-trees-in-python.php
https://automaticaddison.com/iterative-dichotomiser-3-id3-algorithm-from-scratch/
https://medium.com/@lope.ai/decision-trees-from-scratch-using-id3-python-coding-it-up-6b79e3458de4'''
#importing lib required
import numpy as np

class Node:
    '''creating the root node and child node'''
    def __init__(self):
        self.attribute = None 
        self.next = None 
        self.children = None 
        self.valName = None 
        self.depth = 0 # at root depth is 0
        self.leaf = False 
        
class decisionTree:
    '''Creating the decision tree'''
    def __init__(self, df, method='entropy', depth=None, randTieBreak=True, 
                 numerical=False, weights=None): 
        self.attributes = np.array(df.iloc[:,:-1]) 
        self.attrNames = np.array(df.columns[:-1]) # features
        self.labels = np.array(df.iloc[:,-1]) # y labels
        self.label_list = list(set(self.labels)) #getting values of unique labels
        self.node = None # node
        self.numerical = numerical #for numerical and categorical data
        self.media = None
        self.numerical_idx = []
        self.randPick = randTieBreak #when IG of attr is same randtie break to get best_attr
        self.gainMethod = self.H#using entropy to get information gain
        #depth and weights are specified 
        if depth is None:
            self.depthLimit = len(self.attrNames)
        else:
            self.depthLimit = depth
            
        if weights is None:
            self.weights = np.ones((len(self.attributes),))/len(self.attributes)    
        else:
            self.weights = weights
        
        
    def H(self, idx):
        '''Calculates the entropy of an attribute'''
        labels = self.labels[idx] 
        weights = self.weights[idx]
        val_count = np.zeros((len(self.label_list),))
        for i, lab in enumerate(self.label_list):
            setidx = labels == lab
            val_count[i] = sum(weights[setidx])
                        
        p = val_count/sum(val_count) + 1e-10
        H = -(sum(p*np.log2(p)))

        return H    
    
    def IG(self, idx, attrID):
        '''Calculates the information gain'''
        infoGain = self.gainMethod(idx) #calculating the total entropy of the dataset
        attrVals = self.attributes[idx,attrID]
        w = sum(self.weights[idx])
        attrSet = list(set(attrVals)) 
        infoGainAttr = 0
        for value in attrSet:
            idxloc = list(np.where(attrVals == value)[0])
            attridx = list(idx[idxloc])
            attr_w = sum(self.weights[attridx])
            gainSubset = self.gainMethod(attridx)
            infoGainAttr += (attr_w/w)*gainSubset
        infoGain -= infoGainAttr
        return infoGain
    
    def _getNextAttr(self, idx, attrNames):
        '''calc IG of single attr''' 
        attrIDs = [i for i in range(len(self.attrNames)) if self.attrNames[i] in attrNames]
        attrInfoGain = [self.IG(idx, attrID) for attrID in attrIDs]
        maxGain = np.array(attrInfoGain) == max(attrInfoGain)
        if self.randPick and (sum(maxGain) != 1):
            randidx = np.random.choice(np.flatnonzero(maxGain))
            attridx = attrInfoGain.index(attrInfoGain[randidx])
            bestAttr = attrNames[attridx]
            bestAttridx = attrIDs[attridx]
        else:
            attr_idx = attrInfoGain.index(max(attrInfoGain))
            bestAttr = attrNames[attr_idx]
            bestAttridx = attrIDs[attr_idx]
        
        return bestAttr, bestAttridx
    
    def boolconv(self, idx, attrNames):
        attrIDs = list(range(len(self.attrNames)))
        attrID_type = self.attributes[0,attrIDs]
        numAttrID = [ID for i, ID in enumerate(attrIDs) if isinstance(attrID_type[i], (float,int))]
        self.media = np.zeros((len(attrNames),))
        for attr in numAttrID:
            media = np.median(self.attributes[idx, attr])
            self.media[attr] = media
            self.numerical_idx.append(attr)
            self.attributes[:,attr] = self.attributes[:,attr] > media
            
    
    def _ID3Rec(self, idx, attrNames, node, prevMax=None):
        
        if not node: 
            node = Node()
        labelsAttr = self.labels[idx] 
        if len(set(labelsAttr)) == 1:
            node.attribute = self.labels[idx[0]]
            node.leaf = True
            return node
        
        elif len(idx) == 0:
            
            node.attribute = prevMax
            node.leaf = True
            return node
        elif len(attrNames) == 0:
            node.attribute = prevMax
            node.leaf = True
            return node
        w_attr = self.weights[idx]
        l_value, count = np.unique(labelsAttr, return_inverse=True)
        w_idx = [count == i for i in range(len(l_value))]
        w_max = np.array([sum(w_attr[i]) for i in w_idx]).argmax()
        sub_common = l_value[w_max]
        if node.depth == self.depthLimit:
            node.attribute = sub_common
            node.leaf = True
            return node
        bestAttr, bestAttridx = self._getNextAttr(idx, attrNames)
        node.attribute = bestAttr
        node.children = []
        chosenAttrVals = list(set(self.attributes[:, bestAttridx]))
        for val in chosenAttrVals: 
            child = Node()
            child.depth = node.depth + 1
            child.valName = val
            node.children.append(child)
            idxloc = np.where(self.attributes[idx, bestAttridx] == val)[0]
            childidx = idx[list(idxloc)]
            if len(childidx)==0:               
                child.next = self._ID3Rec(childidx, [], child, 
                                          prevMax=sub_common)
            else:
                if len(attrNames) != 0 and bestAttr in attrNames:
                    nextAttrs = attrNames
                    idx2del = (bestAttr == nextAttrs).argmax()
                    nextAttrs = np.delete(nextAttrs, idx2del)
                child.next = self._ID3Rec(childidx, nextAttrs, child, 
                                          prevMax=sub_common)
        
        self.tree = node
        return node # return tree root
  
class applyTree:
    def __init__(self, test, treeInit, numerical=False, 
                 weights=None):
        self.root = treeInit.tree 
        self.attrNames = np.array(test.columns)
        self.startTest = np.array(test.iloc[:,:-1]) # test DataFrame
        self.startLabels = np.array(test.iloc[:,-1]) # test labels
        self.numerical = numerical # if data is numerical or not
        if self.numerical:
            self.media = treeInit.media
            self.numerical_idx = treeInit.numerical_idx
        if weights is None:
            self.weights = np.ones((len(self.startTest),))/len(self.startTest)    
        else:
            self.weights = weights
        self.predict = []
            
    def _applyLoops(self, currNode, subset, sublab):
        errs = np.zeros((len(sublab),))
        for row in range(len(subset)):
            leaf = False
            node = currNode
            while not leaf:
                split = node.attribute
                s_idx = (self.attrNames==split).argmax()
                nextval = subset[row,s_idx]
                for child in node.children:
                    if child.valName == nextval:
                        node = child
                        break
                if node.leaf == True:
                    leaf = True
            errs[row] = sublab[row] != node.attribute
            self.predict.append(node.attribute)
        return errs
    
def run_ID3(self):
    idx = np.arange(len(self.attributes)) # all indices of attributes
    #print(idx.shape)
    attrNames = self.attrNames.copy() # attribute names
    #print(attrNames)
    if self.numerical:
        self.boolconv(idx, attrNames)
    self.node = self._ID3Rec(idx, attrNames, self.node) # get the root node using id3
    return self.node

def predict_ID3(self):
    currNode = self.root # root node of tree
    allTest = self.startTest # test data
    allLabels = self.startLabels # test labels
    if self.numerical:
        for idx in self.numerical_idx:
            allTest[:,idx] = allTest[:,idx].copy() > self.media[idx]
    errs = self._applyLoops(currNode, allTest, allLabels) # running apply tree
    weighted_errs = errs*self.weights
    total_err = np.sum(weighted_errs)
    h_t = errs == 0
    return h_t, total_err
