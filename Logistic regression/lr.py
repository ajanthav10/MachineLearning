import pandas as pd
import numpy as np

'''DATA PREPROCESSING
INPUT- training and testing data
added bais col to X feature and in Y label convert "0" to "-1"'''
training_data=pd.read_csv('./bank-note/train.csv')
X_train = training_data.iloc[:,:-1].to_numpy()
bais_train=np.ones((871, 1))
X_train = np.append(X_train, bais_train, axis=1)
Y_train=training_data.iloc[:,4].to_numpy()
Y_train[Y_train == 0] = -1

#_____Preprocessing test data
testing_data=pd.read_csv('./bank-note/test.csv')
X_test = testing_data.iloc[:,:-1].to_numpy()
bais_test=np.ones((499, 1))
X_test = np.append(X_test, bais_test, axis=1)
Y_test=testing_data.iloc[:,4].to_numpy()
Y_test[Y_test == 0] = -1

def predict(x,y,w)->float:
    y_pred=np.matmul(x,w)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    error=np.sum(np.abs(y_pred - np.reshape(y,(-1,1)))) / 2 / y.shape[0]
    return error

def train_MAP(x,y)->np.ndarray:
    lr=0.01
    d=0.1
    epoch=100
    #gamma=0.1
    v=1
    num_sample = x.shape[0]
    dim = x.shape[1]
    w = np.zeros([1, dim])
    idx = np.arange(num_sample)
    for t in range(epoch):
        np.random.shuffle(idx)
        x = x[idx,:]
        y = y[idx]
        for i in range(num_sample):
            x_i = x[i,:].reshape([1, -1])
            tmp = y[i] * np.sum(np.multiply(w, x_i))
            g = - num_sample * y[i] * x_i / (1 + np.exp(tmp)) + w /v
            # print(g)
            lr = lr / (1 +lr /d * t)
            w = w - lr * g
    return w.reshape([-1,1])
    
def train_ML(x,y)->np.ndarray:
    lr=0.01
    d=0.1
    epoch=100
    #gamma=0.1
    v=1
    num_sample = x.shape[0]
    dim = x.shape[1]
    w = np.zeros([1, dim])
    idx = np.arange(num_sample)
    for t in range(epoch):
        np.random.shuffle(idx)
        x = x[idx,:]
        y = y[idx]
        for i in range(num_sample):
            tmp = y[i] * np.sum(np.multiply(w, x[i,:]))
            g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
            lr = lr / (1 + lr / d * t)
            w = w - lr * g
    return w.reshape([-1,1])

def main():
    v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    print("______________________________Question3a__________________________________")
    for v in v_list:
        #model.set_v(v)
        w= train_MAP(X_train, Y_train)
        train_error= predict (X_train,Y_train,w)
        test_error= predict (X_test,Y_test,w)
        print("Variance:",v,'train_error: ', train_error, ' test_error: ', test_error)
    print("______________________________Question3b__________________________________")
    for v in v_list:

        weight= train_ML(X_train, Y_train)
        train_error= predict (X_train,Y_train,weight)
        test_error= predict (X_test,Y_test,weight)
        print("Variance:",v,'train_error: ', train_error, ' test_error: ', test_error)


if __name__=="__main__":
    main()