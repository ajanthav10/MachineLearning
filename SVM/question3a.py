import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize



'''DATA PREPROCESSING
INPUT- training and testing data
added bais col to X feature and in Y label convert "0" to "-1"
'''
train_data=pd.read_csv('./bank-note/train.csv')
X_train = train_data.iloc[:,:-1].to_numpy()
bais_train=np.ones((871, 1))
X_train = np.append(X_train, bais_train, axis=1)
y_train=train_data.iloc[:,4].to_numpy()
y_train[y_train == 0] = -1
train_data=np.column_stack((X_train,y_train))
#_____Preprocessing test data
test_data=pd.read_csv('./bank-note/test.csv')
X_test = test_data.iloc[:,:-1].to_numpy()
bais_test=np.ones((499, 1))
X_test = np.append(X_test, bais_test, axis=1)
y_test=test_data.iloc[:,4].to_numpy()
y_test[y_test == 0] = -1
test_data=np.column_stack((X_test,y_test))

train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) - 1


def sign_func(x):
    y = 0
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def error(x, y):
    sum = 0
    length = len(x)
    for i in range(length):
        if x[i] != y[i]:
            sum = sum + 1
    return sum / length


def predict(weight, data):
    p_list = []
    for i in range(len(data)):
        p_list.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], weight)))
    label = [i[-1] for i in data]
    return error(p_list, label)


def gaussian_kernel(s_1, s_2, gamma):
    dim = len(s_1) - 1
    s_11 = s_1[0:dim]
    s_22 = s_2[0:dim]
    diff = [s_11[i] - s_22[i] for i in range(dim)]
    kernel = math.e ** (-np.linalg.norm(diff) ** 2 / gamma)
    return kernel


def kernel():
    k_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            k_hat_t[i, j] = (train_data[i][-1]) * (train_data[j][-1]) * np.inner(train_data[i][0:dim_s],
                                                                                 train_data[j][0:dim_s])
    return k_hat_t


def objective_function(x):
    tp1 = x.dot(K_hat_)
    tp2 = tp1.dot(x)
    tp3 = -1 * sum(x)
    return 0.5 * tp2 + tp3


def constraint(x):
    return np.inner(x, np.asarray(y_pred))


def svm_dual(C):
    bd = (0, C)
    bds = tuple([bd for i in range(train_len)])
    x0 = np.zeros(train_len)
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)
    return [sol.fun, sol.x]


def recover_weights(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * train_data[i][-1] * np.asarray(train_data[i][0: dim_s]))
    return sum(ll)


def main(C):
    [sol_f, sol_x] = svm_dual(C)
    weight = recover_weights(sol_x)
    train_error = predict(weight, train_data)
    test_error = predict(weight, test_data)
    print('weight=', weight)
    print('train err=', train_error)
    print('test err=', test_error)


K_hat_ = kernel()
y_pred = [row[-1] for row in train_data]
C = [100 / 873, 500 / 873, 700 / 873]
for i in C:
    main(i)