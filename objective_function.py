import numpy as np
from savepkl import *
from classifier import cnn

def obj_fun1(soln):
    X_train=load('X_train')
    X_test=load('X_test')
    y_train=load('Y_train')
    y_test=load('Y_test')

    # Feature selection
    soln = np.round(soln)
    X_train=X_train[:,np.where(soln==1)[0]]
    X_test = X_test[:, np.where(soln == 1)[0]]
    pred, met, _ = cnn(X_train, y_train, X_test, y_test)
    fit = 1/met[0]

    return fit

def objfun2(soln):
    X_train = load('cur_X_train')
    X_test = load('cur_X_test')
    y_train = load('cur_y_train')
    y_test = load('cur_y_test')

    pred, met = cnn_w(X_train, y_train, X_test, y_test, soln)
    fit = 1 / met[0]
    return fit