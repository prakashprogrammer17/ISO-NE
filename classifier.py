from keras import Sequential
from keras.layers import(Conv2D,MaxPooling2D,Dense,Dropout,Flatten)
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

from savepkl import *
import numpy as np
from confusion_matrix import *

X_train=load('X_train')
X_test=load('X_test')
Y_train=load('Y_train')
Y_test=load('Y_test')
def cnn(X_train,Y_train,X_test,Y_test):

    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1,1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1,1))

    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=200, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    #y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict,confu_matrix(Y_test, y_predict) #confu_matrix(Y_train, y_predict_train,

def ann(X_train,Y_train,X_test,Y_test):

    model = Sequential()
    model.add(Dense(20, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=100, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    #y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict, confu_matrix(Y_test, y_predict) #confu_matrix(Y_train, y_predict_train,1)
def BiLSTM(X_train,Y_train,X_test,Y_test):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1 ))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the Bi-LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=100, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    return y_predict, confu_matrix(Y_test, y_predict)
def pro_classifier():
    pred1=cnn(X_train,Y_train,X_test,Y_test)
    #save('pred1',pred1)

    pred2=ann(X_train,Y_train,X_test,Y_test)
    #save('pred2',pred2)

    pred3=BiLSTM(X_train,Y_train,X_test,Y_test)


pro_classifier()



