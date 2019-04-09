# Gang Wei wg0502@nu.edu
# Keras Case Study
# Multi-classifier of mnist dataset '0-9'
# Demo

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 5000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number,28*28)
    x_test = x_test.reshape(x_test.shape[0],28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train,y_train),(x_test,y_test)



#print(y_train.shape)


(x_train,y_train),(x_test,y_test) = load_data()
model = Sequential()
model.add(Dense(input_dim=28*28,units=200,activation='relu'))

for i in range(0,10):
    model.add(Dense(units=200,activation='relu'))

model.add(Dense(units=10,activation='softmax'))
model.compile(loss='mse',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=100)
result = model.evaluate(x_test,y_test)

print('\nThe accuracy is:'+
      str(100*float(result[1]))[:4]
      +'%')
