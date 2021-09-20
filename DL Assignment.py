import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
%matplotlib inline



from tensorflow import keras
import keras
print(tf.keras.__version__)

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import np_utils


test=pd.read_csv('D:/NIT/Deep Learning/Deep learning assig 1/fashion-mnist_test.csv')
train=pd.read_csv('D:/NIT/Deep Learning/Deep learning assig 1/fashion-mnist_train.csv')




########Data Prep

test_data=np.array(test,dtype='float32')###above need to be converted to numpy arrays in f32 format
train_data=np.array(train,dtype='float32')

##slicing the data into separate labels and pixel file

x_test=test_data[:,1:]/255###selecting the columns with pixels from 1st col till 785th col
y_test=test_data[:,0]### selcting only the 0th col with just labels
 

x_train=train_data[:,1:]/255
y_train=train_data[:,0]

###one hot encoding the y variables

#y_train=np_utils.to_categorical(y_train,10)
#y_test=np_utils.to_categorical(y_test,10)

####train and validation sets

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
                                  
###Selecting a particular row and all the columns e.g 47th row and all the columns 
###in the below example and plotting the pixels to recreate the image
### reshaping it back to 28x28 2D array is important 
              
image=x_train[47, :].reshape((28,28))                                                  
plt.imshow(image)
plt.show()

#####Creating the model

im_rows=28
im_cols=28
batch_size=512
im_shape=(im_rows,im_cols,1)##no of rows, cols and channel is 1 as its a grayscale image

x_train=x_train.reshape(x_train.shape[0], *im_shape)
x_test=x_test.reshape(x_test.shape[0], *im_shape)
x_val=x_val.reshape(x_val.shape[0], *im_shape)

print('x_train shape', format(x_train.shape))
print('y_train shape', format(y_train.shape) )
print('x_test shape', format(x_test.shape) )
print('x_val shape', format(x_val.shape) )
print('y val shape', format(y_val.shape))


model_1=Sequential()

##### Simple Neural Network with single layer

#model_1.add(Conv2D(filters=32,kernel_size=3,input_shape=im_shape,activation='relu'))
model_1.add(Flatten(input_shape=(28,28)))
model_1.add(Dense(10, input_shape=im_shape, name='dense_layer_1', activation='softmax'))
model_1.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model_1.fit(x_train, y_train, batch_size=64, epochs=50)


### MLP model

model_2=Sequential()


model_2.add(Flatten(input_shape=(28,28)))
model_2.add(Dense(50, input_shape=im_shape, name='dense_layer_1', activation='relu'))
model_2.add(Dense(50,name='dense_layer_2', activation='relu'))
model_2.add(Dense(10, activation='softmax'))

model_2.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_2=model_2.fit(x_train, y_train, batch_size=512, epochs=40, validation_data=(x_val, y_val))

print(model_2.summary())

### Visualization of models

print(history_2.history.keys())

## Accuracy plot
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
plt.show()


## Loss plot

plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


###Convulutional Network

model_3=Sequential()

model_3.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape))
model_3.add(MaxPooling2D(pool_size=2))  
model_3.add(Dropout(0.2))

model_3.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model_3.add(MaxPooling2D(pool_size=2))
model_3.add(Flatten())

model_3.add(Dense(32, activation='relu'))
model_3.add(Dense(10, activation='softmax'))

model_3.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_3=model_3.fit(x_train, y_train, batch_size=256, epochs=30, validation_data=(x_val, y_val))
    

print(model_3.summary())
#### Visualization

## Model Accuracy

print(history_3.history.keys())

plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')
plt.show()


### Model Loss

plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')
plt.xlabel('val loss')

plt.legend(['train', 'test'], loc='lower right')
plt.show()



