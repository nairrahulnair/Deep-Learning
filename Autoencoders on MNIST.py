import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf



from tensorflow import keras
import keras
print(tf.keras.__version__)

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import np_utils



###  Importing the dataset

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
y_train
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=100)

image=x_train[55, :].reshape((28,28))                                                  
plt.imshow(image)
plt.show()

im_rows=28
im_cols=28
batch_size=512
im_shape=(im_rows,im_cols,1)##no of rows, cols and channel is 1 as its a grayscale image

x_train=x_train.reshape(x_train.shape[0], *im_shape)
x_test=x_test.reshape(x_test.shape[0], *im_shape)
x_val=x_val.reshape(x_val.shape[0], *im_shape)
y_val=y_val.reshape(y_val.shape[0], *im_shape)

model1=Sequential()

encoder=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='selu'),
    keras.layers.Dense(30, activation='selu')])


decoder=keras.models.Sequential([
    keras.layers.Dense(100, activation='selu', input_shape=[30]),
    keras.layers.Dense(28*28, activation='sigmoid'),
    keras.layers.Reshape([28,28])])

ae=keras.models.Sequential([encoder, decoder])

ae.compile(optimizer='nadam', loss='binary_crossentropy')

ae.summary()

history=ae.fit(x_train, x_train, epochs=100, batch_size=256)

y_pred=ae.predict(x_test)

########### Visualizing the predicted and reconstruced image from row 55

###Selecting a date point from test
image1=x_test[210, :].reshape((28,28))                                                  
plt.imshow(image1)
plt.show()

### rescontructing the above data point
plt.figure(figsize=(10,10))

plt.imshow(y_pred[210,])





