

from os import listdir

import keras
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from PIL import Image

batch_size = 20
num_classes = 10
epochs = 7

# input image dimensions
img_rows, img_cols = 64, 64

#%% load labels
# the data, split between train and test sets
dataDir = "./digitsDataset/"
files = listdir(dataDir)
files.sort()

targets = pd.read_csv(dataDir+"_0.txt", delimiter="\n")
numItems = targets.shape[0]
inputs = np.empty((numItems, img_cols, img_rows, 3))
#%% read images and pad them

for i, file in enumerate(files):
    read = np.asarray(Image.open(dataDir + file))
    print(str(i)+":"+str(read.shape)+file)
    inputs[i] = np.pad(read, (((img_cols-read.shape[0])//2, (img_cols-read.shape[0])//2),
                              ((img_rows-read.shape[1])//2, (img_rows-read.shape[1]+1)//2),
                              (0, 0)),
    'constant')
    # limit
    if i == numItems-1:
        break

#%%
# 80/20 split
x_train = inputs[:int(targets.shape[0]*0.8)]
y_train = targets[:int(targets.shape[0]*0.8)]
x_test = inputs[int(targets.shape[0]*0.8):]
y_test = targets[int(targets.shape[0]*0.8):]

"""if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)"""
input_shape = (img_rows,img_rows,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
np.argmax(model.predict(x_test),axis=1)