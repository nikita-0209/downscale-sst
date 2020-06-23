import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import argparse
import cv2
import numpy.ma as ma

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l', '--file_name_low', action='store', dest='file_name_low', default='./data/' ,
                    help='Low Resolution Images')

    parser.add_argument('-k', '--file_name_high', action='store', dest='file_name_high', default='./data/' ,
                    help='High Resolution images')
   
    values = parser.parse_args()

"""# Create Dataset"""

import h5py

file_name_low=values.file_name_low
file_name_high=values.file_name_high

data= h5py.File(file_name_low, 'r')
X=data['samples']

data= h5py.File(file_name_high, 'r')
Y=data['samples']

"""# Train-Test-Validation Split"""

from sklearn.model_selection import train_test_split

trainX, valX, trainY, valY = train_test_split(X[:], Y[:], test_size=0.33, random_state=42)

#Model

from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Conv2D, Masking, add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import PReLU, Activation
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow import image
from tensorflow import float32, constant_initializer

input_shape=(33,33,1)
batch_size=64
patch_size=(33,33)

def PSNRLoss(y_true, y_pred):
    y_true=K.clip(y_true, 0, 1)
    y_pred=K.clip(y_pred, 0, 1)
    im1 = image.convert_image_dtype(y_true, float32)
    im2 = image.convert_image_dtype(y_pred, float32)
    return -10. * K.log(K.mean(K.square(im2 - im1))) / K.log(10.)
    

input_img = Input(shape=input_shape)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
res_img = model

output_img = add([res_img, input_img])

model = Model(input_img, output_img)

model.compile(loss=MeanSquaredError(),
              optimizer=Adam(learning_rate=0.001),
              # metrics=['accuracy'])
              metrics=[PSNRLoss])


import random
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import math, int64, cast

def gen(features, labels, batch_size, patch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, patch_size[0], patch_size[1], 1))
 batch_labels = np.zeros((batch_size, patch_size[0], patch_size[1], 1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choices(range(len(labels)),k=batch_size)
     batch_features = features[index]
     batch_labels = labels[index]
   yield batch_features, batch_labels

path="ckpts/"+"cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=path, 
    verbose=1, 
    save_weights_only=True,
    monitor='loss', 
    save_best_only=True)

callbacks_list=[cp_callback]

#model.load_weights('/home/ubuntu/enhancer/home/ubuntu/enhancer/srcnn_ckpts/cp-0018.ckpt')

history=model.fit_generator(gen(trainX, trainY, batch_size, patch_size), 
                    steps_per_epoch=trainY.shape[0]//batch_size, 
                    epochs=100, 
                    validation_data=gen(valX, valY, batch_size, patch_size),
                    validation_steps=valY.shape[0]//batch_size,
                    callbacks=callbacks_list
                    )