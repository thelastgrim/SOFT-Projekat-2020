
import  tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Activation, MaxPooling2D, Lambda
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import  tqdm
import cv2
from glob import glob
import os
import csv
from tensorflow.keras.models import model_from_json
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.utils import  plot_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import  tqdm

print(">> IMAGE CLASSIFICATION MODEL CREATION")

print()
print(">> STEP#1 IMAGE PRE-PROCESSING")

#!git clone https://gitlab.com/thelastgrim/ori-projekat.git
#------------------

data = pd.read_csv('datasets_FINAL\\train\\chest_xray_metadata.csv')


#---------------


img_width = 224
img_height = 224

X = []
#for i in tqdm(range(1500)):
for i in tqdm(range(data.shape[0])):
  lent = len(data['X_ray_image_name'][i].split('.'))
  ext="."+data['X_ray_image_name'][i].split('.')[lent-1]
  path = 'datasets_FINAL\\train\\merged\\'+ data['X_ray_image_name'][i].replace(ext, "")+'_resized_merged'+ext
  #path = '/content/dataset-x-ray/data/'+ data['X_ray_image_name'][i]
  try:
    img = image.load_img(path, target_size=(img_width,img_height,3))
    img = image.img_to_array(img)
    img = img/255.0
    X.append(img)
  except:

      print("NEMA SSLIKE ", path)

X = np.array(X)

Y = []
y = data.drop(columns=['Unnamed: 0', 'X_ray_image_name', 'Label', 'Label_2_Virus_category'])[
    0:data.shape[0]] 


y = y.to_numpy()
for i in tqdm(range(data.shape[0])):

  if (pd.isnull(y[i])):
    label = [1, 0, 0]
  elif (y[i] == 'Virus'):
    label = [0, 1, 0]
  elif (y[i] == 'bacteria'):
    label = [0, 0, 1]
  Y.append(label)


Y = np.array(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state =0, test_size=0.15)

opt = Adam(lr=0.00001)


model = Sequential()
model.add(Conv2D(filters=96, input_shape=(350,350,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(4096, input_shape=(350*350*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()

  
opt = Adam(learning_rate=0.000007)


model.compile(loss='categorical_crossentropy',             
            optimizer=optimizers.SGD(lr=0.0001),
            metrics=['accuracy'])
epochs = 10



history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

number_of_epochs_it_ran = len(history.history['loss'])

model.load_weights('.mdl_wts.hdf5')



test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2) # ovo je sa tensorflow sajta
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)



print(">> STEP#4 VISUALIZING ACCURACY AND LOSS")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('ACCURACY')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.title('LOSS')

plt.show()

