
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

def start():

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

  opt = Adam(lr=3e-4)

  base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5', input_shape=(224, 224, 3))
  model1 = models.Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
  #model1.compile()
  #model = models.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
  #model1.output.set_shape(3)
  #model1.output = layers.Dense(3, activation='sigmoid')
  #plot_model(model1)
  model = Sequential()
  model.add(model1.get_layer(index= 0))
  model.add(model1.get_layer(index= 1))
  model.add(model1.get_layer(index= 2))
  model.add(model1.get_layer(index= 3))
  model.add(model1.get_layer(index= 4))
  model.add(model1.get_layer(index= 5))
  model.add(model1.get_layer(index= 6))
  model.add(model1.get_layer(index= 7))
  model.add(model1.get_layer(index= 8))
  model.add(model1.get_layer(index= 9))
  model.add(model1.get_layer(index= 10))
  model.add(model1.get_layer(index= 11))
  model.add(model1.get_layer(index= 12))
  model.add(model1.get_layer(index= 13))
  model.add(model1.get_layer(index= 14))
  model.add(model1.get_layer(index= 15))
  model.add(model1.get_layer(index= 16))
  #model.add(model1.get_layer(index= 17))


  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dense(3, activation='softmax')) 
  model.compile(loss='categorical_crossentropy',             
              optimizer= opt,#optimizers.SGD(lr=0.0001),
              metrics=['accuracy'])
  epochs = 1



  history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

  number_of_epochs_it_ran = len(history.history['loss'])

  #model.load_weights('.mdl_wts.hdf5')


  '''
  score = model.evaluate(X_test, y_test, verbose = 1)
  print("Test score: ", score[0])
  print("Test accuracy: ", score[1])
  '''

  test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2) # ovo je sa tensorflow sajta
  print("Test loss: ", test_loss)
  print("Test accuracy: ", test_acc)



  #nodel.acc

  print(">> STEP#4 VISUALIZING ACCURACY AND LOSS")
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']


  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  #epochs_range = number_of_epochs_it_ran

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

  #TESTTTTTTTTTT

  test_data = pd.read_csv('datasets_FINAL\\test\\00_test.csv')
#--------------------------
  Y_test_data = []
  y_test_data = test_data.drop(columns=['Unnamed: 0', 'X_ray_image_name', 'Label', 'Label_2_Virus_category'])[0:test_data.shape[0]]
  # print(y)
  y_test_data = y_test_data.to_numpy()
  # y.shape
  for i in tqdm(range(len(y_test_data))):
    if (pd.isnull(y_test_data[i])):
      label = [1, 0, 0]
    elif (y_test_data[i] == 'Virus'):
      label = [0, 1, 0]
    elif (y_test_data[i] == 'bacteria'):
      label = [0, 0, 1]
    Y_test_data.append(label)
  Y_test_data = np.array(Y_test_data)

  X_test_data = []
  for i in tqdm(range(test_data.shape[0])):

      path = 'datasets_FINAL\\test\\'+ str(test_data['X_ray_image_name'][i])
      klasa = test_data['Label_1_Virus_category'][i]
      img = image.load_img(path, target_size=(img_width,img_height,3))
      img = image.img_to_array(img)
      img = img/255.0
      X_test_data.append(img)
    
      print(len(X_test_data))
      print(len(Y_test_data))
        
  X_test_data = np.array(X_test_data)
  X_train, X_test, y_train, y_test = train_test_split(X_test_data, Y_test_data, random_state =0, test_size=0.15)


  test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2) 
  print("Test loss: ", test_loss)
  print("Test accuracy: ", test_acc)


   


