import  tensorflow as tf
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
import Segmentation.Constants as Constants
import csv
from tensorflow.keras.models import model_from_json
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.utils import  plot_model

def start():
    
    print(">> IMAGE CLASSIFICATION MODEL CREATION")

    print()
    print(">> STEP#1 IMAGE PRE-PROCESSING")


    #------------------

    #data = pd.read_csv('datasets_FINAL/test/chest_xray_metadata.csv') #promeni putanju do lokalnog foldera ako je potrebno !!!
    data = pd.read_csv('datasets_FINAL/train/00_train.csv')

    #print(data.columns)
    data.shape
    #---------------


    img_width = 224
    img_height = 224

    X = []

    #xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_DIR, "*.jpeg"))
     #           if("_merged" in test_file)]

    xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_TRAIN_DIR, "*.jpeg")) \
                if ("_merged" in test_file)]

    print(len(xray))
    #for i in tqdm(range(142)):
    for lung in xray:
        # path = 'content/ori-projekat/X-Ray/metadata/'+ data['X_ray_image_name'][i]

        #path = 'content/dataset-x-ray/data/'+ data['X_ray_image_name'][i]
        #img = image.load_img(path, target_size=(img_width,img_height,3))
        img = image.load_img(lung, target_size=(img_width,img_height,3))
        img = image.img_to_array(img)
        img = img/255.0 #kada se stavi neki drugi broj umesto 255 kao npr 224 dobijaju se znatno losiji rezultati
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #
        img_tr = (img_gray > 0.5) * (img_gray < 0.8) #
        img_tr_rgb = cv2.cvtColor(img_tr.astype('uint8'), cv2.COLOR_GRAY2RGB) #
        X.append(img)
        #  X.append(img*img_tr_rgb)  #ovako se dodaju rgb slike umesto gray
    
    X = np.array(X)
    #X.shape
    #plt.imshow(X[50])

    # import math
    Y = []
    # y = data.drop(['Unnamed: 0','X_ray_image_name','Label_2_Virus_category', 'Label_1_Virus_category'], axis=1)[0:1500]
    # y = data.drop(['Unnamed: 0','X_ray_image_name', 'Label','Label_2_Virus_category'], axis=1)[0:1500]
    y = data.drop(columns=['Unnamed: 0', 'X_ray_image_name', 'Label', 'Label_2_Virus_category'])[
        0:data.shape[0]]  # [0:1500]

    print("PRE")
    y.head()
    #print(y.dtypes)

    # print(y)
    y = y.to_numpy()
    # y.shape
    # for i in tqdm(range(1500)):
    print(data.shape[0], " sssssssssss")
    for i in tqdm(range(data.shape[0])):
        # print(y[i])
        label = ""
        if (pd.isnull(y[i])):
            label = [1, 0, 0]
        elif (y[i] == 'Virus'):
            label = [0, 1, 0]
        elif (y[i] == 'bacteria'):
            label = [0, 0, 1]
        Y.append(label)
    # print('jej')
    # print(y[i])
    # y.dtype
    print(Y)
    Y = np.array(Y)
    # print(Y)
    print(y[50])
    print(y.shape)
    Y.shape
    print(Y[0])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state =0, test_size=0.15)
    #Simple CNN model based on VGG16
    LEARNING_RATE =0.0005 #start off with high rate first 0.001 #5e-4


    conv_base = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    include_top=False,
                    input_shape=(224, 224, 3))

    conv_base.trainable = False


    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',             
                optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum = 0.0, nesterov = True, name = "SGD"),
                metrics=['accuracy'])

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    epochs = 15

    '''
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.callbacks import ModelCheckpoint
    class EarlyStoppingByLossVal(Callback):
        def __init__(self, monitor='val_accuracy', value=0.00001, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current > 0.84:
            print(current)
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


    callback = EarlyStoppingByLossVal(monitor='val_accuracy', value=0.00001, verbose=1)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy')
    '''
    #history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),  callbacks = [mcp_save])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    number_of_epochs_it_ran = len(history.history['loss'])

    model_json = model.to_json()
    with open("model_vgg.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #Model.model().save_weights("model.h5")
    print("Saved model to disk")

    #model.load_weights('.mdl_wts.hdf5')


    '''
    score = model.evaluate(X_test, y_test, verbose = 1)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])
    '''
    #ili ovo

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

    miss =0
    hit = 0
    #test_data = pd.read_csv('/content/ori-projekat/X-Ray-test/metadata/chest_xray_test_dataset.csv')

    #test_data = pd.read_csv('metadata-test/chest_xray_test_dataset.csv') #ovo
    test_data = pd.read_csv('datasets_FINAL/test/00_test.csv') #OVDE SAMO STAVIS PUTANJU DO CSV FAJLA SLIKA ZA KOJE ZELIS PREDIKCIJU

    #--------------------------
    Y_test_data = []
    y_test_data = test_data.drop(columns=['Unnamed: 0', 'X_ray_image_name', 'Label', 'Label_2_Virus_category'])[0:test_data.shape[0]]
    # print(y)
    y_test_data = y_test_data.to_numpy()
    # y.shape

    for i in tqdm(range(98)):
        #for i in tqdm(range(data.shape[0])):
        # print(y[i])
        if (pd.isnull(y_test_data[i])):
            label = [1, 0, 0]
        elif (y_test_data[i] == 'Virus'):
            label = [0, 1, 0]
        elif (y_test_data[i] == 'bacteria'):
            label = [0, 0, 1]
        Y_test_data.append(label)
    # print('jej')
    # print(y[i])
    # y.dtype
    #print(Y_test_data)
    Y_test_data = np.array(Y_test_data)
    # print(Y)
    #print(y[1499])
    print(y.shape)
    Y_test_data.shape


    #--------------------------
    #test2 = data.drop(columns=['Unnamed: 0', 'Label', 'Label_2_Virus_category' ])[1340:1345]
    #test2.head()

   # xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_TEST_DIR, "*.jpeg"))
    #            if("_merged" in test_file)]

    xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_TEST_DIR, "*.jpeg"))\
                if ("_merged" in test_file)]

    X_test_data = []
    for i in range(len(xray)):
    #for i in tqdm(range(test_data.shape[0])):
        #for i in tqdm(range(test_data.shape[0])):
        if (i == 624):
            break
        #k =  i+1340
        #path = '/content/ori-projekat/X-Ray-test/metadata/'+ str(test_data['X_ray_image_name'][i])
        #  path = 'metadata-test/'+ str(test_data['X_ray_image_name'][i]) #ovo
        #path = '/content/dataset-x-ray/data/'+ str(test_data['X_ray_image_name'][i]) #OVDE STAVIS PUTANJU FOLDERA U KOM TI SE NALAZE SLIKE ZA TEST

        #print(test_data['X_ray_image_name'][i])
        klasa = test_data['Label_1_Virus_category'][i]
        #path = '/content/dataset-x-ray/data/'+ data['X_ray_image_name'][i]
        img = image.load_img(xray[i], target_size=(img_width,img_height,3))
        img = image.img_to_array(img)
        img = img/255.0
        X_test_data.append(img)
        img = img.reshape(1,img_width,img_height,3) #3 je zbog tri kanala bojetj.rgb
        y_prob = model.predict(img) #vraca predikciju za svaku od target klasa
        #print(y_prob[0], " - ", klasa)
        if(y_prob[0][0] > y_prob[0][1] and y_prob[0][0]> y_prob[0][2]): #ako je predvideo da je normal
            if(pd.isnull(klasa)): #i ako stvarna klasa jeste normal
                hit = hit +1
            else:
                miss = miss +1
        elif(y_prob[0][1] > y_prob[0][0] and y_prob[0][1]> y_prob[0][2]): #ako je predvideo da je virus
            if(klasa == 'Virus'): #i ako stvarna klasa jeste virus
                hit = hit +1
            else:
                miss = miss +1
        elif(y_prob[0][2] > y_prob[0][0] and y_prob[0][2]> y_prob[0][1]): #ako je predvideo da je bakterija
            if(klasa == 'bacteria'): #i ako stvarna klasa jeste bakterija
                hit = hit +1
            else:
                miss = miss +1


    X_test_data = np.array(X_test_data)
    X_test_data.shape  
    X_train, X_test, y_train, y_test = train_test_split(X_test_data, Y_test_data, random_state =0, test_size=0.15)


    '''
    score = model.evaluate(X_test, y_test, verbose = 1)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])
    '''
    #ili ovo
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2) # ovo je sa tensorflow sajta
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_acc)


    print("HIT: " + str(hit) + "    MISS: " +  str(miss))
    


def new_function():
    xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_TRAIN_DIR, "*.jpeg")) \
                if ("_resized" not in test_file)]
    print(len(xray), "duzina")
    
    with open('datasets_FINAL/test/FF.csv', 'r') as inp, open('3rd_edit.csv', 'w') as out:
        writer = csv.writer(out)
        counter = 1
      
        for row in csv.reader(inp):
            #print(row[1])
            for i in range(len(xray)):
                row_name = row[1].replace(".jpeg","").replace(".jpg", "")
                xx = xray[i].split("\\")[2].replace(".jpeg", "")
                #print (row_name, " vs ", xx)
                if row_name == xx:
                    writer.writerow(row)
                    break
                    
            print(counter)
            counter+=1
   


        