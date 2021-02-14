
import cv2
import numpy as np
import os
import joblib

from numpy.lib.function_base import percentile
import Segmentation.Constants as Constants
import Segmentation.Model as Model
from glob import glob

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import model_from_json

import matplotlib.pyplot as plt 
from PIL import Image, ImageEnhance

import skimage
from skimage import measure

BATCH_SIZE=2

def test_load_image(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)
        
def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

        cv2.imwrite(result_file, img)


def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)

train_files = glob(os.path.join(Constants.OUTPUT_TRAIN_PICTURES, "*.png"))
test_files = glob(os.path.join(Constants.OUTPUT_TEST_DIR, "*.png"))

def train():
    test_files = [test_file for test_file in glob(os.path.join(Constants.OUTPUT_TEST_DIR, "*.png")) \
              if ("_mask" not in test_file \
                  and "_blur" not in test_file)]
    print(test_files[0])

    validation_data = (test_load_image(test_files[0], target_size=(512, 512)),
                    test_load_image(add_suffix(test_files[0], "blur"), target_size=(512, 512)))

    train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

    train_gen = Model.train_generator(BATCH_SIZE,
                            Constants.OUTPUT_TRAIN_DIR,
                            'xrays',
                            'blured', 
                            train_generator_args,
                            target_size=(512,512))

    


    model = Model.model(input_size=(512,512,1))
    model.compile(optimizer=Adam(learning_rate=3e-4), loss=Model.dice_coef_loss, \
                    metrics=[Model.dice_coef, 'binary_accuracy'])
    model.summary()

    model_checkpoint = ModelCheckpoint('xray_model.hdf5', 
                                    monitor='loss', 
                                    verbose=1, 
                                    #save_best_only=True,
                                    save_weights_only=True)

    history = model.fit_generator(train_gen,
                                steps_per_epoch=len(train_files) / BATCH_SIZE, 
                                epochs=60, 
                                callbacks=[model_checkpoint],
                                validation_data = validation_data)


    test_gen = test_generator(test_files, target_size=(512,512))
    results = model.predict_generator(test_gen, len(test_files), verbose=1)
    save_result(Constants.OUTPUT_TEST_DIR, results, test_files)

    
    #model_json = Model.model().to_json()
    #with open("model.json", "w") as json_file:
     #   json_file.write(model_json)
    # serialize weights to HDF5
    #Model.model().save_weights("model.h5")
    print("Saved model to disk")
    


def predictUsingSavedModel(directory):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("xray_model.hdf5")
    print("Loaded model from disk")

    test_files = [test_file for test_file in glob(os.path.join(directory, "*.*")) \
                if ("_resized" in test_file)]

    test_gen = test_generator(test_files, target_size=(512,512))
    results = loaded_model.predict_generator(test_gen, len(test_files), verbose=1)
    save_result(directory, results, test_files)



'''
for image in xray:
    filename, fileext = os.path.splitext(image)

    SAVE_PATH = os.path.join("%s_enhanced%s" % (filename, fileext))
   
    pic = cv2.imread(image, 0)
    
    #histogram Equalization
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(pic)
    equ = cv2.equalizeHist(pic)
    #res = np.hstack((img, cl1))
    cv2.imwrite(SAVE_PATH,equ)
    
   
    basewidth = 512
    img = Image.open(image).convert('LA')
    # determining the height ratio
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    # resize image and save
    pic = img.resize((basewidth,hsize), Image.ANTIALIAS)

    SAVE_PATH = os.path.join("%s_enhanced%s" % (filename, ".png"))
    pic.save(SAVE_PATH)


   
'''

