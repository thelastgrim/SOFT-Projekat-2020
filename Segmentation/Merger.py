import os
import Segmentation.Constants as Constants
import numpy as np
import cv2
import math
from glob import glob
from tqdm import tqdm

def merge():
    print("Spajanje maski i podela na train i test skup:")
    left_masks = glob(os.path.join(Constants.LEFT_MASK_DIR, '*.png'))

    count = len(left_masks)
    to_train = math.floor(count*Constants.TRAIN_PRECENTAGE)
    
    train_set = left_masks[to_train:]

    for left_mask in tqdm(left_masks):
        base_file = os.path.basename(left_mask)
        image_file = os.path.join(Constants.IMAGE_DIR, base_file)
        right_image_file = os.path.join(Constants.RIGHT_MASK_DIR, base_file)

        image = cv2.imread(image_file)
        image = cv2.resize(image, (512, 512))

        left_mask_image = cv2.imread(left_mask, cv2.IMREAD_GRAYSCALE)
        left_mask_image = cv2.resize(left_mask_image, (512, 512))

        right_mask_image = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)
        right_mask_image = cv2.resize(right_mask_image, (512, 512))

        mask = np.maximum(left_mask_image, right_mask_image)

        if left_mask in train_set:
            filename, fileext = os.path.splitext(base_file)
            cv2.imwrite(os.path.join(Constants.OUTPUT_TRAIN_DIR, base_file), image)
            cv2.imwrite(os.path.join(Constants.OUTPUT_TRAIN_DIR, "%s_mask%s" % (filename, fileext)), mask)
            
        else:
            filename, fileext = os.path.splitext(base_file)
            cv2.imwrite(os.path.join(Constants.OUTPUT_TEST_DIR, base_file), image)
            cv2.imwrite(os.path.join(Constants.OUTPUT_TEST_DIR, "%s_mask%s" % (filename, fileext)), mask)

    print("Gotovo.")