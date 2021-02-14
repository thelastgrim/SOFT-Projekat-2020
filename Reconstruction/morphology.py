import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os


def funkcija():

    data_dir='datasets_FINAL/train/'
    train_imgs = glob(os.path.join(data_dir, "*.jpeg"))
    train_imgs = [f for f in train_imgs if "_proccessed" in f]
    print (len(train_imgs))

    for train_img in tqdm(train_imgs):
      img = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)
      kernel = np.ones((3,3),np.uint8)
      erosion = cv2.erode(img,kernel, iterations=5)
      dilation = cv2.dilate(erosion, kernel, iterations=10)

      filename, fileext = os.path.splitext(train_img)

      SAVE_PATH = os.path.join("%s_dilated%s" % (filename, fileext))

      cv2.imwrite(SAVE_PATH, dilation)
      #cv2.imwrite(os.path.join('datasets_FINAL', os.path.basename(train_img+"_dialted")), dilation)


if __name__ == '__main__':
  funkcija()

