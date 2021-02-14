import os
import cv2
import numpy as np
from glob import glob
import Segmentation.Constants as Constants

def extract2Largest(directory):
    masks = [test_file for test_file in glob(os.path.join(directory, "*.*")) \
              if ("_predict" in test_file \
                  and ".csv" not in test_file)]
    
    c = 1
    lent = len(masks)
    for image in masks:

        filename, fileext = os.path.splitext(image)

        SAVE_PATH = os.path.join("%s_proccessed%s" % (filename, fileext))

        # Read the image
        img=cv2.imread(image)
        imgCont=img.copy()

        # Convert to Gray
        imgGray =255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find edges
        imgEdges = cv2.Canny(imgGray,150,200)

        # Find contour
        contours,hierarchy =cv2.findContours(imgEdges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

        # Loop through contours and find the two biggest area.
        out = np.zeros(img.shape, np.uint8)
        areas = [cv2.contourArea(c) for c in contours]
        areas.sort(reverse=True)
        print (c, "/", lent)
        
        for cont in contours:
            area=cv2.contourArea(cont)
            '''
            if len(areas)<=4:
                s = 2
                #cv2.imwrite(SAVE_PATH, img)
                #break
            '''
            try:
                if area>=areas[1]:
                    #print(area)
                    cv2.fillPoly(out, pts =[cont], color=(240,240,240))
                    
                    #cv2.imshow(" ", out)
                    #cv2.waitKey()
                    #cv2.drawContours(out,cont,-1,(0,145,255),thickness=-1)
            except:
                print("no image")
        c+=1

        

        #(thresh, im_bw) = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imwrite(SAVE_PATH, out)
        
        '''
        im = cv2.imread(image)

        # Generate intermediate image; use morphological closing to keep parts of the brain together
        #gray = cv2.cvtColor(im, cv2.COLOR_GRAYS) 
        gray = cv2.imread('grayscale_image.png', cv2.IMREAD_GRAYSCALE)
        edged = cv2.Canny(gray, 30, 250)  #200
        # Find largest contour in intermediate image
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cnt = max(cnts, key=cv2.contourArea)
        cnt = sorted(contours, key=cv2.contourArea)
        print(len(cnt))
        # Output
        out = np.zeros(im.shape, np.uint8)
        cv2.drawContours(out, cnt, len(contours)-1, 255, cv2.FILLED)
        cv2.drawContours(out, cnt, len(contours)-2, 255, cv2.FILLED)
        #out = cv2.bitwise_and(im, out)

        #cv2.imshow('out', out)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        filename, fileext = os.path.splitext(image)

        SAVE_PATH = os.path.join("%s_proccessed%s" % (filename, fileext))

        #(thresh, im_bw) = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imwrite(SAVE_PATH, out)
        print("Processed (%d/%d)." % (c, lent))
        c+=1
        '''

def applyMaskToImage(directory):

    xray = [test_file for test_file in glob(os.path.join(directory, "*.*")) \
              if ("_resized" in test_file \
                  and "_predict" not in test_file)]

    masks = [test_file for test_file in glob(os.path.join(directory, "*.*")) \
              if ("_proccessed" in test_file)]
    print(len(xray), len(masks))
    for i in range(len(masks)):
        '''
        img = cv2.imread(xray[i])
        mask = cv2.imread(masks[i],0)
        res = cv2.bitwise_and(img, img, mask = mask)
        '''

        img = cv2.imread(xray[i])
        mask2 = cv2.imread(masks[i])/255.0
        # mask by multiplication, clip to range 0 to 255 and make integer
        result2 = (img * mask2).clip(0, 220).astype(np.uint8)
      

    
        filename, fileext = os.path.splitext(xray[i])
        route = filename.split("\\")
        filename = route[0] + "\\"+route[1]+"\\"+"merged\\"+route[2]
        result_file = os.path.join("%s_merged%s" % (filename, fileext))    
        
    
        cv2.imwrite(result_file, result2)
        print("Processed (%d/%d)." % (i+1, len(masks)))
    exit()

def resize_xrays(directory):
    c = 0 
    xray = [test_file for test_file in glob(os.path.join(directory, "*.*")) \
              if ("_proccessed" not in test_file \
                  and "_predict" not in test_file
                  and ".csv" not in test_file)]
    lent = len(xray)
    for image in xray:
        print(image)
        
        img = cv2.imread(image)
        
        ressed = __resize_image(img, (512,512))
        filename, fileext = os.path.splitext(image)
        result_file = os.path.join("%s_resized%s" % (filename, fileext))
        print("Processed (%d/%d)." % (c, lent))
        c+=1
        cv2.imwrite(result_file, ressed)


def __resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def normalize():
    xray = [test_file for test_file in glob(os.path.join(Constants.FINAL_DIR, "*.jpeg")) \
              if ("_resized" in test_file \
                  and "_predict" not in test_file)]
    c = 1
    for image in xray:

        img = cv2.imread(image,0)
        equ = cv2.equalizeHist(img)
        
        filename, fileext = os.path.splitext(image)

        SAVE_PATH = os.path.join("%s_equalized%s" % (filename, fileext))

        cv2.imwrite(SAVE_PATH, equ)
        print("Processed (%d/%d)." % (c, len(xray)))
        c+=1

def tst(directory):
    xray = [test_file for test_file in glob(os.path.join(directory, "*.jpeg")) \
              if ("_merged" in test_file)]
    c = 1
    for image in xray:
        filename, fileext = os.path.splitext(image)

        SAVE_PATH = os.path.join("%s_equalized%s" % (filename, fileext))

        src = cv2.imread(image, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

        cv2.imwrite(SAVE_PATH, dst)
        print("Processed (%d/%d)." % (c, len(xray)))
        c+=1

       