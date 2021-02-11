from Segmentation import Merger as merger
from Segmentation import Train as train
from Segmentation import MaskProccessing as mp
from Segmentation import vgg as vgg
def main():
    #1 merger.merge()
    #mp.resize_xrays()
    #train.predictUsingSavedModel()
    #mp.extract2Largest()
    mp.applyMaskToImage()
    #mp.normalize()




if __name__ == '__main__':
    main()
