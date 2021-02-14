from Segmentation import Merger as merger
from Segmentation import Train as train
from Segmentation import MaskProccessing as mp
from Classification import vgg
from Segmentation import Constants as Constants
def main():
    # popunjava test i train u proccessed
    #merger.merge()
    #train.train() #mreza je vec istrenirana i sacuvana

    #mp.resize_xrays(Constants.FINAL_TRAIN_DIR)
    #mp.resize_xrays(Constants.FINAL_TEST_DIR)

    #train.predictUsingSavedModel(Constants.FINAL_TRAIN_DIR)
    #train.predictUsingSavedModel(Constants.FINAL_TEST_DIR)

    #mp.extract2Largest(Constants.FINAL_TRAIN_DIR)
    #mp.extract2Largest(Constants.FINAL_TEST_DIR)

    #mp.applyMaskToImage(Constants.FINAL_TEST_DIR)
    #mp.applyMaskToImage(Constants.FINAL_TRAIN_DIR)

    #mp.normalize() ne pokretati, za normalizaciju histograma, ali ne pomaze

    # # mp.tst(Constants.FINAL_TEST_DIR)

    vgg.start()
    #vgg.new_function()



if __name__ == '__main__':
    main()
