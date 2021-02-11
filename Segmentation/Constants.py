import os
TEST_PRECENTAGE = 0.25
# TRAIN_PRECENTAGE = 0.75 # procenat slika koji odlazi na treniranje

# INPUT
''' 
    gotov datasets, koji sadrzi xray pluca
    i groundtruth maske levog i desnog plucnog krila
'''
DATASET = os.path.join("datasets_RAW")
TRAIN_DIR = os.path.join(DATASET, "Montgomery")
IMAGE_DIR = os.path.join(DATASET, "Montgomery", "CXR_png")
LEFT_MASK_DIR = os.path.join(DATASET, "Montgomery", "ManualMask", "leftMask")
RIGHT_MASK_DIR = os.path.join(DATASET, "Montgomery",  "ManualMask", "rightMask")

# OUTPUT
''' 
    pocetni dataset pretprocesiran za mrezu i spreman za treniranje: 
        -maske spojene
        -razdvojeni xray-evi na TRAIN i TEST
        -primenjen blur na maske
'''
OUTPUT_DIR = os.path.join("datasets_PROCCESSED", "Montgomery")
OUTPUT_TEST_DIR = os.path.join(OUTPUT_DIR, "test")
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
OUTPUT_TRAIN_PICTURES = os.path.join(OUTPUT_TRAIN_DIR, "xrays")
OUTPUT_TRAIN_MASKS = os.path.join(OUTPUT_TRAIN_DIR, "masks")
OUTPUT_TRAIN_BLURED = os.path.join(OUTPUT_TRAIN_DIR, "blured")
OUTPUT_AUG_DIR = os.path.join(OUTPUT_DIR, "aug")

# DATASET for final step
''' 
    novi dataset koji sadrzi razlicite xray snimke od pocetnog seta
    obelezenih sa tipom pneumonije koji je prisutan na snimku
'''
FINAL_DIR = os.path.join("datasets_FINAL", "test")

