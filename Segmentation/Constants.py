import os
# TEST_PRECENTAGE = 0.25
TRAIN_PRECENTAGE = 0.75 # procenat slika koji odlazi na treniranje

# INPUT
DATASET = os.path.join("datasets_RAW")
TRAIN_DIR = os.path.join(DATASET, "Montgomery")
IMAGE_DIR = os.path.join(DATASET, "Montgomery", "CXR_png")
LEFT_MASK_DIR = os.path.join(DATASET, "Montgomery", "ManualMask", "leftMask")
RIGHT_MASK_DIR = os.path.join(DATASET, "Montgomery",  "ManualMask", "rightMask")

# OUTPUT
OUTPUT_DIR = os.path.join("datasets_PROCCESSED", "Montgomery")
OUTPUT_TEST_DIR = os.path.join(OUTPUT_DIR, "test")
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")



