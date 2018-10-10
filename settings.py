import os

# Base daatset path
BASE_PATH = 'data'

# Record paths
RECORD_PATH = os.path.join(BASE_PATH, 'records')
TRAIN_RECORD = os.path.join(RECORD_PATH, 'train.record')
TEST_RECORD = os.path.join(RECORD_PATH, 'test.record')
CLASSES_FILE = os.path.join(RECORD_PATH, 'classes.pbtxt')

# Test split size
TEST_SIZE = 0.1

# Class label dictionary {class_name_1: 1, class_name2: 2...}
CLASSES = {}