import os
import sys

import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob
import json
from pathlib import Path

# This code is as reworked version of https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb

#####################################################
# Change to path to fit local filestructure
kaggle_info_path = '/home/martin/.kaggle/kaggle.json'
#####################################################

kaggle_info = json.load(open(kaggle_info_path, 'r'))
os.environ['KAGGLE_USERNAME'] = kaggle_info['username']
os.environ['KAGGLE_KEY'] = kaggle_info['key']

# Root directory of the project
ROOT_DIR = os.path.abspath('./lesson3-data')


if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
os.chdir(ROOT_DIR)

# If you are unable to download the competition dataset, check to see if you have
# accepted the user agreement on the competition website.


# Downloading dataset
os.system('kaggle competitions download -c rsna-pneumonia-detection-challenge')

# Unziping dataset
os.system('unzip -q -o stage_1_test_images.zip -d stage_1_test_images')
os.system('unzip -q -o stage_1_train_images.zip -d stage_1_train_images')
os.system('unzip -q -o stage_1_train_labels.csv.zip')

