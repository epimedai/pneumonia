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

from kaggle_info import KaggleInfoContainer

kaggle_info = KaggleInfoContainer()
os.environ['KAGGLE_USERNAME'] = kaggle_info.kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_info.kaggle_key

# Root directory of the project
ROOT_DIR = os.path.abspath('./lesson3-data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
    os.chdir(ROOT_DIR)

    # If you are unable to download the competition dataset, check to see if you have
    # accepted the user agreement on the competition website.

    # Downloading and unziping dataset
    os.system('kaggle competitions download -c rsna-pneumonia-detection-challenge')

    os.system('unzip -q -o stage_1_test_images.zip -d stage_1_test_images')
    os.system('unzip -q -o stage_1_train_images.zip -d stage_1_train_images')
    os.system('unzip -q -o stage_1_train_labels.csv.zip')

