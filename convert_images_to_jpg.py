from pathlib import Path
from tqdm import tqdm
import pydicom
import os
import numpy as np
import scipy.misc

from matplotlib import pyplot as plt

def load_image(image):
    ds = pydicom.read_file(image)
    image = ds.pixel_array
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image

train_image_dir = Path(os.getcwd()+'/lesson3-data/stage_1_train_images')
test_image_dir = Path(os.getcwd()+'/lesson3-data/stage_1_test_images')

for img_dir in tqdm([test_image_dir, train_image_dir]):
    img_files = [str(f) for f in list(img_dir.glob('*.dcm'))]
    to_dir = str(img_dir)+'_jpg'
    for img_file in tqdm(img_files):
        img = load_image(img_file)
        scipy.misc.imsave(f'{to_dir}/{Path(img_file).stem}.jpg', img)
