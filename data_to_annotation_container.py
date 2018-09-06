from bbox.annotation_container import AnnotationContainer, AnnotationEntry, AnnotationInstance
from bbox.instances import BBox
from bbox.dataset_source_provider import DatasetSourceProvider

from matplotlib import pyplot as plt

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import pydicom
import os
import math

def get_dicom_fps(dicom_dir):
    dicom_fps = [str(f) for f in list(dicom_dir.glob('*.dcm'))]
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    anns = pd.read_csv(open(anns, 'r'))
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = str(dicom_dir / (row['patientId']+'.dcm'))
        image_annotations[fp].append(row)
    return image_fps, image_annotations

def load_image(image):
    ds = pydicom.read_file(image)
    image = ds.pixel_array
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image

def fill_annotation_container(image_dir, annotations, container):
    for image_file in tqdm(image_dir):
        annotation = annotations[image_file][0]
        image = load_image(image_file)
        img_width, img_height = image.shape[0], image.shape[1]
        instances = []
        if not math.isnan(annotation['x']):
            xmin = annotation['x']
            ymin = annotation['y']
            width = annotation['width']
            height = annotation['height']
            instances.append(AnnotationInstance(bbox=BBox(xmin=xmin,
                                                          ymin=ymin,
                                                          xmax=xmin + width,
                                                          ymax=ymin + height,
                                                          label='target',
                                                          coordinate_mode='absolute',
                                                          source='HUMAN')))
        entry = AnnotationEntry(Path(image_file).name,
                                (img_width, img_height),
                                dataset_name='pneumonia', instances=instances,
                                dataset_subset='train')
        container.add_entry(entry)

    return container


root_dir = Path(os.getcwd()+'/lesson3-data')
train_dicom_dir = root_dir / 'stage_1_train_images'
test_dicom_dir = root_dir / 'stage_1_test_images'
sample_csv_file = root_dir / 'stage_1_sample_submission.csv'
train_csv_file = root_dir / 'stage_1_train_labels.csv'

# Convert training set
dsp = DatasetSourceProvider()
dsp.add_source(str(train_dicom_dir), dataset_name='pneumonia', dataset_subset='train')
container = AnnotationContainer(dataset_source_provider=dsp)

images, annotations = parse_dataset(train_dicom_dir, train_csv_file)
container = fill_annotation_container(images, annotations, container)
container.write_json(root_dir / 'training_data.json')