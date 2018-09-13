from bbox.annotation_container import AnnotationContainer
from bbox.contrib.tfrecord_formatter import TFRecordFormatter

from pathlib import Path

import os

container_file = Path(os.getcwd()).parent / 'lesson3-data/training_data.json'
container = AnnotationContainer.from_file(container_file)


tfr = TFRecordFormatter(container, container_file.parent / 'training_data.record', ['target'])
broken = tfr.write_tfrecord()
broken.write_json(container_file.parent / 'broken_annotations.json')
broken.summary()
tfr.sanity_check()