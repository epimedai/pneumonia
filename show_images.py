from bbox.annotation_container import AnnotationContainer
from pathlib import Path

a = AnnotationContainer.from_file('/home/martin/PycharmProjects/pneumonia/lesson3-data/training_data.json')

for e in a:
    #name = Path(e.image_path)
    #name = name.parent / name.stem
    #name = str(name) + '.jpg'
    #e.image_path = Path(name)
    e.show()

#a.write_json('/home/martin/PycharmProjects/pneumonia/lesson3-data/training_data.json')
