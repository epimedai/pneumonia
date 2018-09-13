from bbox.annotation_container import AnnotationContainer
from pathlib import Path
import os
from collections import defaultdict

a = AnnotationContainer.from_file(Path(os.getcwd()+'/lesson3-data/training_data.json'))
print(len(a.entries))

a.summary()
#a.analytic().plot_bbox_size_ratio()

count = defaultdict(int)
for e in a:
    count[len(e.instances)]+=1

print('# of boxes in each entry:')
for k, v in count.items():
    print('\t', k, v)

for e in a:
    if len(e.instances) == 1:
        print(e.image_path)
        e.show()

