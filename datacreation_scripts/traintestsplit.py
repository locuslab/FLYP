import os
import numpy as np
import shutil


data_path = "./101_ObjectCategories/"
categories = sorted(os.listdir(data_path))

for cat in categories:
    print(cat)
    os.mkdir(os.path.join("./train/", cat))
    os.mkdir(os.path.join("./val/", cat))
    os.mkdir(os.path.join("./test/", cat))
    
    image_files = os.listdir(os.path.join(data_path, cat))
    choices = np.random.choice([0, 1, 2], size=(len(image_files),), p=[.6, 0.2, 0.2])
    
    for (i,_f) in enumerate(image_files):
        if choices[i]==0:
            dest_path = os.path.join("./train/", cat, _f)
        if choices[i]==1:
            dest_path = os.path.join("./val/", cat, _f)
        if choices[i]==2:
            dest_path = os.path.join("./test/", cat, _f)
        
        origin_path = os.path.join(data_path, cat,  _f)
        shutil.copy(origin_path, dest_path)