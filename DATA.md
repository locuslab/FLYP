Structure for ImageNet Data
```
ILSVRC2012
├── train
│   ├── 
│   └── 
├── val_dirs
│   ├── 
│   └── 
```

to link ILSVRC2012 data, run
```
ln -s PATH_TO_YOUR_ILSVRC2012_DATASET ./datasets/data/ILSVRC2012
```


Structure for FMOW Data
```
fmow_v1.1
├── images
│   ├── 
│   └── 
├── rgb_metadata.csv
├── country_code_mapping.csv
```

to link FMOW data, run
```
ln -s PATH_TO_YOUR_FMOW_DATASET ./datasets/data/fmow_v1.1
```

Structure for sst2 data
```
sst2
├── test
│   ├── negative
│   └── positive
├── train
│   ├── negative
│   └── positive
└── val
    ├── negative
    └── positive
```
to link sst2, run
```
ln -s PATH_TO_YOUR_SST2_DATASET ./datasets/data/sst2
```


structure of patchcamelyon
```
patchcamelyon
├── camelyonpatch_level_2_split_test_meta.csv
├── camelyonpatch_level_2_split_test_x.h5
├── camelyonpatch_level_2_split_test_y.h5
├── camelyonpatch_level_2_split_train_mask.h5
├── camelyonpatch_level_2_split_train_meta.csv
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_valid_meta.csv
├── camelyonpatch_level_2_split_valid_x.h5
├── camelyonpatch_level_2_split_valid_y.h5
```
to link patchcamelyon, run
```
ln -s PATH_TO_YOUR_PATCH_CAM_DATASET ./datasets/data/patchcamelyon
```


stanford
```
StanfordCars
├── cars_test
├── cars_test_annos_withlabels.mat
├── cars_train
├── devkit
```
```
ln -s PATH_TO_YOUR_STANFORD_CARS_DATASET ./datasets/data/StanfordCars
```


flower
```
flowers102
├── all_images
├── imagelabels.mat
├── setid.mat
```
```
ln -s PATH_TO_YOUR_FLOWERS_DATASET ./datasets/data/flowers102
```

caltech
```
├── 101_ObjectCategories
├── caltech101_classnames_clip.py
├── show_annotation.m
├── test
├── train
├── traintestsplit.py
└── val
```
```
ln -s PATH_TO_YOUR_CALTECH_DATASET ./datasets/data/caltech-101
```