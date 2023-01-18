# FLYP: Finetune Like You Pretrain

Code for the paper Finetune like you pretrain: Improved finetuning of zero-shot vision models.

CREDITS: Our code is heavily based on https://github.com/mlfoundations/wise-ft and https://github.com/mlfoundations/open_clip. We thank the authors for open sourcing their code.

## Setting up conda env
```bash
conda create -n flyp python=3.10
conda activate flyp
pip install open_clip_torch
pip install wilds braceexpand webdataset h5py
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
mkdir checkpoints
```

### Add directory to PYTHONPATH:

```bash
cd FLYP
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Datasets
All the datasets we use are available publicly.
Refer to the DATA.md for the respective dataset directory strucutre.

### Script to reproduce on ImageNet
```bash
ln -s PATH_TO_YOUR_ILSVRC2012_DATASET ./datasets/data/ILSVRC2012

python datacreation_scripts/imagenet_csv_creator.py

python src/main.py --train-dataset=ImageNet --epochs=10 --lr=1e-5 --wd=0.1 --batch-size=512 --model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  --template=openai_imagenet_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" --csv-img-key filepath --csv-caption-key title --exp_name=ImageNet/flyp_loss
```

### Script to reproduce on iWILDCam
```bash
ln -s PATH_TO_YOUR_iWILDCam_DATASET ./datasets/data/iwildcam_v2.0

python datacreation_scripts/iwildcam.py

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/flyp_loss
```

### Script to reproduce on FMOW
```bash
ln -s PATH_TO_YOUR_FMOW_DATASET ./datasets/data/fmow_v1.1

python datacreation_scripts/fmow_csv_creator.py

python src/main.py --train-dataset=FMOWIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=FMOWIDVal,FMOWID,FMOWOOD --template=fmow_template --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/fmow.csv" --csv-img-key filepath --csv-caption-key title --exp_name=fmow/flyp_loss
```

### Few shot on SST2

```bash

ln -s PATH_TO_YOUR_SST2_DATASET ./datasets/data/sst2

python datacreation_scripts/sst2.py

arch="ViT-B/16"
k=16

python src/few_shot.py --train-dataset=sst2Val --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=$arch --warmup_length 0 --eval-datasets=sst2Val,sst2Test --template=sst2_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/sst2/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=sst2/"flyp_loss_"$k"shot" --k=$k 
```

For VitL
```
arch="ViT-L/14"
```

### Few shot on PatchCamelyon

```bash
ln -s PATH_TO_YOUR_PATCH_CAM_DATASET ./datasets/data/patchcamelyon

python datacreation_scripts/patchcamelyon.py

arch="ViT-B/16"
k=16

python src/few_shot.py --train-dataset=PatchCamelyonVal --epochs=20 --lr=1e-6 --wd=0.0 --batch-size=256 --model=$arch --warmup_length 0 --eval-datasets=PatchCamelyonVal,PatchCamelyonTest --template=patchcamelyon_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/patchcamelyon/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=patchcamelyon/"flyp_loss_"$k"shot" --k=$k 
```

### Transfer Learning on Caltech

```bash
ln -s PATH_TO_YOUR_CALTECH_DATASET ./datasets/data/caltech-101

python datacreation_scripts/caltech101.py

python src/few_shot.py --train-dataset=Caltech101Val --epochs=100 --lr=1e-5 --wd=0.0 --batch-size=256 --model=ViT-B/16 --warmup_length 500 --eval-datasets=Caltech101Val,Caltech101Test --template=caltech101_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/caltech101/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=caltech101/flyp_loss
```

### Transfer Learning on StanfordCars

```bash
ln -s PATH_TO_YOUR_STANFORD_CARS_DATASET ./datasets/data/StanfordCars

python datacreation_scripts/stanfordCars.py

python src/few_shot.py --train-dataset=StanfordCarsVal --epochs=100 --lr=1e-5 --wd=0.0 --batch-size=256 --model=ViT-B/16 --warmup_length 500 --eval-datasets=StanfordCarsVal,StanfordCarsTest --template=standfordcars_template  --save=./checkpoints/ --data-location=./datasets/data/ --ft_data="./datasets/csv/StanfordCars/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=standfordcars/flyp_loss
```

### Cross Entropy Ablation on ImageNet
This refers to the cross-entropy ablation, where we use language representations as a linear head over the image representations, projecting the image representations to class probabilities. Simply add a flag by --ce_ablation to any of the above cmd line scripts. Here we provide the cmd line script for ImageNet.

```bash
python src/main.py --train-dataset=ImageNet --epochs=10 --lr=1e-5 --wd=0.1 --batch-size=512 --model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  --template=openai_imagenet_template  --save=./checkpoints/ --data-location=./datasets/data/ --exp_name=ImageNet/ce_ablation --ce_ablation
```