
This repository is the code titled "UMIENet: Underwater image enhancement based on multi-degradation knowledge integration".

## UMDKI dataset
Google Drive: [UMDKI](https://drive.google.com/file/d/19WCkl-Bbx_m9xcHdDAfk0mT2EHQF_5Oq/view?usp=sharing)

## train
Change the training parameters and dataset path in **config.yml**
`python train.py`

## test
Change the testing parameters and dataset path in **config.yml**
`python test.py`

## train and test custom dataset
* Dataset Structure
```
dataset/
├── train/
│   ├── input/
│   └── target/
├── test/
│   ├── input/
│   └── target/
└── real_dir/
```
* Change the testing parameters and dataset path in **config.yml**
* `python train.py` or `python test.py`

we use [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) to implement domain adversarial loss in our code.