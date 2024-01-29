# Mult-modal 3D Object Feature Extraction
This repository contains the code for extracting multi-modal features from 3D objects. The features are extracted from the following modalities:
- Multi-view: MVCNN
- Pointclouds: PointNet, DGCNN
- Voxel: 3DShapeNets

## Requirements
- Python 3.6
- Pytorch 1.12
- Hydra-core 1.3.2

## Installation
1. Clone this repository
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Train Model
```bash
python mv_resnet18_train.py
```
### Extract Features
```bash
python mv_resnet18_gen_ft.py
```

