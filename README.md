# vanillapose
6DoF Pose estimation with the conventional encoder + MLP architecture.

This repository exists to compare the performance of the conventional encoder + MLP architecture vs state-of-the-art models like PoseCNN, DenseFusion, PVNet, PVN3D, FFB6D, etc.

## Requirements and Installation
Install dependencies with
```shell
pip install -r requirements.txt
```
It is recommended that installation be performed in an anaconda environment.

## Training Configuration
The Base model architecture consists of an encoder head and a MLP layer. The encoder head and MLP configurations can be specified in a yaml file given in ``configs/experiment/``.

An example of the syntax used for ResNet50 as an encoder and a MLP with 3 layers:
```shell
model:
    type: base
    encoder: resnet50
    mlp_layers: 3
```

After specifying the model, run train_base.py with the following syntax:
```shell
python train_base.py +experiment=<config_name.yaml>
```
For the yaml file provided in this repository:
```shell
python train_base.py +experiment=resnet50_mlp3.yaml
```

## Inference
- to-be-added
