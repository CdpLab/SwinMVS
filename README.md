# SwinMVS
## CODE for **High-Resolution Multi-View Stereo with Multi-Scale Feature Fusion**
## Acquiring Datasets
### DTU
### BlendedMVS
### Tanks&Temples
### ETH3D

## Requirements
bash
pip install torch torchvision numpy opencv-python tensorboardX matplotlib pillow
## Train
python train.py --dataset dtu --datapath /path/to/dtu --trainlist ./lists/dtu/train.txt --logdir ./logs/swinmvs_dtu
## Test
python test.py --dataset dtu --datapath /path/to/dtu --testlist ./lists/dtu/test.txt --checkpoint ./logs/swinmvs_dtu/model.pth --outdir ./results/dtu
