# Person Re-id in the 3D Space

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/layumi/person-reid-3d/blob/master/imgs/demo-1.jpg)

Thanks for your attention. In this repo, we provide the code for the paper [[Person Re-identification in the 3D Space ]](https://arxiv.org/abs/2006.04569).

**I will check and upload the code in this week. **

## Prerequisites
- Python 3.6 or 3.7
- GPU Memory >= 4G (e.g., GTX1080)
- Pytorch = 1.4.0 (Not Latest. Latest version is incompatible, since it changes the C++ interfaces.)
- dgl 

## Install 
Here I use the cuda10.1 by default.
```setup
conda create --name OG python=3.7
conda activate OG
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
pip install dgl-cu101
pip install -r requirements.txt
```

## Prepare Data 
Download Market-1501, DukeMTMC-reID or MSMT17 and unzip them in the '../'

Split the dataset and arrange them in the folder of ID.
```bash 
python prepare_market.py
```

Link the 2DDataset 
```bash 
ln -s ../Market/pytorch  ./2DMarket
```

Generate the 3D data via the code at https://github.com/layumi/hmr 
(I modified the code from https://github.com/akanazawa/hmr)


## Training 
- Market-1501

**OG-Net**
```bash
python train_M.py --batch-size 8 --name ALL_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all
```
**OG-Net-Small**
```bash
python train_M.py --batch-size 8 --name ALL_SDense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all     --feature_dims 48,96,192,384
```

- DukeMTMC-reID

**OG-Net**
```bash
python train_M.py --batch-size 8 --name ALL_Duke_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --dataset-path 2DDuke  --train_all
```
**OG-Net-Small**
```bash
python train_M.py --batch-size 8 --name ALL_Duke_SDense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all    --feature_dims 48,96,192,384 --dataset-path 2DDuke
```

- MSMT-17

**OG-Net**
```bash
python train_M.py --batch-size 8 --name MSMT_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --dataset-path 2DMSMT
```
**OG-Net-Small**
```bash
python train_M.py --batch-size 8 --name ALL_MSMT_SDense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d5+bg_adam_init768_clusterXYZRGB --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 5  --erase 0  --droprate 0.5   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --dataset-path 2DMSMT  --train_all  --feature_dims 48,96,192,384
```

## Evaluation
- Market-1501
```bash 
python test_M.py  --name  ALL_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB
```

- DukeMTMC-reID
```bash 
python test_M.py  --name  ALL_Duke_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB  --data 2DDuke
```

- MSMT-17
```bash 
python test_MSMT.py  --name MSMT_Dense_b8_lr3.5_flip_slim0.5_warm5_scale_e0_d7+bg_adam_init768_clusterXYZRGB
```

## Pre-trained Models
Since OG-Net is really small, I will upload them in this github repo directly. 

## Results
### [Image Classification on Market-1501]
| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My model   |     85%         |      95%       |


