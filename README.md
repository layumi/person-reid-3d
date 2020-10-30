# Person Re-id in the 3D Space

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/layumi/person-reid-3d/blob/master/imgs/demo-1.jpg)

Thanks for your attention. In this repo, we provide the code for the paper [[Person Re-identification in the 3D Space ]](https://arxiv.org/abs/2006.04569).

## News
- **30 Oct 2020.** I simply modify code on three points to further improve the performance: 

1. More training epochs help; (Since we are trained from scratch)

2. I replace the dgl to more efficient KNN implementation to accelebrate training; (DGL does not optimize KNN very well, and Matrix Multiplication works quicker. ) 

3. For MSMT-17 and Duke, some classes contain too many images, while other categories are under-explored. I apply the stratified sampling, which takes training samples of each class with equal probability.

- You may directly download my generated 3D data of the Market-1501 dataset at [[OneDrive]](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EQXEskhdd3xPjdFRxAUtB9cB7RkjAdzS5YXRH8QIf_TWAw?e=IhqNpi) or [[GoogleDrive]](https://drive.google.com/file/d/1ih-LrkdGUvNK3rEUNJIq4LTvcgVOXnnM/view?usp=sharing), and therefore you could skip the data preparation part.

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
conda install -c dglteam dgl-cuda10.1=0.4.3
pip install -r requirements.txt
```

If you face any error, you may first try to re-install open3d. It helps. 

## Prepare Data 
- You may directly download my generated 3D data of the Market-1501 dataset at [[OneDrive]](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EQXEskhdd3xPjdFRxAUtB9cB7RkjAdzS5YXRH8QIf_TWAw?e=IhqNpi) or [[GoogleDrive]](https://drive.google.com/file/d/1ih-LrkdGUvNK3rEUNJIq4LTvcgVOXnnM/view?usp=sharing), and therefore you could skip the data preparation part.

Download Market-1501, DukeMTMC-reID or MSMT17 and unzip them in the `../`

Split the dataset and arrange them in the folder of ID.
```bash 
python prepare_market.py
python prepare_duke.py
python prepare_MSMT.py
```

Link the 2DDataset 
```bash 
ln -s ../Market/pytorch  ./2DMarket
ln -s ../Duke/pytorch  ./2DDuke
ln -s ../MSMT/pytorch  ./2DMSMT
```

Generate the 3D data via the code at https://github.com/layumi/hmr 
(I modified the code from https://github.com/akanazawa/hmr and added 2D-to-3D color mapping.)

## Training 
- Market-1501

**OG-Net**
```bash
python train_M.py --batch-size 16 --name Efficient_ALL_Dense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512   --efficient
```
**OG-Net-Small**
```bash
python train_M.py --batch-size 16 --name Efficient_ALL_SDense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384   --efficient 
```

**OG-Net-Deep**
```bash
python train_M.py --batch-size 16 --name Efficient_ALL_2SDDense_b16_lr3.5_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e500_id2_bn_k9_conv2  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 500  --feature_dims 48,96,96,192,192,384,384  --norm_layer bn --efficient --k 9  --num_conv 2
```

- DukeMTMC-reID

**OG-Net**
```bash
python train_M.py --batch-size 16 --name Efficient_Duke_ALL_Dense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_balance --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512   --efficient --dataset 2DDuke --balance
```

**OG-Net-Small**
```bash
python train_M.py --batch-size 16 --name Efficient_Duke_ALL_SDense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384   --efficient --dataset 2DDuke
```

**OG-Net-Deep**
```bash
python train_M.py --batch-size 16 --name Duke_Efficient_ALL_2SDDense_b16_lr3.5_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 3.5e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --norm_layer bn --efficient --k 9  --num_conv 2  --dataset 2DDuke --balance
```

- CUHK-NP 

**OG-Net**
```bash
python train_M.py --batch-size 16 --name Efficient_CUHK_ALL_Dense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512   --efficient --dataset 2DCUHK
```

**OG-Net-Small**
```bash
python train_M.py --batch-size 16 --name Efficient_CUHK_ALL_SDense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384    --efficient --dataset 2DCUHK
```

**OG-Net-Deep**
```bash
python train_M.py --batch-size 32 --name CUHK_Efficient_ALL_2SDDense_b32_lr3_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 3e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --norm_layer bn --efficient --k 9  --num_conv 2  --dataset 2DCUHK
```

- MSMT-17

**OG-Net**
```bash
python train_M.py --batch-size 16 --name Efficient_MSMT_ALL_Dense_b16_lr3_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e800_wa0.9_balance --slim 0.5 --flip --scale  --lrRate 3e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 800  --feature_dims 64,128,256,512    --efficient --dataset 2DMSMT  --id_skip 0  --wa --wa_start 0.9 --balance
```

**OG-Net-Small**
```bash
python train_M.py --batch-size 16 --name Data_Efficient_MSMT_ALL_SDense_b16_lr3_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_balance --slim 0.5 --flip --scale --lrRate 3e-4 --gpu_ids 0 --warm_epoch 10 --erase 0 --droprate 0.7 --use_dense --bg --adam --init 768 --cluster xyzrgb --train_all --num-epoch 1000 --feature_dims 48,96,192,384 --efficient --dataset 2DMSMT --id_skip 0 --balance
```

**OG-Net-Deep**
```bash
python train_M.py --batch-size 32 --name MSMT_Efficient_ALL_2SDDense_b32_lr3_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 3e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg   --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --norm_layer bn --efficient --k 9  --num_conv 2  --dataset 2DMSMT --balance
```

## Evaluation
- Market-1501
```bash 
python test_M.py  --name  Efficient_ALL_SDense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000 
```

- DukeMTMC-reID
```bash 
python test_M.py  --data 2DDuke --name   Efficient_Duke_ALL_SDense_b16_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000
```

- MSMT-17
```bash 
python test_MSMT.py  --name MSMT_Efficient_ALL_2SDDense_b32_lr3_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance
```

## Pre-trained Models
Since OG-Net is really small, I has included trained models in this github repo `./snapshot`. 

## Results
### [Person Re-ID Performance]
| Model name         | Market  | Duke | MSMT| 
| ------------------ |---------------- | -------------- | -------------- |
| OG-Net-Small |  85.90(66.93) |  75.67(55.72)     |   46.67(22.24)   | 
| OG-Net   |    86.19(68.09)  |   76.93(57.20) |  47.82(22.82)    |

### [ModelNet Performance] 
I add OG-Net code to https://github.com/layumi/dgcnn  
Results on ModelNet are 93.3 Top1 Accuracy / 90.5 MeanClass Top1 Accuracy.


## Citation
You may cite it in your paper. Thanks a lot.
```bibtex
@article{zheng2020person,
  title={Person Re-identification in the 3D Space},
  author={Zhedong Zheng, Yi Yang},
  journal={arXiv 2006.04569},
  year={2020}
}
```

## Related Work
We thank the great works of hmr, DGL, DGCNN and PointNet++. You may check their code at
- https://github.com/akanazawa/hmr
- https://github.com/dmlc/dgl/tree/master/examples/pytorch/pointcloud
- https://github.com/WangYueFt/dgcnn
- https://github.com/erikwijmans/Pointnet2_PyTorch

The baseline models used in the paper are modified from:
- https://github.com/layumi/Person_reID_baseline_pytorch

## Acknowledge
I would like to thank the helpful comments and suggestions from Yaxiong Wang, Yuhang Ding, Qian Liu, Chuchu Han, Tianqi Tang, Zonghan Wu and Qipeng Guo.
