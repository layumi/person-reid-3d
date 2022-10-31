# Person Re-id in the 3D Space

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/layumi/person-reid-3d/blob/master/imgs/demo-1.jpg)

Thanks for your attention. In this repo, we provide the code for the paper [[Parameter-Efficient Person Re-identification in the 3D Space ]](https://arxiv.org/abs/2006.04569), published at IEEE Transactions on Neural Networks and Learning Systems (TNNLS) 2022.

## News
- **29 Sep 2022.** I updated Circle loss, parameter count and the latest snapshots trained on 4 datasets, including Market, Duke, CUHK and MSMT, in `/snapshots`. You can directly test it after dataset preparing. 
- **31 Jul 2021.** Circle loss is added. For the fair comparison with circle loss, I re-train almost all the models with a bigger batch size. The results are updated in the latest arXiv version.
 
- **30 Oct 2020.** I simply modify code on three points to further improve the performance: 

1. More training epochs help; (Since we are trained from scratch)

2. I replace the dgl to more efficient KNN implementation to accelebrate training; (DGL does not optimize KNN very well, and Matrix Multiplication works quicker. ) 

3. For MSMT-17 and Duke, some classes contain too many images, while other categories are under-explored. I apply the stratified sampling (`--balance`), which takes training samples of each class with equal probability.

- You may directly download my generated 3D data of the Market-1501 dataset at [[OneDrive]](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EQXEskhdd3xPjdFRxAUtB9cB7RkjAdzS5YXRH8QIf_TWAw?e=IhqNpi) or [[GoogleDrive]](https://drive.google.com/file/d/1ih-LrkdGUvNK3rEUNJIq4LTvcgVOXnnM/view?usp=sharing), and therefore you could skip the data preparation part.
Just put the datasets in the same folder of the code.
```
├── 2DMarket\
│   ├── query/  
│   ├── train_all/
│   ├── ...
├── 3DMarket+bg\
│   ├── query/  
│   ├── train_all/
│   ├── ...
├── train.py
├── test.py 
├── ...
```

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
conda install matplotlib requests
conda install -c dglteam dgl-cuda10.1=0.4.3
pip install -r requirements.txt
```

If you face any error, you may first try to re-install open3d. It helps. 
And make sure the gcc version is larger than 5.4.0. If you do not have the sudo permission, you may install gcc by conda as follows: 
```
conda install -c brown-data-science gcc          (which is gcc-5.4.0)
gcc -v                                          (to see whether installation is successful)
ln libstdc++.so.6.0.26 libstdc++.so.6            (update lib in /anaconda3/env/OG/lib)
conda install gxx_linux-64
conda install gcc_linux-64
```

## Prepare Data 
- **Download our prepared data.** You may directly download my generated 3D data of the Market-1501 dataset at [[OneDrive]](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EQXEskhdd3xPjdFRxAUtB9cB7RkjAdzS5YXRH8QIf_TWAw?e=IhqNpi) or [[GoogleDrive]](https://drive.google.com/file/d/1ih-LrkdGUvNK3rEUNJIq4LTvcgVOXnnM/view?usp=sharing), and therefore you could totally skip the next data preparation part.

- **3D Part** We first generate the 3D data via the code at https://github.com/layumi/hmr 
(I modified the code from https://github.com/akanazawa/hmr and added 2D-to-3D color mapping.) 

**I remove all 3D faces and only keep 3D points positions&RGB to save the storage & loading burden. You can use any text readers (such as vim) to see my generated obj files.**

- **2D Part** Download Market-1501, DukeMTMC-reID or MSMT17 and unzip them in the `../`

Split the dataset and arrange them in the folder of ID by the following code.
```bash 
python prepare_market.py # You may need to change the download path. 
python prepare_duke.py
python prepare_MSMT.py
```

Link the 2DDataset to this dir.
```bash 
ln -s ../Your_Market/pytorch  ./2DMarket
ln -s ../Your_Duke/pytorch  ./2DDuke
ln -s ../Your_MSMT/pytorch  ./2DMSMT
```

## Training 
- 1. Market-1501

**OG-Net** 86.82 (69.02)
```bash
python train_M.py --batch-size 36 --name Efficient_ALL_Dense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_wa0.9_GeM_bn2_class3_amsgrad --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1500  --feature_dims 64,128,256,512   --efficient  --wa --wa_start 0.9 --gem --norm_layer bn2   --amsgrad --class 3
```

**OG-Net + Circle** 87.80 (70.56)
```bash
python train_M.py --batch-size 36 --name Efficient_ALL_Dense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_wa0.9_GeM_bn2_balance_circle_amsgrad_gamma64 --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1500  --feature_dims 64,128,256,512   --efficient  --wa --wa_start 0.9 --gem --norm_layer bn2 --balance  --circle --amsgrad --gamma 64
```

**OG-Net-Small** 86.79 (67.92)
```bash
python train_M.py --batch-size 36 --name Efficient_ALL_SDense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_wa0.9_GeM_bn2_balance_amsgrad --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384   --efficient  --wa --wa_start 0.9 --gem --norm_layer bn2 --balance  --amsgrad 
```

**OG-Net-Small + Circle** 87.38 (70.48)
```bash
python train_M.py --batch-size 36 --name Efficient_ALL_SDense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_wa0.9_GeM_bn2_balance_circle_amsgrad_gamma64 --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1500  --feature_dims 48,96,192,384   --efficient  --wa --wa_start 0.9 --gem --norm_layer bn2 --balance  --circle --amsgrad --gamma 64
```

**OG-Net-Deep + Circle** 88.81 (72.91)
```bash
python train_M.py --batch-size 30 --name Market_Efficient_ALL_2SDDense_b30_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DMarket --balance --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

- 2. DukeMTMC-reID

**OG-Net-Small** 77.33 (57.74)
```bash
python train_M.py --batch-size 36 --name reEfficient_Duke_ALL_SDense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_class_GeM_bn2_amsgrad --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384   --efficient --dataset 2DDuke --class --wa --wa_start 0.9 --gem --norm_layer bn2  --amsgrad 
```

**OG-Net-Small + Circle** 77.15 (58.51)
```bash
python train_M.py --batch-size 36 --name reEfficient_Duke_ALL_SDense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_balance_GeM_bn2_circle_amsgrad --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384   --efficient --dataset 2DDuke --balance --wa --wa_start 0.9 --gem --norm_layer bn2 --circle --amsgrad
```


**OG-Net** 76.53 (57.92)
```bash
python train_M.py --batch-size 36 --name reEfficient_Duke_ALL_Dense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_class1_GeM_bn2_amsgrad --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512   --efficient --dataset 2DDuke --class 1 --wa --wa_start 0.9 --gem --norm_layer bn2 --amsgrad  
```

**OG-Net + Circle** 78.37 (60.07)
```bash
python train_M.py --batch-size 36 --name reEfficient_Duke_ALL_Dense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_balance_GeM_bn2_circle_amsgrad_gamma64 --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512   --efficient --dataset 2DDuke --balance --wa --wa_start 0.9 --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

**OG-Net-Deep** 76.97 (59.23)
```bash
python train_M.py --batch-size 36 --name Duke_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance_noCircle  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DDuke --balance --gem --norm_layer bn2 --amsgrad 
```

**OG-Net-Deep + Circle** 78.50 (60.7)
```bash
python train_M.py --batch-size 36 --name Duke_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DDuke --balance --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

- 3. CUHK-NP 

**OG-Net** 44.00 (39.28)
```bash
python train_M.py --batch-size 36 --name Efficient_CUHK_ALL_Dense_b36_lr8_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_class1_gem_bn2_amsgrad_wd1e-3 --slim 0.5 --flip --scale  --lrRate 8e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1 --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512    --efficient --dataset 2DCUHK --class 1  --gem --norm_layer bn2  --amsgrad  --wd 1e-3 
```

**OG-Net + Circle** 48.29 (43.73)
```bash
python train_M.py --batch-size 36 --name Efficient_CUHK_ALL_Dense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_class3_gem_bn2_circle_amsgrad_wd1e-3_gamma96 --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 64,128,256,512    --efficient --dataset 2DCUHK --class 3 --gem --norm_layer bn2 --circle --amsgrad --wd 1e-3 --gamma 96
```

**OG-Net-Small** 43.07 (38.06)
```bash 
python train_M.py --batch-size 36 --name Efficient_CUHK_ALL_SDense_b36_lr10_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_gem_bn2_amsgrad_wd1e-3_class1 --slim 0.5 --flip --scale  --lrRate 10e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam   --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384    --efficient --dataset 2DCUHK --gem --norm_layer bn2  --amsgrad --wd 1e-3  --class 1
```

**OG-Net-Small + Circle** 46.43 (41.79)
```bash
python train_M.py --batch-size 36 --name Efficient_CUHK_ALL_SDense_b36_lr8_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_balance_gem_bn2_circle_amsgrad_wd1e-3_gamma64 --slim 0.5 --flip --scale  --lrRate 8e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam   --init 768  --cluster xyzrgb  --train_all   --num-epoch 1000  --feature_dims 48,96,192,384    --efficient --dataset 2DCUHK --balance --gem --norm_layer bn2 --circle --amsgrad --wd 1e-3 --gamma 64
```

**OG-Net-Deep** 45.71 (41.15)
```bash
python train_M.py --batch-size 36 --name CUHK_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_id2_bn_k9_conv2_class3_Nocircle  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1500  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DCUHK --class 3 --gem --norm_layer bn2 --amsgrad 
```

**OG-Net-Deep + Circle** 49.43 (45.71)
```bash
python train_M.py --batch-size 36 --name CUHK_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 1500  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DCUHK --balance --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

- 4. MSMT-17

**OG-Net** 44.27 (21.57)
```bash
python train_M.py --batch-size 36 --name reEfficient_MSMT_ALL_Dense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_balance_GeM_bn2_circle_amsgrad_gamma64_Nocircle --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 64,128,256,512   --efficient --dataset 2DMSMT --balance --wa --wa_start 0.9 --gem --norm_layer bn2  --amsgrad 
```

**OG-Net + Circle** 45.28 (22.81)
```bash
python train_M.py --batch-size 36 --name reEfficient_MSMT_ALL_Dense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_balance_GeM_bn2_circle_amsgrad_gamma64 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 64,128,256,512   --efficient --dataset 2DMSMT --balance --wa --wa_start 0.9 --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

**OG-Net-Small** 43.84 (21.79)
```bash
python train_M.py --batch-size 36 --name reEfficient_MSMT_ALL_SDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_balance_GeM_bn2_circle_amsgrad_gamma64 --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 48,96,192,384   --efficient --dataset 2DMSMT --balance --wa --wa_start 0.9 --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```

**OG-Net-Small + Circle** 42.44 (20.31)
```bash
python train_M.py --batch-size 36 --name reEfficient_MSMT_ALL_SDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_class_GeM_bn2_amsgrad --slim 0.5 --flip --scale  --lrRate 6e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 48,96,192,384   --efficient --dataset 2DMSMT --class 1  --wa --wa_start 0.9 --gem --norm_layer bn2  --amsgrad 
```

**OG-Net-Deep** 44.56 (21.41) 
```bash
python train_M.py --batch-size 30 --name MSMT_Efficient_ALL_2SDDense_b30_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_id2_bn_k9_conv2_balance_nocircle  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DMSMT --balance --gem --norm_layer bn2 --amsgrad 
```

**OG-Net-Deep + Circle** 47.32 (24.07)
```bash
python train_M.py --batch-size 30 --name MSMT_Efficient_ALL_2SDDense_b30_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_id2_bn_k9_conv2_balance  --id_skip 2 --slim 0.5 --flip --scale  --lrRate 4e-4 --gpu_ids 0 --warm_epoch 10  --erase 0  --droprate 0.7   --use_dense  --bg 1  --adam  --init 768  --cluster xyzrgb  --train_all   --num-epoch 600  --feature_dims 48,96,96,192,192,384,384  --efficient --k 9  --num_conv 2  --dataset 2DMSMT --balance --gem --norm_layer bn2 --circle --amsgrad --gamma 64
```


## Evaluation
- Market-1501
```bash 
python test_M.py  --name  Market_Efficient_ALL_2SDDense_b30_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance
```

- DukeMTMC-reID
```bash 
python test_M.py  --data 2DDuke --name Duke_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1000_id2_bn_k9_conv2_balance
```

- CUHK
```bash 
python test_M.py  --data 2DCUHK --name CUHK_Efficient_ALL_2SDDense_b36_lr6_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e1500_id2_bn_k9_conv2_balance
```

- MSMT-17
```bash 
python test_MSMT.py  --name MSMT_Efficient_ALL_2SDDense_b30_lr4_flip_slim0.5_warm10_scale_e0_d7+bg_adam_init768_clusterXYZRGB_e600_id2_bn_k9_conv2_balance
```

## Pre-trained Models
Since OG-Net is really small, I has included trained models in this github repo `./snapshot`. 

If the model is trained on CUHK, Duke or MSMT, I will add dataset name in the model name, otherwise the model is trained on Market.

### [ModelNet Performance] 
I add OG-Net code to https://github.com/layumi/dgcnn  
Results on ModelNet are 93.3 Top1 Accuracy / 90.5 MeanClass Top1 Accuracy.


## Citation
You may cite it in your paper. Thanks a lot.
```bibtex
@article{zheng2022person,
  title={Parameter-Efficient Person Re-identification in the 3D Space},
  author={Zheng, Zhedong and Wang, Xiaohan and Zheng, Nenggan and Yang, Yi},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  doi={10.1109/TNNLS.2022.3214834},
  note={\mbox{doi}:\url{10.1109/TNNLS.2022.3214834}},
  year={2022}
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
