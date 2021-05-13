# QC-DGM
This is the official PyTorch implementation and models for our CVPR 2021 paper: [Deep Graph Matching under Quadratic Constraint](https://Zerg-Overmind.github.io/files/cvpr2021_Gao.pdf).

It also contains the configuration files to reproduce the results of qc-DGM_1 reported in the paper on Pascal VOC Keypoint and Willow Object Class dataset.

## Get started

1. pytorch (GPU version) >= 1.1 
2. ninja-build: ``apt-get install ninja-build``
3. python packages: ``pip install tensorboardX scipy easydict pyyaml``
4. Download dataset:
   1. Pascal VOC Keypoint:
     * Download and tar [VOC2011 keypoints](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html), and the path looks like: ``./data/PascalVOC/VOC2011``.
     * Download and tar [Berkeley annotation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz), and the path looks like: ``./data/PascalVOC/annotations``.
     * The train/test split of Pascal VOC Keypoint is available in: ``./data/PascalVOC/voc2011_pairs.npz``.
   2. Willow Object Class dataset:
     * Download and unzip [Willow ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip), and the path looks like: ``./data/WILLOW-ObjectClass``.

## Training

1. Run training and evaluation on Pascal VOC Keypoint:

   ``python train_eval.py --cfg ./experiments/QCDGM_voc.yaml``
   
   or you could replace the default ``./experiments/QCDGM_voc.yaml`` with path to your own configuration file.
2. Run training and evaluation on Willow Object Class dataset:

   ``python train_eval.py --cfg ./experiments/QCDGM_willow.yaml`` 
  
   or you could replace the default ``./experiments/QCDGM_willow.yaml`` with path to your own configuration file.
   
## Evaluation

1. Run evaluation on Pascal VOC Keypoint on epoch k:

   ``python eval.py --cfg ./experiments/QCDGM_voc.yaml --epoch k`` 

   or you could replace the default ``./experiments/QCDGM_voc.yaml`` with path to your own configuration file.
2. Run evaluation on Willow Object Class dataset on epoch k:  
    
   ``python eval.py --cfg ./experiments/QCDGM_willow.yaml --epoch k`` 
   
   or you could replace the default ``./experiments/QCDGM_voc.yaml`` with path to your own configuration file.
   
## Results and model zoo
We report the performance on Pascal VOC Keypoint and Willow Object Class datasets.

**Pascal VOC Keypoint**

|  method  | Download | aero | bike | bird | boat | bottle |  bus  | car  | cat  | chair | cow  | table | dog  | horse | mbike | person | plant | sheep | sofa | train |  tv  |   mean   |
| -------- | -------- | ---- | ---- | ---- | ---- | ------ | ----- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ---- | ----- | ---- | -------- |
|  qc-DGM  | [parameter](https://drive.google.com/file/d/1uiNstmYg_J9252ybbl0PKz_-qWB2qJMQ/view?usp=sharing)| 48.4 | 61.6 | 65.3 | 61.3 |  82.4  | 79.6 | 74.3 | 72.0 | 41.8 | 68.8 | 65.0 | 66.1 | 70.9 | 69.6 |  48.2  | 92.1 | 69.0 | 66.7 | 90.4 | 91.8 |  69.3  |

For the convenience of evaluation, our trained parameter file is also provided by BaiduYun [download](https://pan.baidu.com/s/1ODcbCUP2PyXXzHBEs70e1Q) link with extracting code **vocc**. Download the parameter file with path to ``./output/QCDGM_voc/params/`` and run evaluation on Pascal VOC Keypoint.

**Willow Object Class**

| method | Download | face | m-bike | car |  duck  | wbottle |  mean  |
| -------| -------- |------| ------ | --- | ------ | ------- | -------|
| qc-DGM | [parameter](https://drive.google.com/file/d/16jhOBpAEUREbqjxzjoW0KbsJkWOZfJ_i/view?usp=sharing) | 100.0 | 95.0 | 93.8 | 93.8 |  97.6 | 96.0 |  

For the convenience of evaluation, our trained parameter file is also provided by BaiduYun [download](https://pan.baidu.com/s/1vvdzjYzc2y2U0FCd2kvqdg) link with extracting code **will**. Download the parameter file with path to ``./output/QCDGM_willow/params/`` and run evaluation on Willow Object Class dataset. 
 
## Citation
```text
@InProceedings{Gao_2021_CVPR,
author = {Gao, Quankai and Wang, Fudong and Xue, Nan and Yu, Jin-Gang and Xia, Gui-Song},
title = {Deep Graph Matching under Quadratic Constraint},
booktitle = {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
year = {2021}
}
