# Stacked Autoencoder-based Compression of Point Cloud Geometry

Xuewei Cao, Wenbiao Zhou, Shuyu Yan, Genpei Liu

> **Abstract:** *Point clouds have gained widespread application in various fields, but their high resolution often results in large data volumes, posing challenges for storage, transmission, and processing.  Traditional 2D image or video compression methods are unsuitable due to the spatial irregularity and sparseness of point clouds.  Inspired by the effectiveness of autoencoders in visual analysis tasks and image compression, this paper proposes a novel stacked autoencoder-based geometry compression method for point clouds.  By transforming point clouds into Morton codes using a linear octree structure and further encoding them into integer sequences, the proposed method leverages stacked autoencoders to reduce the dimensions of these sequences, achieving both high reconstruction quality and high compression ratios.  Experimental results demonstrate that our method outperforms many other geometry compression methods, especially for small-size point clouds.  By increasing the coding depth of the linear octree, our approach can even achieve lossless compression results, showcasing its potential as an effective geometry compression technique for point clouds.* 
<hr />

## Compression Architecture
<img src = "compression_architecture.png">

## Installation
1. Clone our repository
```
git clone https://github.com/XueweiCao/SAE-GPCC.git
cd SAE-GPCC
```
2. Make conda environment
```
conda create -n sae_gpcc python = 3.10
conda activate sae_gpcc
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Train
1. Put datasets in ./datasets/train/
2. Train models
```
python ./train.py
```
3. Train results will be saved in ./run/train/

## Test
1. Put trained models in ./model/option/
2. Put datasets in ./datasets/test/
3. Test datasets
```
python ./test.py
```
4. Test results will be saved in ./run/test/
