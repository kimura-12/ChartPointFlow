# Topology-Aware Flow-based Point Cloud Generation

This repository contains a PyTorch implementation of ChartPointFlow on 2D synthetic datasets and 3D point cloud  datasets.

## Introduction
  <div align="center">
  <img src="https://github.com/kimura-12/CPF_edit/blob/master/assets/sequence.jpg" width=65%>
  </div>

  Point clouds have attracted attention as a representation of an object's surface.
  Deep generative models have typically used a continuous map from a dense set in a latent space to express their variations.
  However, a continuous map cannot adequately express the varying numbers of holes.
  That is, previous approaches disregarded the topological structure of point clouds.
  Furthermore, a point cloud comprises several subparts, making it difficult to express it using a continuous map.
  This paper proposes ChartPointFlow, a flow-based deep generative model that forms a map conditioned on a label.
  Similar to a manifold chart, a map conditioned on a label is assigned to a continuous subset of a point cloud .
  Thus, ChartPointFlow is able to maintain the topological structure with clear boundaries and holes, whereas previous approaches generated blurry point clouds with fuzzy holes.
  The experimental results show that ChartPointFlow achieves state-of-the-art performance in various tasks, including generation, reconstruction, upsampling, and segmentation.
## Requirements

- python==3.6.10
- pytorch==1.0.1
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)==0.0.1
- gcc

## Datasets

Please use this [link](https://github.com/stevenygd/PointFlow) and follow the instructions to download ShapeNetCore dataset (version2).

## Training

For 3D practical datasets, run `python train.py --help` to show the options in details.

To reproduce the results on the airplane category, run

```bash
# with a single GPU
CUDA_VISIBLE_DEVICES=0 scripts/train.sh
# with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_dist.sh
```

For 2D synthetic datasets, run

```bash
CUDA_VISIBLE_DEVICES=0 python train_toy.py --data {circle,2sines,double_moon,four_circle} --y_dim {4,4,2,8}
```

## Pretrained models

Pretrained models can be downloaded from this [link](https://drive.google.com/drive/folders/1PLCi8e4QxTAqJgsaLHDHZpCQiEX-USlR?usp=sharing).

## Evaluation

```bash
# For preparation,
cd metrics/pytorch_structural_losses/
make clean
make
# To evaluate the model,
CUDA_VISIBLE_DEVICES=0 scripts/evaluate.sh
```

## Copyrights

- Most codes are obtained from PointFlow: https://github.com/stevenygd/PointFlow.
- models/AF.py is obtained from SoftFlow: https://github.com/ANLGBOY/SoftFlow.
