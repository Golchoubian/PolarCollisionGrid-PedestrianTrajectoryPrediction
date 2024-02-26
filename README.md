# Polar Collision Grid pedestrian trajectory prediction model

This repository contains the code for our paper titled "Polar Collision Grids: Effective Interaction Modelling for Pedestrian Trajectory Prediction in Shared Space Using Collision Checks",  which was published and presented at ITSC 2023. The complete paper is accessible via the [IEEE portal](https://ieeexplore.ieee.org/abstract/document/10422509) or [ArXiv](https://arxiv.org/abs/2308.06654).

## Setup

Create a conda environmnet using python version 3.9, and install the required python packages
```bash
conda create --name PCG python=3.9
conda activate PCG
pip install -r requirements.txt
```
Install pytorch version 2.2.1 using the instructions [here](https://pytorch.org/get-started/locally/)

## Overview

describe the files


## Dataset

## Model training

## Model evaluation


## Visualization

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{golchoubian2023polar,
  title={Polar Collision Grids: Effective Interaction Modelling for Pedestrian Trajectory Prediction in Shared Space Using Collision Checks},
  author={Golchoubian, Mahsa and Ghafurian, Moojan and Dautenhahn, Kerstin and Azad, Nasser Lashgarian},
  booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={791--798},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgment
This project is builds upon the codebase from social-lstm repsitory,
developed by [quancore](https://github.com/quancore/social-lstm) as a pytorch implementation of the Social LSTM model proposed by [Alahi et al](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf).
The Social LSTM model itself is also used as a baseline for comparison with our propsed CollisionGrid model.
