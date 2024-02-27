# Polar Collision Grid pedestrian trajectory prediction model

This repository contains the code for our paper titled "Polar Collision Grids: Effective Interaction Modelling for Pedestrian Trajectory Prediction in Shared Space Using Collision Checks",  which was published and presented at ITSC 2023. The complete paper is accessible via the [IEEE portal](https://ieeexplore.ieee.org/abstract/document/10422509) or [ArXiv](https://arxiv.org/abs/2308.06654).


<div style="display: inline-block;">
    <img src="https://github.com/Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction/blob/master/figure/RelatedInfo.png" alt="Related Info" width="300" hspace="50"> 
   <img src="https://github.com/Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction/blob/master/figure/OveralFramework.png" alt="Overall Framework" width="500">
</div>



## Setup

Create a conda environmnet using python version 3.9, and install the required python packages
```bash
conda create --name PCG python=3.9
conda activate PCG
pip install -r requirements.txt
```
Install pytorch version 2.2.1 using the instructions [here](https://pytorch.org/get-started/locally/)

## Overview

This repository contains the code for our Polar Collision Grid model (`model_collisionGrid.py`), along with the baseline models used for comparison in our paper. These baselines consist of the Social LSTM model (with relevant python files in `model_SocialLSTM.py` and `grid.py`), the Vanilla LSTM model (`model_Vanilla_LSTM.py`), and the Linear Regression model (`LinearRegression.py`). The selection of the method for training and testing is determined by the `method` argument in both train.py and test.py.

## Dataset

The HBS dataset, which includes trajectories of both pedestrians and vehicles collected from a shared space, is used for training and testing our data-driven trajectory prediction model. The initial dataset, stored as `hbs.csv` in the `Data` folder, undergoes preprocessing. This involves adapting it to a format compatible with our code, utilizing the functions available in the `datalader.py` file. Subsequently, the data is partitioned into train and test sets, which are already stored in the `Data` folder.

## Model training

Our Polar Collision Grid model, as well as any of the data-driven baseline models (Social LSTM and Vanilla LSTM), can be trained for 200 epochs by executing the `train.py` file. A log file containing progress information during training will be stored under `Store_Results\log`. Following the completion of each epoch, the model will be saved in the `Store_Results\model` directory. Additionally, a plot illustrating the average displacement error and Negative Log Likelihood loss over epochs, and a more detailed one over batch numbers, will be saved in the `Store_Results\plot\train\` directory.


<div style="display: inline-block;">
    <img src="https://github.com/Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction/blob/master/figure/loss_plot_epoch.png" alt="Related Info" width="500" hspace="50"> 
</div>

## Model evaluation

Our trained models are stored in folders named relative to their corresponding models under the `Store_Results\model` directory. By executing the `test.py` script, the model saved in these directories will be loaded and tested on the test set. Depending on the chosen method, the `epoch` argument associated with the saved model in its folder should be adjusted. The terminal will display the performance of the saved model for the defined evaluation metrics, and the outputted trajectories for the test set will be saved as a `test_result.pkl` file in the `Store_Results/plot/test` directory.

For the results reported in the paper, the test files are stored in the method's associated folder within the `Store_Results/plot/test` directory. These files can be utilized to run the `visualization.py` script and generate the table results in the paper by executing the `TableResults.py` file for the selected method.


## Visualization
The predicted trajectories can be visualized for selected samples in the test set by running the `visualization.py` file. The resulting figures are stored in the `Store_Results\plot\test\plt\compare` directory.

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
