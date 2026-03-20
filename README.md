# NaturalScenesClassification

## Project Overview

This project uses deep learning to classify cheat X-ray images into three classes: COVID-19, Normal, and Pneumonia, We implement 2 MLP variants, 2 CNN variants, and a KNN classifier with and without PCA, tp evaluate how model architecture and depth affect classification performance. 

## Dataset

Our dataset is availible to the public on Kaggle: 

https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images/data

After copying the repository the data should be included but if any errors occur, you can download the dataset and place it in a folder called `data/` in the same directory as the notebook.

## Required Dependencies

Install all of the required dependencies by running: 

`pip install torch torchvision numpy matplotlib scikit-learn`

## Setting Up

1. Clone or download this repository
2. Download the dataset from the link if not included and place it in the data/ folder, the folder structure should look like this:
   `
project/
├── main.ipynb
└── data/
    ├── COVID/
    ├── NORMAL/
    └── PNEUMONIA/`
   
4. Install the required dependencies
5. Open the Jupyter Notebook labeled main.ipynb
6. Run all cells from top to bottom

## How to train the model

All of the model training is handled inside of the Jupyter Notebook main.ipynb
 ### MLP Models
 We define 2 MLP models labeled MLP1 (2 hidden layers) and MLP2 (3 hidden layers)
 One cell defined the models, one cell defines the training function, and then we have 1 cell for each model that trains the model for 20 epochs and prints the loss and accuracy. 

 ### CNN Models
 We define 2 CNN models labeled CNN1 (2 hidden layers) and CNN2 (3 hidden layers)
 After the MLP trainiing, testing and evaluation is complete, we move to the CNN section where one cell defines the models, one cell defines the training function, and then we have 1 cell for each model that trains the model for 20 epochs and prints the loss and accuracy. 

 ### KNN Models
 One cell extracts the features from the data loaders, then we train 2 KNN with k values of 1, 3, 5, 10, 20. 50 with and without PCA


ALL MODELS USE A FIXED RANDOM SEED OF 0 FOR REPRODUCIBILITY

## How to evaluate the model

 ### MLP Models
 After the cell where we trained the models, we have one cell that defines the evaluation function and one that evaluates both MLP one and MLP two on the test set and prints the test loss, test, accuracy, and runtime

 ### CNN Models
 After the cell where we trained the models, we have one cell that defines the evaluation function and one that evaluates both MLP one and MLP two on the test set and prints the test loss, test, accuracy, and runtime

 ### Confusion Matrices and per class metrics
 Once we have printed the above information for both models, the following cell cells, generate the confusion, matrixes, and classification report for each model

## Expected Outputs

When you run all of the cells in the notebook, you should see the following:


### Data Loading
`Train: 3659 | Test: 1569`


### Training output per epoch (example)
`Epoch 1/20 | Loss: 0.5300 | Acc: 0.8700
Epoch 2/20 | Loss: 0.4800 | Acc: 0.9100
...
Epoch 20/20 | Loss: 0.3500 | Acc: 0.9800`


### Test evaluation output (example)
`
MLP1 Test Loss: 0.XXXX | MLP1 Test Acc: 0.XXXX
MLP2 Test Loss: 0.XXXX | MLP2 Test Acc: 0.XXXX
CNN1 Test Loss: 0.XXXX | CNN1 Test Acc: 0.XXXX
CNN2 Test Loss: 0.XXXX | CNN2 Test Acc: 0.XXXX
`


### Charts generated:

MLP1 vs MLP2 Loss per Epoch
MLP1 vs MLP2 Accuracy per Epoch
CNN1 vs CNN2 Loss per Epoch
CNN1 vs CNN2 Accuracy per Epoch
Confusion matrix for each of the 4 models
KNN accuracy and inference time charts
