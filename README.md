# kaggle-feedback-prize-effectiveness, part of 2nd place solution
This is the training code of a part of 2nd place solution for the Kaggle competition, Feedback Prize - Predicting Effective Arguments.

Team's solution summary : https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347359  
Inference code : TBD

## solution overview
![part_of_solution_Tom](https://user-images.githubusercontent.com/10670302/186792541-904bfbf8-0ca6-4c1b-9222-7a6e089c6108.png)

## HARDWARE
Ubuntu 18.04 LTS (2TB boot disk)  
Core i7-10700K  
64GB memory  
1 x NVIDIA GeForce RTX3090  

## SOFTWARE
Python 3.7.3  
CUDA 11.0  
cuddn 8.4.0  
nvidia drivers v.460  


## Usage
1. data preparation for kaggle train data  
```
cd src/00_EDA/00_v2_07/
```
run feedback2-playground.ipynb

2. data preparation for 2021 competition's data  
```
cd src/00_EDA/00_v1_13/
```
run feedback2-playground.ipynb
