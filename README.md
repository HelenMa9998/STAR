## STAR
### Introduction
This project is the code for paper Spatio-temporal Memory Aggregation for 3D Medical Image Segmentation with Semi-supervised Active Learning based on python and pytorch framework.
  

### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.8.5                    
pytorch                   1.10.1         
torchvision               0.11.2         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.6.2            
numpy                     1.19.2        
opencv                    4.4.0         
pandas                    1.4.4              
scikit-learn              1.2.0               
```  

The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment firstly.

---  
## Usage 
### 1) Download Project 
The project structure and intention are as follows : 
```
STEMA			                                                  # Source code		
    ├── seed.py			 	                                      # Set up random seed
    ├── query_strategies		                                  # All query strategies
    │   ├── margin_sampling.py                                    # Marginal query method
    │   ├── hybrid_sampling.py                                    # hybrid query method, integrating Bayesian methods with high density and diversity selection
    │   ├── kcenter_greedy.py                                     # Coreset query method
    │   ├── bayesian_active_learning_disagreement_dropout.py	  # Deep Bayesian query method
    │   ├── entropy_sampling.py		                              # Entropy-based query method
    │   ├── entropy_sampling_dropout.py		                      # Entropy-based MC dropout query method
    │   ├── random_sampling.py		                              # Random selection
    │   ├── strategy.py                                           # Functions needed for query strategies
    ├── data.py	                                                  # Prepare the dataset & initialization and update for training dataset
    ├── handlers.py                                               # Get data loader for the dataset
    ├── main.py			                                          # An example of code utilization, including the whole process of active learning
    ├── nets.py		                                              # Training models and methods needed for the query method
    ├── supervised_baseline.py	                                  # An example of supervised learning training process
    └── utils.py			                                      # Important setups including network, dataset, hyperparameters...
```
### 2) Datasets preparation 
1. Download the datasets from the official address:
   
   BraTS 2019 Dataset: https://www.med.upenn.edu/cbica/brats2019/data.html
   
   Medical Segmentation Decathlon Dataset: http://medicaldecathlon.com/

2. Modify the data folder path for specific dataset in `data.py`

### 3) Run Active learning process 
Please confirm the configuration information in the [`utils.py`]
```
  python main.py \
      --n_round 16 \
      --n_query 100 \
      --dataset_name MSSEG \
      --method_name our_method\
      --training_method supervised_train_acc \
      --seed 42
```
The training results will be saved to the corresponding directory(save name) in `performance.csv`.  
You can also run `supervised_baseline.py` by
```
python supervised_baseline.py
```

## Visualization
1 Active learning performance visualization  
After you got the `performance.csv`, you can run `visualization.py` to visualize the whole process

