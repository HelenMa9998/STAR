import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint
import random
from data import Data
from utils import get_dataset, get_net, get_strategy, get_handler
from config import parse_args
from seed import setup_seed
# from visualization import visualiazation
import pdb
args = parse_args()
pprint(vars(args))
print()

# fix random seed
setup_seed()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load dataset
X_train, Y_train, X_val, Y_val, X_test, Y_test, train_num_slices_per_patient, val_num_slices_per_patient, test_num_slices_per_patient = get_dataset(args.dataset_name,supervised=False)
dataset = Data(X_train, Y_train, X_val, Y_val, X_test, Y_test) 

train_handler = get_handler(args.dataset_name, train=True) # two slices for train
prop_handler = get_handler(args.dataset_name, prop=True) # group slices for propagation 
eval_handler = get_handler(args.dataset_name) # single slice for val and test

# start experiment
# initilalize labeled pool 
init_num, non_blank_idx = dataset.initialize_labels_K(train_num_slices_per_patient, args.k_init_labeled) 
print("init_num",init_num)
print(f"number of labeled pool: {init_num}")
print(f"number of unlabeled pool: {dataset.n_pool-init_num}")
print(f"number of testing pool: {dataset.n_test}")
print()

# load prop network
prop_net = get_net(args.dataset_name, device, prop=True) 

# load query strategy
strategy = get_strategy(args.strategy_name)(dataset, prop_net) 

# Round 0 start
print("Round 0")
rd = 0
accuracy = []
size = []
query_samples = []

# define dataloader 
_, norm_train_loader = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
_, prop_train_loader = dataset.get_data(pseudo_idxs = None, k = 2, train_num_slices_per_patient = train_num_slices_per_patient, handler = train_handler)
norm_val_data = dataset.get_val_data(eval_handler)
_, prop_val_loader = dataset.get_prop_val_data(k = 2, val_num_slices_per_patient = val_num_slices_per_patient, handler = train_handler)
test_loader = dataset.get_test_data(eval_handler)
_, prop_loader = dataset.get_data(pseudo_idxs = None, k = 11, train_num_slices_per_patient = train_num_slices_per_patient, handler = prop_handler)

# round 0 training
# train encoder and decoder (normal supervised training)
prop_net.supervised_val_loss(norm_train_loader,norm_val_data,rd)
if args.method_name == "our_method": 
    # train key val encoder
    prop_net.prop_train(prop_train_loader,prop_val_loader,rd) 
    # use prop_net prop the info → generate pseudo label
    pseudo_idxs, pseudo_label, query_idxs = prop_net.prop(prop_loader)#这可能又不一样了


# Round 0 test 
test_preds,targets = strategy.predict(test_loader,test_num_slices_per_patient)
testing_accuracy = dataset.cal_test_acc(test_preds, targets)
print(f"Round 0 testing accuracy: {testing_accuracy}")  # get model performance for test dataset
accuracy.append(testing_accuracy)
size.append(init_num)

# active learning cycle begins
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")
    # AL query
    if args.method_name == "AL_baseline":
        query_idxs = strategy.query(args.n_query, eval_handler, args.method_name) 

    elif args.method_name == "AL_Semi_baseline":
        query_idxs, pseudo_idxs, pseudo_label = strategy.query(args.n_query, eval_handler, args.method_name) 
        dataset.update_pseudo_label(pseudo_idxs, pseudo_label) 
        _, norm_train_loader = dataset.get_labeled_data(eval_handler, pseudo_idxs=pseudo_idxs)

    # update labeled data pool
    strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning

    if args.method_name == "our_method":
        # lable propagation
        pseudo_label = {idx: label for idx, label in zip(pseudo_idxs, pseudo_label)}
        pseudo_idxs = list(set([i for i in pseudo_idxs if i not in query_idxs]))
        print("same",len(pseudo_idxs))
        pseudo_label = {key: pseudo_label[key] for key in pseudo_idxs}
        dataset.update_pseudo_label(pseudo_idxs,pseudo_label) 

        # get dataloader with new pseudo label
        _, norm_train_loader = dataset.get_labeled_data(eval_handler, pseudo_idxs=pseudo_idxs)
        _, prop_train_loader = dataset.get_data(pseudo_idxs = pseudo_idxs, k = 2, train_num_slices_per_patient = train_num_slices_per_patient, handler = train_handler)
        _, prop_loader = dataset.get_data(pseudo_idxs = pseudo_idxs, k = 11, train_num_slices_per_patient = train_num_slices_per_patient, handler = prop_handler)
    
    if args.method_name == "AL_baseline":
        _, norm_train_loader = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
        
    #train
    prop_net.supervised_val_loss(norm_train_loader,norm_val_data,rd)

    if args.method_name == "our_method":
        prop_net.prop_train(prop_train_loader,prop_val_loader,rd)#只根据labeled data去学习encoder、key val encoder、decoder
        pseudo_idxs, pseudo_label, query_idxs = prop_net.prop(prop_loader) 

    # calculate accuracy
    test_preds,targets = strategy.predict(test_loader,test_num_slices_per_patient) # get model prediction for test dataset
    testing_accuracy = dataset.cal_test_acc(test_preds, targets)
    print(f"Round {rd} testing accuracy: {testing_accuracy}")
    accuracy.append(testing_accuracy)
    labeled_idxs, _ = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
    size.append(len(labeled_idxs))

# save the result
dataframe = pd.DataFrame(
    {'model': 'Unet', 'Method': args.strategy_name, 'Training dataset size': size, 'Accuracy': accuracy})
dataframe.to_csv(f"./{args.strategy_name}.csv", index=False, sep=',')