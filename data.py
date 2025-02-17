from operator import index
import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import torch.nn as nn


from seed import setup_seed
from data_func import *

import numpy as np
setup_seed()

# 3D sice coefficient
def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    # dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = 1
        logits[logits>=0.5] = 1
        logits[logits<0.5] = 0
        logits = logits.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * intersection + self.epsilon / ((logits + targets).sum(-1) + self.epsilon)
#         dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
#         print(dice_score)
        return dice_score
    
class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        self.X_train = X_train # used for maintaining original label
        self.Y_train = Y_train
        self.Y_train_pseudo = Y_train # used for pseudo training
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test 
        
        # self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.unlabeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        # use idx to distinguish labeled and unlabeled data取1000张有target的sample
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        count = 0
        for i in tmp_idxs:
            if np.sum(self.Y_train[i])!=0:
                self.labeled_idxs[i] = True
                count+=1
                if count == num:
                    break

    def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        start_idx = 0
        non_blank_idx = []

        # print("去除空白前X_train",self.X_train.shape) #(7750, 1, 240, 240)
        for i in range(len(num_slices_per_patient)):#([512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 336, 336, 336, 336, 336])
            num_slices = num_slices_per_patient[i]
            num_full_segments = (num_slices // k)+1 # 每个病人多少初始化slice 25 
            last_segment_size = num_slices % k # 每个病人剩余多少slice 12 
            # print("num_full_segments",num_full_segments)
            selected_slices = [start_idx + k*j for j in range(num_full_segments)]#[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480]
            # print("selected_slices",selected_slices)
            start_idx += num_slices
            self.labeled_idxs[selected_slices] = True
            # print("selected_slices",selected_slices)
            # print("num_slices_per_patient",num_slices_per_patient)
            for j in range(len(selected_slices)-1): #[0, 30, 60, 90, 120, 150]   0 
                
                if np.sum(self.Y_train[selected_slices[j]])==0 and np.sum(self.Y_train[selected_slices[j+1]])!=0:
                    start_blank_idx = selected_slices[j]
                if np.sum(self.Y_train[selected_slices[j]])!=0 and np.sum(self.Y_train[selected_slices[j+1]])==0:
                    end_blank_idx = selected_slices[j+1]
            # print(start_blank_idx,end_blank_idx)
            non_blank_idx.extend(range(start_blank_idx, end_blank_idx+1))
        return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]),non_blank_idx

    def delete_black_slices(self, index):
        self.X_train = self.X_train[index]
        self.Y_train = self.Y_train[index]
        self.labeled_idxs = self.labeled_idxs[index]
        self.n_pool = len(self.X_train)
        print("去除空白后X_train",self.X_train.shape)
        print(len(self.labeled_idxs))
        print("labeled",self.X_train[self.labeled_idxs].shape)
        print(len(self.labeled_idxs))
    
    # def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
    #     self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
    #     start_idx = 0
    #     cumulative_counts = []
    #     cumulative_sum = 0

    #     for num_slices in num_slices_per_patient:
    #         num_full_segments = (num_slices // k) + 1
    #         last_segment_size = num_slices % k
    #         selected_slices = [start_idx + k*j for j in range(num_full_segments)]
    #         start_idx += num_slices
    #         # selected_slices.append(start_idx-1)  # Add the last slice
    #         self.labeled_idxs[selected_slices] = True #[0, 20, 40, 60, 80, 100, 120, 140]
    #         # print("selected_slices",selected_slices)
    #     # for count in num_slices_per_patient:
    #     #     cumulative_sum += count
    #     #     cumulative_counts.append(cumulative_sum)
    #     # print("num_slices_per_patient",cumulative_counts)
    #     #     if last_segment_size > 0:
    #     #         last_selected_slices = [i+k*num_full_segments for i in range(last_segment_size)]
    #     #         self.X_train = self.X_train[~last_selected_slices]
    #     #         self.Y_train = self.Y_train[~last_selected_slices]
    #     # print("initialize_labels_K data",self.X_train.shape,self.Y_train.shape)

    #     return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]) # 8*50


    # def initialize_labels_K(self, interval): #病人slice 不能整除K
    #     tmp_idxs = np.zeros(self.n_pool, dtype=bool)
    #     for i in range(self.n_patients):
    #         start_idx = i * self.n_slices
    #         tmp_idxs[start_idx:start_idx+self.n_slices:interval] = True
    #     return len(tmp_idxs[tmp_idxs])

    # def get_labeled_data(self):
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     # print("labeled data", labeled_idxs.shape)
    #     # print("labeled_idxs ", labeled_idxs)
    #     return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")

    def get_prop_val_data(self, k, val_num_slices_per_patient, handler): 
        # get labeled data for training
        # print(len(self.n_pool))
        # print(len(self.labeled_idxs))

        labeled_idxs = np.arange(len(self.X_val), dtype=int)
        return labeled_idxs, handler(self.X_val, self.Y_val, labeled_idxs, k, val_num_slices_per_patient, mode="val")
    
    
    def get_labeled_data(self, handler, pseudo_idxs):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs].tolist()

        if pseudo_idxs != None:
            labeled_idxs.extend(pseudo_idxs) #把pseudo label加进去 进行采样
        labeled_idxs = np.array(labeled_idxs)
        # print("labeled data", labeled_idxs.shape)
        print("labeled_idxs_normal", len(labeled_idxs))
        return labeled_idxs, handler(self.X_train[labeled_idxs], self.Y_train_pseudo[labeled_idxs],mode="train")#Y_train_pseudo还是Y_train
    
    def get_data(self, pseudo_idxs, k, train_num_slices_per_patient, handler): 
        # get labeled data for training
        # print(len(self.n_pool))
        # print(len(self.labeled_idxs))

        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs].tolist()
        print("normal",len(labeled_idxs))

        if pseudo_idxs != None:
            labeled_idxs.extend(pseudo_idxs) #把pseudo label加进去 进行采样
        labeled_idxs = np.array(labeled_idxs)
        # print(len(pseudo_idxs))
        print("pseudo",len(labeled_idxs))
        return labeled_idxs, handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)

    # def get_data(self, pseudo_idxs, k, train_num_slices_per_patient, handler): 
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     if pseudo_idxs != None:
    #         labeled_idxs = np.concatenate((np.arange(self.n_pool, dtype=int)[self.labeled_idxs], pseudo_idxs), axis=0)
    #     print(len(labeled_idxs))
    #     return labeled_idxs, handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)


    # used for pseudo label filter remove blank patches
    def delete_black_patch(self, index, preds):
        black_index = []
        for i in range(preds.shape[0]):#24537
            idx = preds[i]
            index[i]
            pred = (preds[i][1] > 0.5).int()
            if torch.sum(pred)==0:
                black_index.append(idx)
        return black_index

    # def get_unlabeled_data(self, index=None): #index是空白patch
    #     # get unlabeled data for active learning selection process
    #     unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
    #     # print("unlabeled_idxs",unlabeled_idxs.shape)
    #     if index!=None:
    #         self.labeled_idxs[index] = True #5486
    #         unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
    #         self.labeled_idxs[index] = False
    #     return unlabeled_idxs
    
    def get_unlabeled_data(self, handler, rd=None, index=None): #index是空白patch
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        # print("unlabeled_idxs",unlabeled_idxs.shape)
        if index!=None:
            self.labeled_idxs[index] = True #5486
            unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
            self.labeled_idxs[index] = False
        # if rd ==8: 
            # print("get_unlabeled_data_x, get_unlabeled_data_y", self.X_train[unlabeled_idxs].shape, self.Y_train[unlabeled_idxs].shape)
        return unlabeled_idxs, handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],mode="val")

    def update_pseudo_label(self, idxs, label):
        # used for pseudo labeling, change the correct label to pseudo label要考虑后续可能又需要原本label了
        # self.X_train = torch.tensor(self.X_train)
        self.Y_train_pseudo = self.Y_train
        self.Y_train_pseudo[idxs] = torch.stack(list(label.values()), dim=0).squeeze()

    def get_train_data(self, handler):
        # get validation dataset if exist
        return handler(self.X_train, self.Y_train,mode="val")
        
    def get_val_data(self, handler):
        # get validation dataset if exist
        return handler(self.X_val, self.Y_val,mode="val")

    def get_test_data(self, handler):
        # get test dataset if exist
        return handler(self.X_test, self.Y_test,mode="val")

    # def cal_test_acc(self, logits, targets):
    #     # calculate accuracy for test dataset
    #     dscs = []
    #     for prediction, target in zip(logits, targets):
    #         dsc = cal_subject_level_dice(prediction, target, class_num=2)
    #         dscs.append(dsc)
    #         dice = np.mean(dscs)
    #     return dice

    def cal_test_acc(self, logits, targets):#(6200, 1, 240, 240)
        # calculate accuracy for test dataset
        dice_coeff = dice_coefficient()
        dscs = []
        # print("logits",logits.shape,"targets",targets.shape)
        for i in range(len(logits)):
            dsc = dice_coeff(targets[i],logits[i])
            dscs.append(dsc)
        dice = np.mean(dscs)
        return dice

    def cal_train_acc(self, preds):
        # calculate accuracy for train dataset for early stopping
        return 1.0 * (self.Y_train == preds).sum().item() / self.n_pool

    def add_labeled_data(self, data, label):
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        data = torch.reshape(data, (len(data),128,128))
        # data = torch.unsqueeze(data, 1)
        self.X_train = torch.tensor(self.X_train)#([25537, 128, 128])
        self.Y_train = torch.tensor(self.Y_train)
        self.X_train = torch.cat((self.X_train, data), 0)#([26037, 128, 128])
        self.Y_train = torch.cat((self.Y_train, label), 0)
        # print("labeled_idxs",self.labeled_idxs.shape)
        array = np.ones(len(data),dtype=bool)
        self.labeled_idxs = np.append(self.labeled_idxs, array)
        self.n_pool += len(data)
        return np.array(self.X_train)

    def get_label(self, idx):
        # Get the real label (share lable) for adversarial samples
        self.Y_train = np.array(self.Y_train)
        label = torch.tensor(self.Y_train[idx])
        return label

    def cal_target(self):
        target_num = []
        for i in range(len(self.Y_train)):
            target_num.append(np.sum(self.Y_train[i]))
        return target_num

def get_images(folders_name):
    image_path = r'../BraTS2019/LGG'
    images = []
    masks = []
    num_slices_per_patient = []
    for fld_name in folders_name:
        path_img_flair = os.path.join(image_path, fld_name, fld_name + '_flair.nii.gz')
        path_label = os.path.join(image_path, fld_name, fld_name + '_seg.nii.gz')

        img_flair = sitk.ReadImage(path_img_flair)
        img_flair = sitk.GetArrayFromImage(img_flair)

        label = sitk.ReadImage(path_label)
        label = sitk.GetArrayFromImage(label)
        label[(label >= 2)] = 1
        num_slices_per_patient.append(img_flair.shape[0])
        for index in range(0,img_flair.shape[0]):
            img_flair_ = img_flair[index]
            img_flair_ = np.expand_dims(img_flair_, axis=0)

            label_ = label[index]
            images.append(img_flair_)
            masks.append(label_)
        
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.uint8)
    return images, masks, num_slices_per_patient

def get_MSSEG(handler,supervised = False):
    #both 2d and 3d 
    # train_dir_name = "../MSSEG/Training/"
    # test_dir_name = "../MSSEG/Testing/"

    # ps=get_path(train_dir_name)
    # train_images_path = np.stack([name for name in [
    # #     [os.path.join(train_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/DP_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # ] if name is not None], axis=1)

    # train_masks_path = np.stack([name for name in [
    #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)

    # train_brain_masks_path = np.stack([name for name in [
    #     [os.path.join(train_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)

    # ps=get_path(test_dir_name)
    # test_images_path = np.stack([name for name in [
    # #     [os.path.join(test_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # ] if name is not None], axis=1)

    # test_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)

    # test_brain_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)
   
   
    # # train_images, train_num_slices_per_patient = get_image(train_images_path,label=False)
    # # train_masks, _ = get_image(train_masks_path,label=True) 
    # # train_brain_masks, _ = get_image(train_brain_masks_path,label=True)

    # # test_images, test_num_slices_per_patient = get_image(test_images_path,label=False)  
    # # test_masks, _ = get_image(test_masks_path,label=True)
    # # test_brain_area_masks, _ = get_image(test_brain_masks_path,label=True)

    # # # print(np.array(train_images).shape)
    # # # print(np.array(train_masks).shape)
    # # # print(np.array(train_brain_masks).shape)

    # # print(np.array(test_images).shape)
    # # print(np.array(test_masks).shape)
    # # print(np.array(test_brain_area_masks).shape)

    # # train_brain_images,train_brain_masks,train_brain_area_masks = get_brain_area(train_images,train_brain_masks,train_masks)
    # # print(train_brain_images.shape)
    # # print(train_brain_masks.shape)

    # # test_brain_images,test_brain_masks,test_brain_area_masks = get_brain_area(test_images,test_brain_area_masks,test_masks)
    # # print(test_brain_images.shape)
    # # print(test_brain_masks.shape)
    # # print(test_brain_area_masks.shape)

    # # train_x,val_x,train_y,val_y = train_test_split(train_brain_images,train_brain_masks,test_size=0.2,random_state=42)

    # # # #2d
    # # # #切分2d slice
    # # x_train = get_2d_slice(train_x,train_y,restrict=True)
    # # y_train = get_2d_slice(train_y,train_y,restrict=True)
    # # print(x_train.shape,y_train.shape)

    # # x_val = get_2d_slice(val_x,val_y,restrict=False)
    # # y_val = get_2d_slice(val_y,val_y,restrict=False)
    # # print(x_val.shape,y_val.shape)

    # # x_test = get_2d_slice(test_brain_images,test_brain_masks,restrict=False)
    # # y_test = get_2d_slice(test_brain_masks,test_brain_masks,restrict=False)
    # # print(x_test.shape,y_test.shape)

    # # # #为切分2d patch 防止有的无法整除
    # # # full_train_imgs_list = paint_border_overlap(x_train_slice,stride=32)
    # # # print(np.array(full_train_imgs_list).shape)
    # # # full_train_masks_list = paint_border_overlap(y_train_slice,stride=32)
    # # # print(np.array(full_train_masks_list).shape)

    # # # # full_val_imgs_list = paint_border_overlap(x_val_slice,stride=64)
    # # # # print(np.array(full_val_imgs_list).shape)
    # # # # full_val_masks_list = paint_border_overlap(y_val_slice,stride=64)
    # # # # print(np.array(full_val_masks_list).shape)

    # # full_test_imgs_list = paint_border_overlap(x_test_slice,stride=96)
    # # print(np.array(full_test_imgs_list).shape)
    # # full_test_masks_list = paint_border_overlap(y_test_slice,stride=96)
    # # print(np.array(full_test_masks_list).shape)

    # # # #得到64*64 2d patch
    # # # x_train,y_train = extract_ordered_overlap(np.array(full_train_imgs_list),label=full_train_masks_list,stride=32,train=True)
    # # # print(np.array(x_train).shape,np.array(y_train).shape)

    # # x_test,y_test = extract_ordered_overlap(np.array(full_test_imgs_list),label=full_test_masks_list,stride=96,train=False)
    # # print(np.array(x_test).shape,np.array(y_test).shape)

    # # train_x,val_x,train_y,val_y = train_test_split(x_train,y_train,test_size=0.2,random_state=42)
    # # print(np.array(train_x).shape)
    # # print(np.array(val_x).shape)




    # # 3d 补充原本的图像
    # # full_train_imgs_list = paint_border_overlap_3d(train_x,stride=16)
    # # print(np.array(full_train_imgs_list).shape)
    # # full_train_masks_list = paint_border_overlap_3d(train_y,stride=16)
    # # print(np.array(full_train_masks_list).shape)

    # # full_val_imgs_list = paint_border_overlap_3d(val_x,stride=48)
    # # print(np.array(full_val_imgs_list).shape)
    # # full_val_masks_list = paint_border_overlap_3d(val_y,stride=48)
    # # print(np.array(full_val_masks_list).shape)

    # # full_imgs_list = paint_border_overlap_3d(test_images,stride=48)
    # # print(np.array(full_imgs_list).shape)
    # # full_masks_list = paint_border_overlap_3d(test_masks,stride=48)
    # # print(np.array(full_masks_list).shape)

    # # # x_train = extract_ordered_overlap_3d(np.array(full_train_imgs_list),label=full_train_masks_list,stride=16,train=True)
    # # # print(np.array(x_train).shape)
    # # # y_train = extract_ordered_overlap_3d(np.array(full_train_masks_list),label=full_train_masks_list,stride=16,train=True)
    # # # print(np.array(y_train).shape)

    # # # x_val = extract_ordered_overlap_3d(np.array(full_val_imgs_list),label=full_val_masks_list,stride=48,train=False)
    # # # print(np.array(x_val).shape)
    # # # y_val = extract_ordered_overlap_3d(np.array(full_val_masks_list),label=full_val_masks_list,stride=48,train=False)
    # # # print(np.array(y_val).shape)

    # # # x_test = extract_ordered_overlap_3d(np.array(full_imgs_list),stride=48,train=False)
    # # # print(np.array(x_test).shape)
    # # # y_test = extract_ordered_overlap_3d(np.array(full_masks_list),stride=48,train=False)
    # # # print(np.array(y_test).shape)

    # # x_train = train_x
    # # y_train = train_y
    # # x_val = val_x
    # # y_val = val_y


    # # x_train = torch.load('../MSSEG/x_train_2d.pt')
    # # y_train = torch.load('../MSSEG/y_train_2d.pt')
    # # x_val = torch.load('../MSSEG/x_val_2d.pt')
    # # y_val = torch.load('../MSSEG/y_val_2d.pt')
    # # x_test = torch.load('../MSSEG/x_test_2d.pt')
    # # y_test = torch.load('../MSSEG/x_test_2d.pt')

    # # x_test_slice = np.load("../MSSEG/x_test_slice.npy", allow_pickle=True)#拼回2d
    # # full_test_imgs_list = np.load("../MSSEG/full_test_imgs_list.npy", allow_pickle=True)#拼回3d
    # # test_brain_images = np.load("../MSSEG/test_brain_images.npy", allow_pickle=True)#拼回3d
    # # test_brain_masks = np.load("../MSSEG/test_brain_masks.npy", allow_pickle=True)#求dice

    # # x_test = np.load("../MSSEG/x_test_2d_patch.npy", allow_pickle=True)
    # # y_test = np.load("../MSSEG/y_test_2d_patch.npy", allow_pickle=True)

    # # if supervised == True:
    # #     x_train = np.load("../MSSEG/x_train_2d_patch.npy", allow_pickle=True)
    # #     y_train = np.load("../MSSEG/y_train_2d_patch.npy", allow_pickle=True)
    # #     x_val = np.load("../MSSEG/x_val_2d_patch.npy", allow_pickle=True)
    # #     y_val = np.load("../MSSEG/y_val_2d_patch.npy", allow_pickle=True)

    # # else:
    # #     x_train = np.load("../MSSEG/x_train_2d_patch_full.npy", allow_pickle=True)
    # #     y_train = np.load("../MSSEG/y_train_2d_patch_full.npy", allow_pickle=True)
    # #     x_val = np.load("../MSSEG/x_val_2d_patch_full.npy", allow_pickle=True)
    # #     y_val = np.load("../MSSEG/y_val_2d_patch_full.npy", allow_pickle=True)

    # # print(x_train.shape, y_train.shape)
    # # print(x_val.shape, y_val.shape)
    # # print(x_test.shape, y_test.shape)

    # # return x_train, y_train, x_val, y_val, x_test, y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks
    # # train_slice_per_patient_sum = []
    # # sum = 0
    # # for i in train_num_slices_per_patient:
    # #     sum += i
    # #     train_slice_per_patient_sum.append(sum)
    # # np.save("../MSSEG/x_train", x_train)
    # # np.save("../MSSEG/y_train", y_train)
    # # np.save("../MSSEG/x_val", x_val)
    # # np.save("../MSSEG/y_val", y_val)
    # # np.save("../MSSEG/x_test", x_test)
    # # np.save("../MSSEG/y_test", y_test)
    # # np.save("../MSSEG/train_num_slices_per_patient", train_num_slices_per_patient)
    # # np.save("../MSSEG/test_num_slices_per_patient", test_num_slices_per_patient)

    # # x_train = np.load("../BraTS2019/x_train.npy", allow_pickle=True)
    # # y_train = np.load("../BraTS2019/y_train.npy", allow_pickle=True)
    # # x_val = np.load("../BraTS2019/x_val.npy", allow_pickle=True)
    # # y_val = np.load("../BraTS2019/y_val.npy", allow_pickle=True)
    # # x_test = np.load("../BraTS2019/x_test.npy", allow_pickle=True)
    # # y_test = np.load("../BraTS2019/y_test.npy", allow_pickle=True)
    # # train_num_slices_per_patient = np.load("../MSSEG/train_num_slices_per_patient.npy", allow_pickle=True)
    # # test_num_slices_per_patient = np.load("../MSSEG/test_num_slices_per_patient.npy", allow_pickle=True)

    # # # # 数据集拆分比例
    # train_ratio = 0.7
    # val_ratio = 0.15
    # test_ratio = 0.15
    # image_path = r'../BraTS2019/LGG'
    # # # 遍历数据集路径下的所有文件
    # folders_name = os.listdir(image_path)
    # # random.shuffle(folders_name)

    # # # 计算数据集拆分的索引
    # num_train = int(len(folders_name) * train_ratio)
    # num_val = int(len(folders_name) * val_ratio)
    # num_test = len(folders_name) - num_train - num_val

    # # # print(num_train)
    # train_path = folders_name[:num_train]
    # val_path = folders_name[num_train:num_train+num_val]
    # test_path = folders_name[num_train+num_val:num_train+num_val+num_test]
    # # print(len(folders_name))
    # # print("train",len(train_path))
    # # print("val",len(val_path))
    # # print("test",len(test_path))

    # train_images, train_labels, train_num_slices_per_patient = get_images(train_path)
    # val_images, val_labels, val_num_slices_per_patient = get_images(val_path)
    # test_images, test_labels, test_num_slices_per_patient = get_images(test_path)
    # print(train_images.shape,train_labels.shape)
    # print(val_images.shape,val_labels.shape)
    # print(test_images.shape,test_labels.shape)

    # # standarization
    # train_mean = np.mean(train_images)
    # train_std = np.std(train_images)

    # train_images = (train_images - train_mean) / train_std
    # val_images = (val_images - train_mean) / train_std
    # test_images = (test_images - train_mean) / train_std
    # print(train_images.shape,train_labels.shape)
    # print(val_images.shape,val_labels.shape)
    # print(test_images.shape,test_labels.shape)

    # np.save('../BraTS2019/train_num_slices_per_patient.npy', train_num_slices_per_patient)
    # np.save('../BraTS2019/val_num_slices_per_patient.npy', val_num_slices_per_patient)
    # np.save('../BraTS2019/test_num_slices_per_patient.npy', test_num_slices_per_patient)


    # np.save('../BraTS2019/train_image.npy', train_images)
    # np.save('../BraTS2019/train_label.npy', train_labels)
    # np.save('../BraTS2019/val_image.npy', val_images)
    # np.save('../BraTS2019/val_label.npy', val_labels)
    # np.save('../BraTS2019/test_image.npy', test_images)
    # np.save('../BraTS2019/test_label.npy', test_labels)

    # x_train = np.load('../BraTS2019/train_image_2024.npy')#(28055, 1, 240, 240)
    # y_train = np.load('../BraTS2019/train_label_2024.npy')
    # x_val = np.load('../BraTS2019/val_image_2024.npy')#(5890, 1, 240, 240)
    # y_val = np.load('../BraTS2019/val_label_2024.npy')
    # x_test = np.load('../BraTS2019/test_image_2024.npy')#(6200, 1, 240, 240)
    # y_test = np.load('../BraTS2019/test_label_2024.npy')

    # val_num_slices_per_patient = np.load('../BraTS2019/val_num_slices_per_patient_2024.npy')

    # train_num_slices_per_patient = np.load('../BraTS2019/train_num_slices_per_patient_2024.npy')
    # test_num_slices_per_patient = np.load('../BraTS2019/test_num_slices_per_patient_2024.npy')

    x_train = np.load('../BraTS2019/train_image.npy')#(28055, 1, 240, 240)
    y_train = np.load('../BraTS2019/train_label.npy')
    x_val = np.load('../BraTS2019/val_image.npy')#(5890, 1, 240, 240)
    y_val = np.load('../BraTS2019/val_label.npy')
    x_test = np.load('../BraTS2019/test_image.npy')#(6200, 1, 240, 240)
    y_test = np.load('../BraTS2019/test_label.npy')

    # x_train = np.load('../BraTS2019/train_image_f.npy')#(28055, 1, 240, 240)
    # y_train = np.load('../BraTS2019/train_label_f.npy')
    # x_val = np.load('../BraTS2019/val_image_f.npy')#(5890, 1, 240, 240)
    # y_val = np.load('../BraTS2019/val_label_f.npy')
    # x_test = np.load('../BraTS2019/test_image_f.npy')#(6200, 1, 240, 240)
    # y_test = np.load('../BraTS2019/test_label_f.npy')

    val_num_slices_per_patient = np.load('../BraTS2019/val_num_slices_per_patient.npy')

    train_num_slices_per_patient = np.load('../BraTS2019/train_num_slices_per_patient.npy')
    test_num_slices_per_patient = np.load('../BraTS2019/test_num_slices_per_patient.npy')

    print(x_train.shape,y_train.shape)
    print(x_val.shape,y_val.shape)
    print(x_test.shape,y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, train_num_slices_per_patient, val_num_slices_per_patient, test_num_slices_per_patient