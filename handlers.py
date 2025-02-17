import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from monai import transforms
from seed import setup_seed
import torch
import albumentations as A
import bisect


setup_seed()
# get dataloader
# online data augumentation
# keys = ("image", "label")
# class aug():
#     def __init__(self):
#         self.random_rotated = transforms.Compose([
#             transforms.AddChanneld(keys),  # 增加通道，monai所有Transforms方法默认的输入格式都是[C, W, H, ...],第一维一定是通道维
#             transforms.RandRotate90d(keys, prob=1, max_k=3, spatial_axes=(0, 1), allow_missing_keys=False),
#             transforms.RandFlipd(keys, prob=1, spatial_axis=(0, 1), allow_missing_keys=False),
#             transforms.RandGaussianNoised(keys, prob=0.1, mean=0.0, std=0.1, allow_missing_keys=False),
# #             transforms.NormalizeIntensityd(keys, allow_missing_keys=False),
#             transforms.ToTensord(keys)
#         ])
    
#     def forward(self,x):
#         x = self.random_rotated(x)
#         return x
    
class MSSEG_Handler_2d(Dataset):
    def __init__(self,image,label,mode="train"):
        self.image=np.array(image)
        self.label=np.array(label)
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate (limit=90, interpolation=1,always_apply=False, p=0.5),
                # A.Resize(width=256, height=256 ,p=1),
        ]) 
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.label)
    def __getitem__(self,index): 
        img = self.image[index].astype(np.float32)
        label = self.label[index].astype(np.uint8)
        if self.transform!=None: 
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        img = torch.tensor(img)
        label = torch.tensor(label).unsqueeze(0)
        return img, label, index


class Prop_Handler(Dataset) :
    def __init__(self,X_train,Y_train,labeled_idxs,k,train_slice_per_patient_sum, mode="train"):
    # labeled slice 与其label X_train[labeled_idxs] Y_train[labeled_idxs]
        # print(labeled_idxs)
        self.X_train = X_train[labeled_idxs]#(400, 1, 240, 240)
        self.Y_train = Y_train[labeled_idxs]#(400, 240, 240)
        self.labeled_idxs = labeled_idxs
        self.k = k
        self.cumulative_counts = []
        cumulative_sum = 0
        # self.patient_ids = patient_ids
        self.train_slice_per_patient_sum = train_slice_per_patient_sum
        # self.index_list = index_list
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate (limit=90, interpolation=1,always_apply=False, p=0.5),
                # A.Resize(width=256, height=256, p=1),
        ]) 
        else:
            self.transform=None

         #[155, 310, 465, 620, 775, 930, 1085, 1240, 1395, 1550, 1705, 1860, 2015, 2170, 2325, 2480, 2635, 2790, 2945, 3100, 3255, 3410, 3565, 3720, 3875, 4030, 4185, 4340, 4495, 4650, 4805, 4960, 5115, 5270, 5425, 5580, 5735, 5890, 6045, 6200, 6355, 6510, 6665, 6820, 6975, 7130, 7285, 7440, 7595, 7750]

        self.labeled_data_count = []
        sidx = 0
        for count in self.train_slice_per_patient_sum: #[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
            eidx = sidx + count
            labeled_idxs_in_patient = [idx for idx in self.labeled_idxs if sidx <= idx < eidx]
            self.labeled_data_count.append(len(labeled_idxs_in_patient))
            sidx = eidx

        for count in self.labeled_data_count:
            cumulative_sum += count
            self.cumulative_counts.append(cumulative_sum)


    def __getitem__(self, idx): 
        # if idx not in self.labeled_idxs: #不能根据labled idx来 要不然会越来越多？
        #     return None
        # print(idx)
        start_idx = idx #每个sample的第一张
        end_idx = start_idx + self.k#每个sample的最后一张
        # 所有切片需要属于同一个病人，如果不是则跳过
        for i in range(len(self.cumulative_counts) - 1): #([512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 336, 336, 336, 336, 336])
            if start_idx < self.cumulative_counts[i] and end_idx > self.cumulative_counts[i]:
                    end_idx = self.cumulative_counts[i]-1
                    start_idx = end_idx-self.k
                    # raise ValueError("Invalid index: {idx}")

        #     break
        # #看是否一样
        # index = bisect.bisect_right(self.cumulative_counts, start_idx)
        # if index < len(self.cumulative_counts) and end_idx >= self.cumulative_counts[index]:
        #     end_idx = self.cumulative_counts[index]
        #     start_idx = end_idx - self.k
        images = self.X_train[start_idx:end_idx] # idx, start_idx, end_idx, images.shape,masks.shape 1 1 3 (2, 1, 240, 240) (2, 240, 240)
        masks = self.Y_train[start_idx:end_idx]
        # print(start_idx,end_idx)
        if self.transform is not None:
            image_all = torch.empty((len(images), *images[0].shape), dtype=torch.float32)
            mask_all = torch.empty((len(masks), 1, *masks[0].shape), dtype=torch.uint8)

            mask_all = torch.empty((len(masks), *masks[0].shape), dtype=torch.uint8)
            for i in range(len(images)):
                # Apply transformation to each slice
                transformed = self.transform(image=images[i], mask=masks[i])
                image_all[i] = torch.tensor(transformed['image'])
                # mask_all[i] = torch.tensor(transformed['mask']).unsqueeze(0)
                mask_all[i] = torch.tensor(transformed['mask'])

        else:
            image_all = torch.tensor(images, dtype=torch.float32)
            # mask_all = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
            mask_all = torch.tensor(masks, dtype=torch.float32)

        # print(image_all.shape,mask_all.shape)
        return image_all, mask_all#注意这里是所有mask都给出了 后面需要过滤一下
        
    def __len__(self):
        # return len(self.X_train)
        return len(self.X_train)-1


# prop
class Prop_pseudo_Handler(Dataset) :
    def __init__(self,X_train,Y_train,labeled_idxs,k,train_slice_per_patient_sum):
        self.X_train = X_train
        self.Y_train = Y_train
        # print(self.X_train.shape,self.Y_train.shape)
        self.labeled_idxs = labeled_idxs
        self.k = k
        self.cumulative_counts = []
        cumulative_sum = 0
        # self.patient_ids = patient_ids
        self.train_slice_per_patient_sum = train_slice_per_patient_sum
        # self.index_list = index_list

        for count in self.train_slice_per_patient_sum:
            cumulative_sum += count
            self.cumulative_counts.append(cumulative_sum)

    def __getitem__(self, idx): 
        labeled_slice = []
        if idx not in self.labeled_idxs: #不能根据labled idx来 要不然会越来越多？
            return None
        start_idx = idx #每个sample的第一张
        end_idx = start_idx + self.k#每个sample的最后一张

        #防止最后数量不一致报错
        if end_idx > len(self.X_train):
            end_idx = len(self.X_train)
            start_idx = end_idx-self.k  

        # 所有切片需要属于同一个病人
        for i in range(len(self.cumulative_counts) - 1): #([512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 336, 336, 336, 336, 336])
            if start_idx < self.cumulative_counts[i] and end_idx > self.cumulative_counts[i]:
                end_idx = self.cumulative_counts[i]-1
                start_idx = end_idx-self.k
                break        
        
        # 每组labeled slice
        for i in range(start_idx, end_idx):
            if i in self.labeled_idxs: 
                labeled_slice.append(i-start_idx) #绝对值 labeled slice 从0开始
        # #看是否一样
        # index = bisect.bisect_right(self.cumulative_counts, start_idx)
        # if index < len(self.cumulative_counts) and end_idx >= self.cumulative_counts[index]:
        #     end_idx = self.cumulative_counts[index]
        #     start_idx = end_idx - self.k
        
        images = self.X_train[start_idx:end_idx]
        masks = self.Y_train[start_idx:end_idx]

        # print("all_idx",np.array(range(start_idx,end_idx+1)),"start_idx",start_idx,"end_idx",end_idx,"labeled_slice",labeled_slice,"images",images.shape,"masks",masks.shape)
        return np.array(range(start_idx,end_idx+1)), images, masks, np.array(labeled_slice)#注意这里是所有mask都给出了 后面需要过滤一下
        
    def __len__(self):
        return len(self.X_train)-1
    
        # return len(self.labeled_idxs)/self.k