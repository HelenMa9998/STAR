import math
from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import torchvision
from collections import OrderedDict
from tqdm import tqdm
from seed import setup_seed
import pdb
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim


setup_seed()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=0.0001):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits = (logits > 0.5).float()
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * (intersection + self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits.sum(-1) + targets.sum(-1)) + self.epsilon)
        return torch.mean(dice_score)

# including different training method for active learning process (train acc=1, val loss, val acc, epoch)
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def supervised_val_loss(self, data, val_data,rd):
        n_epoch = 100
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        train_loss=0
        validation_loss = 0
        train_dice=0
        val_dice=0
        self.clf = self.net(phase='test').to(self.device)
        self.clf.train()
        path = './result/'
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        if rd==0:
            self.clf = self.clf
        else:
            # self.clf = torch.load('./result/model.pth')

            self.clf = torch.load('./result/model_1.pth', map_location="cuda:0")
            # self.clf.Encoder_M.load_state_dict(self.clf.Encoder_Q.state_dict(),strict=False)
        # self.clf.Encoder_Q.requires_grad_(True)
        # self.clf.Decoder.requires_grad_(True)

        optimizer = optim.Adam(self.clf.parameters(), lr=0.0001)
        # criterion = nn.BCEWithLogitsLoss()
        criterion = FocalLoss(alpha=0.6, gamma=2,logits=True)

        loader=DataLoader(data, **self.params['norm_args'])
        val_loader=DataLoader(val_data, **self.params['val_args'])
        sigmoid = nn.Sigmoid()

        dice = dice_coefficient()
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):#([8, 1, 240, 240])
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x, phase = 'test')
                loss = criterion(out.float(),y.float())
                loss.backward()
                optimizer.step()
                # train_dice += dice(y,sigmoid(out))
                # train_loss += loss#一个epoch的loss
            # print("\n epoch",epoch,"train loss: ",train_loss/(batch_idx+1),"train_dice: ",train_dice/(batch_idx+1))
            # clear loss and auc for training
            # train_loss=0
            # train_dice=0

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (valinputs, valtargets, idxs) in enumerate(val_loader):
                    # valinputs, valtargets = valinputs.unsqueeze(1), valtargets.unsqueeze(1)
                    valinputs, valtargets = valinputs.to(self.device), valtargets.to(self.device)
                    valoutputs = self.clf(valinputs, phase = 'test')
                    validation_loss += criterion(valoutputs.float(), valtargets.float())
            #         val_dice += dice(valtargets,sigmoid(valoutputs))
            # print(" epoch: ",epoch,"val loss: ",validation_loss/(valbatch_idx+1),"val_dice: ",val_dice/(valbatch_idx+1))
            
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if validation_loss / (valbatch_idx + 1) < best['loss']:
                trigger = 0
                best['epoch'] = epoch
                best['loss'] = validation_loss / (valbatch_idx + 1)
                # print(best['epoch'],best['loss'])
                torch.save(self.clf, './result/model.pth')
            # print("\n best performance at Epoch :{}, loss :{}".format(best['epoch'],best['loss']))
            validation_loss = 0
            # val_dice=0
            if trigger >= 5:
                break
        torch.cuda.empty_cache()
 

    def prop_train(self, train_data, val_data, rd):
        n_epoch = 100
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        train_loss = 0
        val_loss = 0
        train_dice = 0
        val_dice = 0
        dice = dice_coefficient()

        self.clf = self.net(phase='train').to(self.device)
        # self.clf = self.net().to(self.device)
        self.clf = torch.load('./result/model.pth', map_location="cuda:0")
        self.clf.Encoder_M.load_state_dict(self.clf.Encoder_Q.state_dict(),strict=False)
        
        # self.clf.Encoder_Q.eval()
        # self.clf.Encoder_Q.requires_grad_(False)

        # self.clf.Decoder.eval()
        # self.clf.Decoder.requires_grad_(False)

        optimizer = optim.Adam(self.clf.parameters(), lr=0.00005)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss(alpha=0.6, gamma=2,logits=True)

        sigmoid = nn.Sigmoid()
        
        train_loader = DataLoader(train_data, **self.params['train_args'])
        val_loader = DataLoader(val_data, **self.params['val_args'])
        # dataset_length = len(train_loader.dataset)
        # print("Dataset Length:", dataset_length)

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            # self.clf.train()
            for batch_idx, data in enumerate(train_loader):

                frames, masks = data
                frames = frames.to(self.device)#([32, 5, 1, 240, 240])
                masks = masks.to(self.device)#([32, 5, 1, 240, 240])
                
                # label_idxs = torch.stack(label_idxs, dim=1).detach().cuda()
                N, T, C, H, W = frames.size()#([32, 5, 1, 240, 240])
                total_loss = 0
                total_dice = 0
                keys = []
                vals = []

                count = 0

                # for idx in range(N):#第一个batch的
                #     frame = frames[idx]
                #     mask = masks[idx]
                for t in range(0, T-1):
                    # tmp_mask = masks[:,t,:,:]#([1, 240, 240])
                    #label prop train
                    # frame = frames[:,t,:,:].squeeze()#([32, 240, 240])
                    # mask = masks[:,t,:,:]#([32, 1, 240, 240])

                    # print(frame.shape,mask.shape)
                    key, val, _ = self.clf(frame=frames[:,t,:,:], mask= masks[:,t,:,:]) #获取frame[t-1:t, :, :, :]（memory）的key val ([32, 240, 240])
                    keys.append(key)#([32, 128, 15, 15])
                    vals.append(val)
                    tmp_key = torch.cat(keys, dim=1)
                    tmp_val = torch.cat(vals, dim=1)

                    # print(label_idx[i+1])
                    logits,_ = self.clf(frame=frames[:,t+1,:,:], keys=tmp_key, values=tmp_val)
                    # out = torch.softmax(logits, dim=1)
                    gt = masks[:,t+1,:,:] #这个mask第一轮是从哪里来的？？
                    prop_loss = criterion(logits.squeeze(1).float(),gt.float())
                    total_loss += prop_loss
                        # total_dice += dice(gt,sigmoid(logits))

                total_loss = total_loss / (N * (T-1))
                # total_dice = total_dice / (N * (T-1))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # train_loss += total_loss#一个epoch的loss
                # train_dice += total_dice
                total_loss=0
                # total_dice=0
            # print("\n epoch",epoch,"train loss: ",train_loss/(batch_idx+1),"train_dice: ",train_dice/(batch_idx+1))
            # train_dice = 0
            # train_loss = 0
            
            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, data in enumerate(val_loader):
                    # for batch_idx, (x, y, idxs) in enumerate(train_loader):#([8, 1, 240, 240]
                
                    # if data is None:
                    #     continue
                    frames, masks = data
                    frames = frames.to(self.device)#([32, 5, 1, 240, 240])
                    masks = masks.to(self.device)#([32, 5, 1, 240, 240])
                    # print(frames.shape)
                    # print(masks.shape)
                    
                    # label_idxs = torch.stack(label_idxs, dim=1).detach().cuda()
                    N, T, C, H, W = frames.size()#([32, 5, 1, 240, 240])
                    total_loss = 0
                    total_dice = 0
                    keys = []
                    vals = []
                    # x = frames.squeeze(1)# ([32, 1, 240, 240])
                    # y = masks.squeeze(1)

                    # for idx in range(N):#第一个batch的
                    #     frame = frames[idx]
                    #     mask = masks[idx]
                    for t in range(0, T-1):
                        tmp_mask = masks[:,t,:,:]#([1, 240, 240])
                        #label prop train
                        # frame = frames[:,t,:,:].squeeze()
                        # mask = masks[:,t,:,:]
                        
                        #label prop train
                        key, val, _ = self.clf(frame=frames[:,t,:,:], mask= masks[:,t,:,:]) #获取frame[t-1:t, :, :, :]（memory）的key val ([1, 240, 240])
                        keys.append(key)
                        vals.append(val)
                        tmp_key = torch.cat(keys, dim=1)#([1, 7200, 128])
                        tmp_val = torch.cat(vals, dim=1)#([1, 7200, 512])
                        # print(label_idx[i+1])
                        logits,_ = self.clf(frame=frames[:,t+1,:,:], keys=tmp_key, values=tmp_val)
                        # out = torch.softmax(logits, dim=1)
                        gt = masks[:,t+1,:,:] #这个mask第一轮是从哪里来的？？
                        prop_loss = criterion(logits.squeeze(1).float(),gt.float())

                        total_loss += prop_loss
                        # total_dice += dice(gt,sigmoid(logits))

                    total_loss = total_loss / (N * (T-1))
                    # total_dice = total_dice / (N * (T-1))
                    val_loss += total_loss#一个epoch的loss
                    # val_dice += total_dice
                    total_loss=0
                trigger += 1
                # early stopping condition: if the acc not getting larger for over 10 epochs, stop
                if val_loss / (valbatch_idx + 1) < best['loss']:
                    trigger = 0
                    best['epoch'] = epoch
                    best['loss'] = val_loss / (valbatch_idx + 1)
                    # print(best['epoch'],best['loss'])
                    torch.save(self.clf, './result/model_1.pth')
                # print("\n best performance at Epoch :{}, loss :{}".format(best['epoch'],best['loss']))
                val_loss = 0
                val_dice=0
                if trigger >= 5:
                    break
            torch.cuda.empty_cache()

    def calculate_iou(self, pred_mask, true_mask):
        intersection = (pred_mask & true_mask).sum()
        union = (pred_mask | true_mask).sum()
        iou = intersection / union
        return iou
    
    def prop(self, data): #如果是一个trainloader就可以跟上面合并了
        # label prop
        model = torch.load('./result/model_1.pth')
        sigmoid = nn.Sigmoid()
        pred = []
        index = []
        similarities = []
        label_similarity = []
        image_similarity = []
        feature_similarity = []
        model.eval()
        dice = dice_coefficient()
        def custom_collate(batch):
            batch_data = [data for data in batch if data is not None and len(data[1]) == 11]
            if len(batch_data) == 0:
                return None
            return torch.utils.data._utils.collate.default_collate(batch_data)
    
        loader = DataLoader(data, **self.params['prop_args'], collate_fn=custom_collate)
        
        with torch.no_grad():
            # print("aaaaaa")
            for batch_idx, data in enumerate(loader):
                if data is None:
                    continue
                
                all_idxs, frames, masks, label_idxs = data

                frames = frames.cuda() # ([1, 21, 1, 240, 240])
                masks = masks.cuda() # ([1, 21, 240, 240])
                
                N, T, C, H, W = frames.size()#([1, 21, 1, 240, 240])

                # out, quality, ious = model(frame=frames, mask=masks)
                keys = []
                vals = []
                
                for idx in range(N):
                    frame = frames[idx] # T, C, H, W
                    mask = masks[idx]
                    # print("idx",idx)
                    label_idx = label_idxs[idx]
                    all_idx = all_idxs[idx]
                    
                    # 得到所有labeled slice
                    all_memory_slice = frame[label_idx]
                    all_memory_mask_slice = mask[label_idx]
                    memory_feature = []
                    # all_memory_mask_slice = torch.mean(all_memory_mask_slice.float(), dim=0)
                    
                    for i in label_idx:
                        
                        memory_slice = frame[i]
                        memory_mask_slice = mask[i]

                        key, val, feature = model(frame=memory_slice, mask=memory_mask_slice.unsqueeze(0),sim=True) #获取frame（memory）的key val
                        keys.append(key)
                        vals.append(val)
                        tmp_key = torch.cat(keys, dim=1)
                        tmp_val = torch.cat(vals, dim=1)
                        memory_feature.append(feature)
                    memory_feature = torch.stack(memory_feature)

                    for t in range(0, T-1):  
                        # print("bbbb")
                        if t not in label_idx: 
                            # print("cccc")

                            logits,feature = model(frame=frame[t], keys=tmp_key, values=tmp_val,sim=True)#([1, 1, 240, 240])
                            out = sigmoid(logits)
                            pred.append(out.cpu())

                            #针对prediction-level
                            # label_sim = dice(all_memory_mask_slice, out.squeeze().repeat(all_memory_mask_slice.shape[0],1,1)).mean() #dice
                            # print(all_memory_slice.squeeze().shape)
                            # print(frame[t].repeat(all_memory_slice.shape[0],1,1).cpu().numpy().shape)

                            # sample ssim
                            # ssim_values = []
                            # for i in range(all_memory_slice.shape[0]): 
                            #     similarity = ssim(all_memory_slice[i].squeeze(), frame[t].cpu().numpy().squeeze())
                            #     ssim_values.append(similarity)
                            # image_sim = sum(ssim_values) / len(ssim_values)
                            # similarity = ssim(all_memory_slice.squeeze(), frame[t].repeat(all_memory_slice.shape[0],1,1).cpu().numpy()).mean()
                            
                            # feature-level
                            # memory_feature = torch.cat(memory_feature, dim=0)#([768, 60, 60])
                            # feature = feature.view( -1)  # 2 57600
                            # memory_feature = memory_feature.view(len(memory_feature), -1)  # 2 57600

                            feature_sim = torch.nn.functional.cosine_similarity(feature, memory_feature).mean() # cos sim

                            # difference = feature.unsqueeze(0) - memory_feature  # 在维度 0 上添加一个维度，以便进行广播计算
                            # squared_difference = difference ** 2
                            # sum_squared_difference = squared_difference.sum(dim=(1, 2, 3))  # 在维度 1, 2, 3, 4 上求和
                            # feature_sim = torch.sqrt(sum_squared_difference).mean()
                            

                            # prediction-level
                            # out = out.view(-1)  # 将 out 转换为形状为 (C * H * W,) 的向量 57600
                            # all_memory_mask_slice = all_memory_mask_slice.view(len(all_memory_mask_slice), -1)  # 2 57600
                            # label_sim = torch.nn.functional.cosine_similarity(out, all_memory_mask_slice).mean() # cos sim
                            # label_sim = self.calculate_iou(out.squeeze(), all_memory_mask_slice)

                            # difference = out.unsqueeze(0) - all_memory_mask_slice  # 在维度 0 上添加一个维度，以便进行广播计算
                            # squared_difference = difference ** 2
                            # sum_squared_difference = squared_difference.sum(dim=(1, 2, 3))  # 在维度 1, 2, 3, 4 上求和
                            # label_sim = torch.sqrt(sum_squared_difference).mean()

                            # label_similarity.append(label_sim)
                            feature_similarity.append(feature_sim)

                            # image_similarity.append(image_sim)
                            index.append(all_idx[t])

        index = np.array(index)
        probs = torch.cat(pred, dim=0)#(40617, 1, 128, 128)

        # label_similarity = torch.tensor(label_similarity)
        # image_similarity = torch.tensor(image_similarity)
        feature_similarity = torch.tensor(feature_similarity)

        unique_indices = np.unique(index)
        # unique_label_similarity = torch.tensor([label_similarity[index == idx].mean().item() for idx in unique_indices])
        # unique_image_similarity = torch.tensor([image_similarity[index == idx].mean().item() for idx in unique_indices])
        unique_feature_similarity = torch.tensor([feature_similarity[index == idx].mean().item() for idx in unique_indices])

        unique_probs = torch.stack([torch.mean(probs[index == idx],dim=0) for idx in unique_indices])

        # 位次相加
        # label_ranks = torch.argsort(unique_label_similarity.sort()[1]) #样本对应的位次
        # # image_ranks = torch.argsort(unique_image_similarity.sort()[1])
        # feature_ranks = torch.argsort(unique_feature_similarity.sort()[1])

        # # combined_rank = torch.argsort(label_ranks+image_ranks)
        # combined_rank = torch.argsort(label_ranks+feature_ranks)


        # combined_similarity = torch.tensor([label_score + image_score for label_score, image_score in zip(unique_label_similarity, unique_image_similarity)])
        # combined_similarity = torch.tensor([label_score + image_score for label_score, image_score in zip(unique_label_similarity, unique_feature_similarity)])


        # probs = torch.tensor([probs[index == idx].mean().item() for idx in unique_indices])
        # unique_similarities = torch.tensor([combined_similarity[index == idx].mean().item() for idx in unique_indices])

        # similarities = {idx: avg_prob for idx, avg_prob in zip(index, similarities)}
        # similarities = dict(sorted(similarities.items(), key=lambda item: item[1]))

        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
        unique_uncertainties = torch.tensor([uncertainties[index == idx].mean().item() for idx in unique_indices])

        # print(len(list(similarities.keys())[-100:]))
        # print(len(list(similarities.keys())[:100]))
        # print(probs[torch.tensor(list(similarities.keys())[:100])].shape)
        return list(unique_indices[unique_uncertainties.sort()[1][-100:]]), unique_probs[unique_uncertainties.sort()[1][-100:]], list(unique_indices[unique_feature_similarity.sort()[1][:100]])



    def pseudo_predict(self, unlabeled_idxs, unlabeled_data):
        self.clf = self.net(phase='test').to(self.device)
        self.clf = torch.load('./result/model.pth')
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clf = self.clf.to(device)
        self.clf.eval()
        preds = []
        targets = []
        
        loader = DataLoader(unlabeled_data, **self.params['test_args'])
        sigmoid = nn.Sigmoid()
        
        with torch.no_grad(): 
            for x, y, idxs in loader:

                # print(x.shape,y.shape)
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x,phase='test')
                out = sigmoid(out)
                outputs = out.cpu()
                preds.append(outputs)
                # targets.append(y.data.cpu().numpy())
        probs = torch.cat(preds, dim=0)#(40617, 1, 128, 128)
        log_probs = torch.log(probs)
        # print(probs.shape)
        uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
        return unlabeled_idxs[uncertainties.sort()[1][-200:]], probs[uncertainties.sort()[1][-200:]]
    
## restore to original dimensions
    def predict(self, data, num_slices_per_patient):
        self.clf = self.net(phase='test').to(self.device)
        self.clf = torch.load('./result/model.pth')
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clf = self.clf.to(device)
        self.clf.eval()
        preds = []
        targets = []
        
        loader = DataLoader(data, **self.params['test_args'])
        sigmoid = nn.Sigmoid()
        
        with torch.no_grad(): 
            for x, y, idxs in loader:

                # print(x.shape,y.shape)
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x,phase='test')
                out = sigmoid(out)
                outputs = out.data.cpu().numpy()
                preds.append(outputs)
                targets.append(y.data.cpu().numpy())
        predictions = np.concatenate(preds, axis=0)#(40617, 1, 128, 128)
        targets = np.concatenate(targets, axis=0)#(40617, 1, 128, 128)
        return predictions, targets


    # Calculating probability for prediction, used as uncertainty
    def predict_prob(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                prob = self.clf(x,phase='test')
                # prob = F.softmax(out, dim=1) # torch.Size([8, 2, 64, 64, 64])
                probs[idxs] = prob.cpu() 

        return probs

    # Calculating 10 times probability for prediction, the mean used as uncertainty
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf = torch.load('./result/model.pth')
        self.clf.train()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    # x, y = x.unsqueeze(1), y.unsqueeze(1)
                    x, y = x.to(self.device), y.to(self.device)
                    prob = self.clf(x,phase='test')
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    # Used for Bayesian sampling
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf = torch.load('./result/model.pth')
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    # x, y = x.unsqueeze(1), y.unsqueeze(1)
                    x, y = x.to(self.device), y.to(self.device)
                    prob = self.clf(x,phase='test')
                    probs[i][idxs] += prob.cpu()
        return probs

    def get_embeddings(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.clf(x, embedding = True)
                embeddings[idxs] = e1.cpu().reshape(len(x),-1)
        return embeddings

    
    def predict_sim(self, data):
        self.clf = torch.load('./result/model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1, 512, 512])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.unsqueeze(1), y.unsqueeze(1)
                x, y = x.to(self.device), y.to(self.device)
                prob = self.clf(x,phase='test')
                # prob = F.softmax(out, dim=1) # torch.Size([8, 2, 64, 64, 64])
                probs[idxs] = prob.cpu() 
        return probs


CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 
    
class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:])
        m = self.ResMM(m)
        return m
    
class Decoder(nn.Module):
    def __init__(self, inplane, mdim, expand):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim) # 卷积 + 残差
        self.RF3 = Refine(128 * expand, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(64 * expand, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, f):
        # print("f",f.shape)
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256
        p2 = self.pred2(F.relu(m2))
        # print("p2",p2.shape)

        p = F.interpolate(p2, size=f.shape[1:]) 
        return p

class Encoder_M(nn.Module): # 三层resent 只不过输入是图片+target+背景
    def __init__(self, arch):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, :1, :, :])

        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_bg):#in_f torch.Size([1, 240, 240])??  ([2, 1, 240, 240])??
        # print("Encoder_M_in_f",in_f.shape)
        # print("Encoder_M_in_m",in_m.shape)

        f = torch.unsqueeze(in_f, dim=1).float()#([32, 240, 240])
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()#([32, 1, 240, 240])
        # print("Encoder_M_f",f.shape)
        # print("Encoder_M_m",m.shape)

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024

        return r4, r3, r2, c1
 
class Encoder_Q(nn.Module): #三层resent
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, :1, :, :])
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):#([32, 240, 240])
        # f = (in_f - self.mean) / self.std
        f = torch.unsqueeze(in_f, dim=1).float()#([32, 1, 240, 240])
        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024 
        return r4, r3, r2, c1

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()#keys, values, k4, v4)
 
    def forward(self, m_in, m_out, q_in, q_out):  # keys([1, 7200, 128]), values([1, 7200, 512]), k4([32, 128, 225]), v4
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape
        qi = q_in.view(-1, C, H*W) 
        # print(m_in.shape)
        # print(qi.shape)

        p = torch.bmm(m_in, qi) # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1) # no x centers x hw

        mo = m_out.permute(0, 2, 1) # no x c x centers 
        mem = torch.bmm(mo, p) # no x c x hw
        mem = mem.view(no, vd, H, W)

        mem_out = torch.cat([mem, q_out], dim=1) # 变成新feature：旧feature+memory feature

        return mem_out


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        # self.Key = nn.Linear(indim, keydim)
        # self.Value = nn.Linear(indim, valdim)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class prop_model(nn.Module):
    def __init__(self, phase):
        super(prop_model, self).__init__()

        keydim = 128
        valdim = 512
        arch = 'resnet50'

        expand = CHANNEL_EXPAND[arch]
        self.phase = phase
        self.Encoder_M = Encoder_M(arch) #对memory的encoder：slice+mask 实现时就是把slice feature加入mask object与bg feature
        self.Encoder_Q = Encoder_Q(arch) #对query的encoder：就一个slice feature

        self.keydim = keydim #key的大小
        self.valdim = valdim #val的大小

        self.KV_M_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim) #输入 memory feature得到 key and value
        self.KV_Q_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim) #输入 query feature得到 key and value

        self.Memory = Memory() #query与memory中slice计算new feature
        self.Decoder = Decoder(2*valdim, 256, expand)


    # def load_param(self, weight):
        # s = self.state_dict()
        pretrained_model = torch.load("../stm_cycle_100.pth", map_location="cuda:0")

        pretrained_model['Encoder_M.conv1.weight'] = pretrained_model['Encoder_M.conv1.weight'][:, :1, :, :]
        pretrained_model['Encoder_Q.conv1.weight'] = pretrained_model['Encoder_Q.conv1.weight'][:, :1, :, :]
        pretrained_model['Decoder.pred2.weight'] = pretrained_model['Decoder.pred2.weight'][:1, :, :, :]
        pretrained_model['Decoder.pred2.bias'] = pretrained_model['Decoder.pred2.bias'][:1]
        self.load_state_dict(pretrained_model)

    def memorize(self, frame, mask, sim = False): #([32, 240, 240])
        r4, r3, r2, _  = self.Encoder_M(frame, mask, torch.clamp(1.0 - mask, min=0.0, max=1.0)) #frame([1, 240, 240]) mask([1, 240, 240])
        # r4, _, _, _ = self.Encoder_Q(frame) 

        # torch.Size([16, 512, 64, 64])
        # torch.Size([16, 256, 128, 128])
        # torch.Size([16, 1024, 32, 32])
        _, c, h, w = r4.size() # no, c, h, w
        memfeat = r4
        k4, v4 = self.KV_M_r4(memfeat) # ([32, 128, 15, 15]) ([32, 512, 15, 15])
        k4 = k4.permute(0, 2, 3, 1).contiguous().view(k4.shape[0], -1, self.keydim)#([32, 225, 128])
        v4 = v4.permute(0, 2, 3, 1).contiguous().view(v4.shape[0], -1, self.valdim)#([32, 225, 512])
        concatenated_features = None
        # print("memorize979",k4.shape,v4.shape)
        if sim == True:
            # print("memorize982",k4.shape,v4.shape)

            new_size = (32, 256, 60, 60)  # 期望的新尺寸
            # 使用 bilinear 插值调整特征的大小
            r4_resized = F.interpolate(r4.unsqueeze(0), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            r3_resized = F.interpolate(r3.unsqueeze(0), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            concatenated_features = torch.cat((r4_resized, r3_resized, r2.squeeze()), dim=0)
            # r4_resized = F.interpolate(r4.unsqueeze(0).cpu(), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            # r3_resized = F.interpolate(r3.unsqueeze(0).cpu(), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            # concatenated_features = torch.cat((r4_resized.cpu(), r3_resized.cpu(), r2.cpu()), dim=1)
        return k4, v4, concatenated_features

    def segment(self, frame, keys, values, sim=False): # ([1, 240, 240]),([1, 7200, 128]),([1, 7200, 128])
        # segment one input frame
        # print("segmentframe",frame.shape)
        r4, r3, r2, _ = self.Encoder_Q(frame)#([32, 1024, 15, 15]) ([32, 512, 30, 30]) ([32, 256, 60, 60])

        n, c, h, w = r4.size()
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16 ([1, 128, 15, 15])
        # print(k4.shape)
        m4 = self.Memory(keys, values, k4, v4) # 与memory中slice获取当前slice新featuree([256, 128])
        logit = self.Decoder(m4, r3, r2, frame) # 多尺度获得更好的结果

        concatenated_features = None
        # print("memorize979",k4.shape,v4.shape)
        if sim == True:
            # print("memorize982",k4.shape,v4.shape)

            new_size = (32, 256, 60, 60)  # 期望的新尺寸
            # 使用 bilinear 插值调整特征的大小
            r4_resized = F.interpolate(r4.unsqueeze(0), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            r3_resized = F.interpolate(r3.unsqueeze(0), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            concatenated_features = torch.cat((r4_resized, r3_resized, r2.squeeze()), dim=0)

            # r4_resized = F.interpolate(r4.unsqueeze(0).cpu(), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            # r3_resized = F.interpolate(r3.unsqueeze(0).cpu(), size=new_size[1:], mode='trilinear', align_corners=False).squeeze()
            # concatenated_features = torch.cat((r4_resized.cpu(), r3_resized.cpu(), r2.cpu()), dim=1)
            # print(r4_resized.shape)
            # print(r3_resized.shape)
            # print(r2.shape)


        return logit,concatenated_features

    def forward(self, frame, mask=None, keys=None, values=None, criterion=None, phase="train", sim=False):
        if phase == 'test': 
            # print("805",frame.shape)
            r4, r3, r2, c1 = self.Encoder_Q(frame.squeeze(1))
            pred = self.Decoder(r4, r3, r2, frame.squeeze(1))
            
            return pred #这里需要统一两个encoder

        elif phase == 'train': 
            if mask is not None: # 已经存在mask

                return self.memorize(frame.squeeze(1), mask, sim) # Encoder_M(跟encoderq差不多 可以迁移一下)  KV_M_r4(一层)
            else: 
                return self.segment(frame.squeeze(1), keys, values, sim) # Encoder_Q   KV_Q_r4(一层)  Memory(无参数)  Decoder 
        
        else:
            raise NotImplementedError('unsupported forward mode %s' % self.phase)
