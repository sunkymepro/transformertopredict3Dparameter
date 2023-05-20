import torch
import torch.nn as nn
import numpy as np
import cv2
from encoder import nlp_transformer_backbone_mask as models
from encoder import output_linear_div_ori3 as linear
from torch.utils.data import DataLoader
from dataset import data_deca
from itertools import cycle
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import argparse
import collections
from soft_dtw_cuda import SoftDTW
import random

BATCH_SIZE=80  #960
BEGIN_EPOCH=0
END_EPOCH=450000
mse_loss2 = nn.L1Loss(reduction='sum')
mse_loss2.cuda()
Tensor = torch.cuda.FloatTensor
dataset = data_deca()   #创建dataset
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  #创建dataloader
writer = SummaryWriter('logdis_transformer_nlp_pre_sub_linear_test')
model = models().cuda()   #初始化transformer模型
ckpt = torch.load(os.path.join('./checkpoint_nlp_pre/model_div_mask_fix_complex_20.pth'))
model.load_state_dict(ckpt)
model.train()
linear_model1 = linear().cuda()   #初始化后处理encoder模型
ckpt = torch.load(os.path.join('./checkpoint_nlp_pre/linear_div_mask_fix_complex_20.pth'))
linear_model1.load_state_dict(ckpt)
linear_model1.train()
optimizer_G = torch.optim.Adam(list(model.parameters()) + list(linear_model1.parameters()), lr=0.0002, betas=(0.5, 0.999))  #优化器
sdtw = SoftDTW(use_cuda=True, gamma=0.1)  # soft dtw loss
iteration_all = 0
for i in tqdm(range(BEGIN_EPOCH, END_EPOCH)):
    loss_list = []
    for iteration,data in enumerate(loader):
        src = data['input'].cuda()   #src bx20x6
        src_true = data['input2'].cuda()  #tgt bx20x6    20个embedding，每个embedding的维度为6
        pre1 = model(src, src_true)   #transformer模型得到初步结果
        pre2 = linear_model1(pre1.permute(1, 2, 0))  #经过后处理encoder得到最终结果
        pre2 = pre2.permute(0, 2, 1)
        lossdata_ori = mse_loss2(pre2, src_true)   #计算L1 loss
        loss_pose1 = mse_loss2(pre2[:,:,[0]], src_true[:,:,[0]])
        loss_pose2 = mse_loss2(pre2[:,:,[1]], src_true[:,:,[1]])
        loss_pose3 = mse_loss2(pre2[:,:,[2]], src_true[:,:,[2]])
        loss_pose4 = mse_loss2(pre2[:,:,[3]], src_true[:,:,[3]])
        loss_pose5 = mse_loss2(pre2[:,:,[4]], src_true[:,:,[4]])
        loss_pose6 = mse_loss2(pre2[:,:,[5]], src_true[:,:,[5]])
        loss_dtw = sdtw(pre2, src_true).sum()  #计算sdtw loss
        loss_G = lossdata_ori + loss_dtw
        optimizer_G.zero_grad()
        loss_G.backward(retain_graph = True)
        optimizer_G.step()
        if iteration%1000==0:  #保存模型参数以及生成结果
            torch.save(model.state_dict(), os.path.join('./checkpoint_nlp_pre', 'model_div_mask_fix_complex_ran.pth'))
            torch.save(linear_model1.state_dict(), os.path.join('./checkpoint_nlp_pre', 'linear_div_mask_fix_complex_ran.pth'))
            save_data = {
                'gen' : pre2,
                'src' : src_true
            }
            torch.save(save_data, './save_image_unet_div2/data_ori_sub_test.pth')
        if iteration%100==0:
            writer.add_scalar('loss pose1', loss_pose1/80, iteration_all)
            writer.add_scalar('loss pose2', loss_pose2/80, iteration_all)
            writer.add_scalar('loss pose3', loss_pose3/80, iteration_all)
            writer.add_scalar('loss pose4', loss_pose4/80, iteration_all)
            writer.add_scalar('loss pose5', loss_pose5/80, iteration_all)
            writer.add_scalar('loss pose6', loss_pose6/80, iteration_all)
            writer.add_scalar('loss dtw', loss_dtw/80, iteration_all)
            writer.add_scalar('loss', lossdata_ori/80, iteration_all)
            iteration_all+=1
